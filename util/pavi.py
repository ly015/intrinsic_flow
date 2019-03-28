import os
import time
from datetime import datetime
from getpass import getuser
from socket import gethostname
from threading import Thread

import requests
from six.moves.queue import Empty, Queue


class PaviClient(object):

    def __init__(self, url='http://pavi.parrotsdnn.org/log/', username=None, password=None, instance_id=None):
        self.url = url
        self.instance_id = instance_id
        self.log_queue = None
        if username is not None:
            self.username = str(username)
        else:
            username = os.getenv('PAVI_USERNAME')
            if username:
                self.username = username
            else:
                raise ValueError('Pavi username is not specified')
        if password is not None:
            self.password = str(password)
        else:
            password = os.getenv('PAVI_PASSWORD')
            if password:
                self.password = password
            else:
                raise ValueError('Pavi password is not specified')

    def connect(self, model_name, work_dir=None, info=dict(), timeout=5):
        print('connecting pavi service {}...'.format(self.url))
        post_data = dict(
            time=str(datetime.now()),
            username=self.username,
            password=self.password,
            instance_id=self.instance_id,
            model=model_name,
            work_dir=os.path.abspath(work_dir) if work_dir else '',
            session_file=info.get('session_file', ''),
            session_text=info.get('session_text', ''),
            model_text=info.get('model_text', ''),
            device='{}@{}'.format(getuser(), gethostname()))
        try:
            response = requests.post(self.url, json=post_data, timeout=timeout)
        except Exception as ex:
            print('fail to connect to pavi service: {}'.format(ex))
        else:
            if response.status_code == 200:
                self.instance_id = response.text
                print('pavi service connected, instance_id: {}'.format(
                    self.instance_id))
                self.log_queue = Queue()
                self.log_thread = Thread(target=self.post_log, args=(3, 3, 3))
                self.log_thread.daemon = True
                self.log_thread.start()
                return True
            else:
                print('fail to connect to pavi service, status code: '
                      '%d, err message: %s', response.status_code,
                      response.reason)
        return False

    def log(self, phase, iter_num, outputs):
        if self.log_queue is not None:
            logs = {
                'time': str(datetime.now()),
                'instance_id': self.instance_id,
                'flow_id': phase,
                'iter_num': iter_num,
                'outputs': outputs,
                'msg': ''
            }

            # print(logs)
            self.log_queue.put(logs)

    def post_log(self, max_retry, queue_timeout, req_timeout):
        while True:
            try:
                log = self.log_queue.get(timeout=queue_timeout)
            except Empty:
                time.sleep(1)
            except Exception as ex:
                print('fail to get logs from queue: {}'.format(ex))
            else:
                retry = 0
                while retry < max_retry:
                    try:
                        response = requests.post(
                            self.url, json=log, timeout=req_timeout)
                    except Exception as ex:
                        retry += 1
                        print('error when posting logs to pavi: {}'.format(ex))
                    else:
                        status_code = response.status_code
                        if status_code == 200:
                            break
                        else:
                            print('unexpected status code: %d, err msg: %s' %
                                  (status_code,response.reason))
                            retry += 1
                if retry == max_retry:
                    print('fail to send logs of iteration %d' % log['iter_num'])
