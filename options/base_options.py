from __future__ import division, print_function

import torch
import argparse
import os
import util.io as io


def opt_to_str(opt):
        return '\n'.join(['%s: %s' % (str(k), str(v)) for k, v in sorted(vars(opt).items())])

    

class BaseOptions(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.opt = None
    
    def initialize(self):
        parser = self.parser
        # basic experiment options
        parser.add_argument('--id', type = str, default = 'default', help = 'experiment ID. the experiment dir will be set as "./checkpoint/id/"')
        parser.add_argument('--gpu_ids', type = str, default = '0', help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
 
        self.initialized = True
        

    def auto_set(self):
        '''
        options that will be automatically set
        '''
        # set training status
        self.opt.is_train = self.is_train

        # set gpu_ids
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            g_id = int(str_id)
            if g_id >= 0:
                self.opt.gpu_ids.append(g_id)
        # set gpu devices
        if len(self.opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.opt.gpu_ids])
            torch.cuda.set_device(0)
            


    def parse(self, ord_str = None, display = True):
        '''
        Parse option from terminal command string. If ord_str is given, parse option from it instead.
        '''

        if not self.initialized:
            self.initialize()

        if ord_str is None:
            self.opt = self.parser.parse_args()
        else:
            ord_list = ord_str.split()
            self.opt = self.parser.parse_args(ord_list)

        self.auto_set()
        # display options
        if display:
            print('------------ Options -------------')
            for k, v in sorted(vars(self.opt).items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
        return self.opt
    
    def save(self, fn=None):
        if self.opt is None:
            raise Exception("parse options before saving!")
        if fn is None:
            expr_dir = os.path.join('checkpoints', self.opt.id)
            io.mkdir_if_missing(expr_dir)
            if self.opt.is_train:
                fn = os.path.join(expr_dir, 'train_opt.json')
            else:
                fn = os.path.join(expr_dir, 'test_opt.json')
        io.save_json(vars(self.opt), fn)

    def load(self, fn):
        args = io.load_json(fn)
        return argparse.Namespace(**args)
