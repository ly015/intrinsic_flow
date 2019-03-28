import json
import cPickle
import os
import shutil

#io functions of SCRC
def load_str_list(filename, end = '\n'):
    with open(filename, 'r') as f:
        str_list = f.readlines()
    str_list = [s[:-len(end)] for s in str_list]
    return str_list

def save_str_list(str_list, filename, end = '\n'):
    str_list = [s+end for s in str_list]
    with open(filename, 'w') as f:
        f.writelines(str_list)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(json_obj, filename):
    with open(filename, 'w') as f:
        # json.dump(json_obj, f, separators=(',\n', ':\n'))
        json.dump(json_obj, f, indent = 0, separators = (',', ': '))

def mkdir_if_missing(output_dir):
  """
  def mkdir_if_missing(output_dir)
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def save_data(data, filename):
    with open(filename, 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

def load_data(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data

def copy(fn_src, fn_tar):
    shutil.copyfile(fn_src, fn_tar)


