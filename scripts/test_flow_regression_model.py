from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from models.flow_regression_model import FlowRegressionModel
from options.flow_regression_options import TestFlowRegressionOptions
from util.visualizer import Visualizer
from util.loss_buffer import LossBuffer

import util.io as io
import os
import sys
import numpy as np
import tqdm
from collections import OrderedDict

parser = TestFlowRegressionOptions()
opt = parser.parse()
parser.save()
print('load training options.')
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'is_train', 'batch_size', 'which_epoch', 'debug'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)
# create model
model = FlowRegressionModel()
model.initialize(opt)
# create data loader
val_loader = CreateDataLoader(opt, split='test')
# create visualizer
visualizer = Visualizer(opt)
# test
loss_buffer = LossBuffer(size=len(val_loader))
model.output = {}
model.eval()

for i, data in enumerate(tqdm.tqdm(val_loader, desc='Test', total=len(val_loader))):
    if i == len(val_loader):
        break
    model.set_input(data)
    model.test(compute_loss=True)
    loss_buffer.add(model.get_current_errors())

errors = loss_buffer.get_errors()
info = OrderedDict([('model_id', opt.id), ('epoch', opt.which_epoch)])
log_str = visualizer.log(info, errors, log_in_file=False)
print(log_str)
