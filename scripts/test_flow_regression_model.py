from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from models.flow_regression_model import FlowRegressionModel
from options.flow_regression_options import TrainFlowRegressionOptions
from util.visualizer import FlowVisualizer
from util.loss_buffer import LossBuffer

import util.io as io
import os
import sys
import numpy as np
import tqdm

opt = TrainFlowRegressionOptions().parse()
print('load training options.')
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'is_train', 'batch_size', 'which_epoch', 'fn_pose', 'silh_dir'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)
# create model
model = FlowRegressionModel()
model.initialize(opt)
# create data loader
val_loader = CreateDataLoader(opt, split='test')
# create visualizer
visualizer = FlowVisualizer(opt)
# test
loss_buffer = LossBuffer(size=len(val_loader))
model.output = {}
model.eval()
nbatch = opt.nbatch if opt.nbatch >= 0 else len(val_loader)

for i, data in enumerate(tqdm.tqdm(val_loader, desc='Test', total=nbatch)):
    if i == nbatch:
        break
    model.set_input(data)
    model.test(compute_loss=True)
    loss_buffer.add(model.get_current_errors())

test_error = loss_buffer.get_errors()
visualizer.print_error(test_error)
