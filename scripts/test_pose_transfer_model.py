from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from models.pose_transfer_model import PoseTransferModel
from options.pose_transfer_options import TestPoseTransferOptions
from util.visualizer import FlowVisualizer
from util.loss_buffer import LossBuffer

import util.io as io
import os
import sys
import numpy as np
import tqdm
import cv2
import time

parser = TestPoseTransferOptions()
opt = parser.parse()
parser.save()
print('load training options.')
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'is_train', 'batch_size', 'which_epoch', 'n_vis'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)

# create model
model = PoseTransferModel()
model.initialize(opt)
# create data loader
val_loader = CreateDataLoader(opt, split=opt.data_split)
# create visualizer
visualizer = FlowVisualizer(opt)

# visualize
if opt.n_vis > 0:
    print('visualizing first %d samples'%opt.n_vis)
    num_vis_batch = int(np.ceil(1.0*opt.n_vis/opt.batch_size))
    val_visuals = None
    for i, data in enumerate(tqdm.tqdm(val_loader, desc='Visualize', total=num_vis_batch)):
        if i == num_vis_batch:
            break
        model.set_input(data)
        model.test(compute_loss=False)
        visuals = model.get_current_visuals()
        if val_visuals is None:
            val_visuals = visuals
        else:
            for name, v in visuals.iteritems():
                val_visuals[name][0] = torch.cat((val_visuals[name][0], v[0]),dim=0)
    visualizer.visualize_image(epoch=opt.which_epoch, subset='test', visuals=val_visuals)
    print('done visualization.')

# test
n_test_batch = len(val_loader) if opt.n_test_batch == -1 else opt.n_test_batch
if n_test_batch > 0:
    loss_buffer = LossBuffer(size=n_test_batch)
    model.output = {}
    if opt.save_output:
        output_dir = os.path.join(model.save_dir, opt.output_dir)
        io.mkdir_if_missing(output_dir)

    total_time = 0
    for i, data in enumerate(tqdm.tqdm(val_loader, desc='Test', total=n_test_batch)):
        if i == n_test_batch:
            break
        tic = time.time()
        model.set_input(data)
        model.test(compute_loss=True, meas_only=True)
        toc = time.time()
        total_time += (toc-tic)
        loss_buffer.add(model.get_current_errors())
        # save output
        if opt.save_output:
            id_list = model.input['id']
            images = model.output['img_out'].cpu().numpy().transpose(0,2,3,1)
            images = ((images+1.0)*127.5).clip(0,255).astype(np.uint8)
            for (sid1, sid2), img in zip(id_list, images):
                img = img[...,[2,1,0]] # convert to cv2 format
                cv2.imwrite(os.path.join(output_dir, '%s_%s.jpg'%(sid1, sid2)), img)

    test_error = loss_buffer.get_errors()
    test_error['sec per image'] = total_time/(opt.batch_size*n_test_batch)
    visualizer.print_error(test_error)
