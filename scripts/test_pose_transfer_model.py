from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from models.cascade_transfer_model import CascadeTransferModel
from options.cascade_transfer_options import TestCascadeTransferOptions
from misc.visualizer import FlowVisualizer
from misc.loss_buffer import LossBuffer

import util.io as io
import os
import sys
import numpy as np
import tqdm
import cv2
import time

opt = TestCascadeTransferOptions().parse()
print('load training options.')
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'is_train', 'which_epoch', 'fn_joint', 'silh_dir', 'fn_split', 'data_item_list'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)

# create model
model = CascadeTransferModel()
model.initialize(opt)
# create data loader
val_loader = CreateDataLoader(opt, split=opt.data_split)
# create visualizer
visualizer = FlowVisualizer(opt)

# visualize
if opt.visualize:
    if opt.test_nvis > 0:
        print('visualizing first %d samples'%opt.test_nvis)
        num_vis_batch = int(np.ceil(1.0*opt.test_nvis/opt.batch_size))
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

if opt.visualize_batch:
    for i, data in enumerate(tqdm.tqdm(val_loader, desc='Visualize Batches', total=opt.test_batch_nvis)):
        if i == opt.test_batch_nvis:
            break
        model.set_input(data)
        model.test(compute_loss=False)
        visuals = model.get_current_visuals()
        visualizer.visualize_image(epoch='%s_%d'%(opt.which_epoch, i), subset='test', visuals=visuals)
    print('done batch visualization')

if opt.nbatch == 0:
    exit(1)

# test
loss_buffer = LossBuffer(size=len(val_loader))
model.output = {}
nbatch = opt.nbatch if opt.nbatch > 0 else len(val_loader)
if opt.save_output:
    output_dir = os.path.join(model.save_dir, opt.output_dir)
    io.mkdir_if_missing(output_dir)

total_time = 0
for i, data in enumerate(tqdm.tqdm(val_loader, desc='Test', total=nbatch)):
    if i == nbatch:
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
test_error['sec per image'] = total_time/(opt.batch_size*nbatch)
visualizer.print_error(test_error)
