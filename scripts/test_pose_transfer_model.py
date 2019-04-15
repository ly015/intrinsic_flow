from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from models.pose_transfer_model import PoseTransferModel
from options.pose_transfer_options import TestPoseTransferOptions
from util.visualizer import Visualizer
from util.loss_buffer import LossBuffer

import util.io as io
import os
import sys
import numpy as np
import tqdm
import cv2
import time
from collections import OrderedDict

parser = TestPoseTransferOptions()
opt = parser.parse(display=False)
parser.save()
print('load training options.')
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'is_train', 'batch_size', 'which_epoch', 'n_vis', 'debug'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)

# create model
model = PoseTransferModel()
model.initialize(opt)
# create data loader
val_loader = CreateDataLoader(opt, split=opt.data_split)
# create visualizer
visualizer = Visualizer(opt)

# visualize
if opt.n_vis > 0:
    print('visualizing first %d samples'%opt.n_vis)
    num_vis_batch = min(int(np.ceil(1.0*opt.n_vis/opt.batch_size)), len(val_loader))
    val_loader.dataset.set_len(num_vis_batch*opt.batch_size)
    val_visuals = None
    for i, data in enumerate(tqdm.tqdm(val_loader, desc='Visualize')):
        model.set_input(data)
        model.test(compute_loss=False)
        visuals = model.get_current_visuals()
        if val_visuals is None:
            val_visuals = visuals
        else:
            for name, v in visuals.iteritems():
                val_visuals[name][0] = torch.cat((val_visuals[name][0], v[0]),dim=0)
    
    fn_vis = os.path.join('checkpoints', opt.id, 'vis', 'test_epoch%s.jpg'%opt.which_epoch)
    visualizer.visualize_results(val_visuals, fn_vis)

# test
if opt.n_test_batch != 0:
    val_loader.dataset.set_len(opt.n_test_batch*opt.batch_size)
    loss_buffer = LossBuffer(size=len(val_loader))
    model.output = {}
    if opt.save_output:
        output_dir = os.path.join(model.save_dir, opt.output_dir)
        io.mkdir_if_missing(output_dir)

    total_time = 0
    for i, data in enumerate(tqdm.tqdm(val_loader, desc='Test')):
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
    test_error['sec_per_image'] = total_time/(opt.batch_size*len(val_loader))
    info = OrderedDict([('model_id', opt.id), ('epoch', opt.which_epoch)])
    log_str = visualizer.log(info, test_error, log_in_file=False)
    print(log_str)

