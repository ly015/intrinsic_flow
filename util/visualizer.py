from __future__ import division, print_function

import torch
import torchvision

import os
import time
import util.io as io
import util.image as image
from util.pavi import PaviClient
from options.base_options import opt_to_str
import numpy as np
from collections import OrderedDict
from util import pose_util, flow_util


def seg_to_rgb(seg_map, with_face=False):
    if isinstance(seg_map, np.ndarray):
        if seg_map.ndim == 3:
            seg_map = seg_map[np.newaxis,:]
        seg_map = torch.from_numpy(seg_map.transpose([0,3,1,2]))
    elif isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.cpu()
        if seg_map.dim() == 3:
            seg_map = seg_map.unsqueeze(0)

    if with_face:
        face = seg_map[:,-3::]
        seg_map = seg_map[:,0:-3]

    if seg_map.size(1) > 1:
        seg_map = seg_map.max(dim=1, keepdim=True)[1]
    else:
        seg_map = seg_map.long()

    b,c,h,w = seg_map.size()
    assert c == 1

    cmap = [[73,0,255], [255,0,0], [255,0,219], [255, 219,0], [0,255,146], [0,146,255], [146,0,255], [255,127,80], [0,255,0], [0,0,255],
            [37, 0, 127], [127,0,0], [127,0,109], [127,109,0], [0,127,73], [0,73,127], [73,0, 127], [127, 63, 40], [0,127,0], [0,0,127]]
    cmap = torch.Tensor(cmap)/255.
    cmap = cmap[0:(seg_map.max()+1)]

    rgb_map = cmap[seg_map.view(-1)]
    rgb_map = rgb_map.view(b, h, w, 3)
    rgb_map = rgb_map.transpose(1,3).transpose(2,3)
    rgb_map.sub_(0.5).div_(0.5)

    if with_face:
        face_mask = ((seg_map == 1) | (seg_map == 2)).float()
        rgb_map = rgb_map * (1 - face_mask) + face * face_mask

    return rgb_map

def merge_visual(visuals, kword_params={}):
    imgs = []
    vis_list = []
    for name, (vis, vis_type) in visuals.iteritems():
        vis = vis.cpu()
        if vis_type == 'rgb':
            vis_ = vis
        elif vis_type == 'seg':
            vis_ = seg_to_rgb(vis)
        elif vis_type == 'segf':
            if 'shape_with_face' in kword_params:
                shape_with_face = kword_params['shape_with_face']
            else:
                shape_with_face = False
            vis_ = seg_to_rgb(vis, shape_with_face)
        elif vis_type == 'edge':
            size = list(vis.size())
            size[1] = 3
            vis_ = vis.expand(size)
        elif vis_type == 'color':
            if vis.size(1) == 6:
                vis_ = vis[:,0:3] + vis[:,3::]
        elif vis_type == 'pose':
            # vis = vis.max(dim=1, keepdim=True)[0].expand(vis.size(0), 3, vis.size(2),vis.size(3))
            torch.save(vis, 'test.pth')
            pose_maps = vis.cpu().numpy().transpose(0,2,3,1)
            np_vis_ = np.stack([pose_util.draw_pose_from_map(m, radius=6, threshold=0.)[0] for m in pose_maps])
            vis_ = vis.new(np_vis_.transpose(0,3,1,2))
        imgs.append(vis_)
        vis_list.append(name)

    imgs = torch.stack(imgs, dim=1)
    imgs = imgs.view(imgs.size(0)*imgs.size(1), imgs.size(2), imgs.size(3), imgs.size(4))
    imgs.clamp_(-1.0, 1.0)
    return imgs, vis_list

class Visualizer(object):
    def __init__(self, opt):
        self.opt = opt
        self.exp_dir = os.path.join('./checkpoints', opt.id)
        self.log_file = None
    
    def __del__(self):
        if self.log_file:
            self.log_file.close()
    
    def _open_log_file(self):
        fn = 'train_log.txt' is self.opt.is_train else 'test_log.txt'
        self.log_file = open(os.path.join(self.exp_dir, fn), 'w')
        print(time.ctime(), file=self.log_file)
        print('pytorch version: %s' % torch.__version__, file=self.log_file)
    

    def log(self, info='', errors={}):
        '''
        Save log information into log file
        Input:
            info (dict or str): model id, iteration number, learning rate, etc.
            error (dict): output of loss functions or metrics.
        Output:
            log_str (str) 
        '''
        if self.log_file is None:
            self._open_log_file()
        
        if isinstance(info, str):
            info_str = info
        elif isinstance(info, dict):
            info_str = '  '.join(['{}: {}'.format(k,v) for k, v in info.iteritems()])
        
        error_str = '  '.join(['%s: %.4f'%(k,v) for k, v in errors.iteritems()])
        log_str = '[%s]  %s' %(info_str, error_str)

        print(log_str, file=self.log_file)
        return log_str
    
    def visualize_results(self, )