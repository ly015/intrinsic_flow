from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *
import cv2
import numpy as np
import os
import util.io as io
from util.image import imshow
from util import flow_util, pose_util
import skimage.transform
import skimage.measure

class FlowDataset(BaseDataset):
    def name(self):
        return 'FlowDataset'
    
    def initialize(self, opt, split):
        self.opt = opt
        self.data_root = opt.data_root
        self.split = split
        #############################
        # load data
        #############################
        print('loading data ...')
        data_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        self.img_dir = os.path.join(opt.data_root, opt.img_dir)
        self.seg_dir = os.path.join(opt.data_root, opt.seg_dir)
        self.corr_dir = os.path.join(opt.data_root, opt.corr_dir)
        self.pose_label = io.load_data(os.path.join(opt.data_root, opt.fn_pose))
        #############################
        # create index list
        #############################
        self.id_list = data_split[split]
        #############################
        # other
        #############################
        if opt.debug:
            self.id_list = self.id_list[0:32]
        self.tensor_normalize_std = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
    def __len__(self):
        return len(self.id_list)
    
    def to_tensor(self, img):
        return torch.Tensor(img.transpose((2,0,1)))
    
    def read_image(self, sid, img_dir=None):
        if img_dir is None:
            img_dir = self.img_dir
        fn = os.path.join(img_dir, sid+'.jpg')
        img = cv2.imread(fn).astype(np.float32) / 255.
        img = img[...,[2,1,0]]
        return img

    
    def read_seg(self, sid):
        fn = os.path.join(self.seg_dir, sid+'.bmp')
        seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)[...,np.newaxis]
        return seg
    

    def read_corr(self, sid1, sid2):
        '''
        Output:
            corr_2to1: (h,w,2) correspondence from image 2 to image 1. corr_2to1[y,x] = [u,v], means image2[y,x] -> image1[v,u]
            mask_2: (h,w) visibility mask of image 2.
                0: human pixel with correspondence
                1: human pixel without correspondece
                2: background pixel
        '''
        fn = os.path.join(self.corr_dir, '%s_%s.corr'%(sid2, sid1))
        corr_2to1, mask_2 = flow_util.read_corr(fn)
        return corr_2to1, mask_2
    
    @staticmethod
    def scale_image(img, scale, pad=0, interp=None):
        h, w = img.shape[0:2]
        nh, nw = int(round(h*scale)), int(round(w*scale))
        if interp is None:
            if isinstance(img.dtype, np.integer):
                interp = cv2.INTER_NEAREST
            else:
                interp = cv2.INTER_LINEAR
        nimg = cv2.resize(img, (nw, nh), interpolation=interp)
        if scale > 1:
            l = int(round((nw-w)/2.0))
            t = int(round((nh-h)/2.0))
            nimg = nimg[t:(t+h),l:(l+w)]
        else:
            l = int(round((w-nw)/2.0))
            t = int(round((h-nh)/2.0))
            nimg_pad = np.ones(img.shape, dtype=img.dtype) * pad
            nimg_pad[t:(t+nh),l:(l+nw)] = nimg
            nimg = nimg_pad
        return nimg
    
    @staticmethod
    def scale_coord(coord, scale, sz):
        w, h = sz
        nh, nw = round(h*scale), round(w*scale)
        coord = coord*scale
        if scale > 1:
            l = int(round((nw-w)/2.0))
            t = int(round((nh-h)/2.0))
            coord[:,0] -= l
            coord[:,1] -= t
        else:
            l = int(round((w-nw)/2.0))
            t = int(round((h-nh)/2.0))
            coord[:,0] += l
            coord[:,1] += t
        return coord
    
    @staticmethod
    def shift_image(img, dx, dy, pad=0):
        nimg = np.ones(img.shape, dtype=img.dtype)*pad
        if dx > 0:
            if dy > 0:
                nimg[dy:,dx:] = img[:-dy,:-dx]
            elif dy == 0:
                nimg[:,dx:] = img[:,:-dx]
            else:
                dy = -dy
                nimg[:-dy,dx:] = img[dy:,:-dx]
        elif dx == 0:
            if dy > 0:
                nimg[dy:,:] = img[:-dy,:]
            elif dy < 0:
                dy = -dy
                nimg[:-dy,:] = img[dy:,:]
        else:
            dx = -dx
            if dy > 0:
                nimg[dy:,:-dx] = img[:-dy,dx:]
            elif dy == 0:
                nimg[:,:-dx] = img[:,dx:]
            else:
                dy = -dy
                nimg[:-dy,:-dx] = img[dy:,dx:]
        return nimg

   
    def __getitem__(self, index):
        
        ######################
        # load data
        ######################
        sid1, sid2 = self.id_list[index]
        img_1 = self.read_image(sid1)
        img_2 = self.read_image(sid2)
        seg_label_1 = self.read_seg(sid1)
        seg_label_2 = self.read_seg(sid2)
        joint_c_1 = np.array(self.pose_label[sid1])
        joint_c_2 = np.array(self.pose_label[sid2])
        corr_2to1, vis_2 = self.read_corr(sid1, sid2)
        vis_2 = vis_2[...,np.newaxis] #(H,W,1) uint8
        flow_2to1 = flow_util.corr_to_flow(corr_2to1, vis_2, order='HWC')
        ######################
        # smooth visiblity map
        ######################
        if self.opt.vis_smooth_rate > 0:
            vis_2b = cv2.medianBlur(vis_2, self.opt.vis_smooth_rate)[...,np.newaxis]
            m = (vis_2<2).astype(np.uint8)
            vis_2 = vis_2b*m + vis_2*(1-m)
        ######################
        # augmentation
        ######################
        if self.opt.is_train and self.split == 'train':
            # flow_2to1, vis_2, img_2, seg_2, joint_c_2 = self.random_scale_and_shift(flow_2to1, vis_2, img_2, seg_2, joint_c_2, silh_2)
            dx = np.random.randint(-self.opt.aug_shiftx_range, self.opt.aug_shiftx_range)
            dy = np.random.randint(-self.opt.aug_shifty_range, self.opt.aug_shifty_range)
            flow_2to1[...,0] -= dx
            flow_2to1[...,1] -= dy
            vis_2 = self.shift_image(vis_2, dx, dy, pad=2)
            flow_2to1 = self.shift_image(flow_2to1, dx, dy, pad=0) * (vis_2<2).astype(np.float32)
            img_2 = self.shift_image(img_2, dx, dy, pad=0)
            seg_label_2 = self.shift_image(seg_label_2, dx, dy, pad=0)

            v = (joint_c_2[:,0] >= 0) & (joint_c_2[:,1] >= 0)
            joint_c_2[v,0] += dx
            joint_c_2[v,1] += dy
            v = (joint_c_2[:,0]<0)|(joint_c_2[:,0]>self.opt.image_size[1])|(joint_c_2[:,1]<0)|(joint_c_2[:,1]>self.opt.image_size[0])
            joint_c_2[v,:]=-1

        ######################
        # create pose representation
        ######################
        joint_1 = kp_to_map(img_sz=(img_1.shape[1], img_1.shape[0]), kps=joint_c_1, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        joint_2 = kp_to_map(img_sz=(img_2.shape[1], img_2.shape[0]), kps=joint_c_2, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        ######################
        # create seg representation
        ######################
        seg_1 = seg_label_to_map(seg_label_1, nc=self.opt.seg_nc, bin_size=self.opt.seg_bin_size)
        seg_2 = seg_label_to_map(seg_label_2, nc=self.opt.seg_nc, bin_size=self.opt.seg_bin_size)
        ######################
        # pack output
        ######################
        data = {
            'img_1': self.tensor_normalize_std(self.to_tensor(img_1)),
            'img_2': self.tensor_normalize_std(self.to_tensor(img_2)),
            'joint_c_1': torch.Tensor(joint_c_1),
            'joint_c_2': torch.Tensor(joint_c_2),
            'joint_1': self.to_tensor(joint_1),
            'joint_2': self.to_tensor(joint_2),
            'seg_label_1': self.to_tensor(seg_label_1),
            'seg_label_2': self.to_tensor(seg_label_2),
            'seg_1': self.to_tensor(seg_1),
            'seg_2': self.to_tensor(seg_2),
            'flow_2to1': self.to_tensor(flow_2to1),
            'vis_2': self.to_tensor(vis_2),
            'id_1': sid1,
            'id_2': sid2
        }
        return data






