from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *
import cv2
import numpy as np
import os
import util.io as io
from util import flow_util

class GeneralPairDataset(BaseDataset):
    def name(self):
        return 'GeneralPairDataset'
    
    def initialize(self, opt, split):
        self.opt = opt
        self.data_root = opt.data_root
        self.split = split
        #############################
        # set path / load label
        #############################
        data_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        self.img_dir = os.path.join(opt.data_root, opt.img_dir)
        self.seg_dir = os.path.join(opt.data_root, opt.seg_dir)
        self.silh_dir = os.path.join(opt.data_root, opt.silh_dir)
        self.corr_dir = os.path.join(opt.data_root, opt.corr_dir)
        self.joint_label = io.load_data(os.path.join(opt.data_root, opt.fn_joint))
        self.view_type = io.load_json(os.path.join(opt.data_root, opt.fn_view))
        assert opt.dataset_name != 'dfm_aug', 'GeneralDataset does not support dfm_aug now'
        #############################
        # set item list
        #############################
        # Todo: this should be set in option.auto_set()
        if self.opt.data_item_list is None:
            self.opt.data_item_list = ['img', 'seg', 'silh', 'joint', 'flow', 'view']

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
        self.to_pil_image = transforms.ToPILImage()
        self.pil_to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.2)

    def __len__(self):
        return len(self.id_list)
    
    def to_tensor(self, np_data):
        return torch.Tensor(np_data.transpose((2,0,1)))
    
    def read_image(self, sid, img_dir=None):
        if img_dir is None:
            img_dir = self.img_dir
        fn = os.path.join(img_dir, sid+'.jpg')
        img = cv2.imread(fn).astype(np.float32) / 255.
        img = img[...,[2,1,0]]
        return img
    
    def read_seg(self, sid, seg_dir=None):
        if seg_dir is None:
            seg_dir = self.seg_dir
        fn = os.path.join(seg_dir, sid+'.bmp')
        seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)[...,np.newaxis]
        return seg
    
    def read_flow(self, sid1, sid2, corr_dir=None):
        '''
        Output:
            flow_2to1: (h,w,2) correspondence from image 2 to image 1. corr_2to1[y,x] = [u,v], means image2[y,x] -> image1[v,u]
            vis_2: (h,w) visibility mask of image 2.
                0: human pixel with correspondence
                1: human pixel without correspondece
                2: background pixel
        '''
        if corr_dir is None:
            corr_dir = self.corr_dir
        fn = os.path.join(corr_dir, '%s_%s.corr'%(sid2, sid1))
        corr_2to1, vis_2 = flow_util.read_corr(fn)
        vis_2 = vis_2[...,np.newaxis]
        flow_2to1 = flow_util.corr_to_flow(corr_2to1, vis_2, order='HWC')
        if self.opt.vis_smooth_rate > 0:
            vis_2b = cv2.medianBlur(vis_2, self.opt.vis_smooth_rate)[...,np.newaxis]
            m = (vis_2<2).astype(np.uint8)
            vis_2 = vis_2b*m + vis_2*(1-m)
        return flow_2to1, vis_2
    
    def read_corr(self, sid1, sid2, corr_dir=None):
        '''
        Output:
            corr_2to1: (h, w, 2)
            vis_2: (h, w)
        '''
        try:
            if corr_dir is None:
                corr_dir = self.corr_dir
            fn = os.path.join(corr_dir, '%s_%s.corr'%(sid2, sid1))
            corr_2to1, vis_2 = flow_util.read_corr(fn)
            vis_2 = vis_2[...,np.newaxis]
            if self.opt.vis_smooth_rate > 0:
                vis_2b = cv2.medianBlur(vis_2, self.opt.vis_smooth_rate)[...,np.newaxis]
                m = (vis_2<2).astype(np.uint8)
                vis_2 = vis_2b*m + vis_2*(1-m)
            return corr_2to1, vis_2
        except:
            h, w = self.opt.image_size
            return np.zeros((h,w,2),dtype=np.float32), np.ones((h,w,1), dtype=np.float32)*2

    
    def shift_image(self, img, dx, dy, pad=0):
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
    
    def shift_flow(self, flow, dx, dy, vis=None):
        flow[...,0] -= dx
        flow[...,1] -= dy
        flow = self.shift_image(flow, dx, dy, pad=0)
        if vis is None:
            return flow
        else:
            vis = self.shift_image(vis, dx, dy, pad=2)
            flow = flow*(vis<2).astype(np.float32)
            return flow, vis

    
    def shift_coords(self, coords, dx, dy):
        v = (coords[:,0]>=0) & (coords[:,1]>=0)
        coords[v,0] += dx
        coords[v,1] += dy
        coords[(coords[:,0]<0)|(coords[:,0]>self.opt.image_size[1])|(coords[:,1]<0)|(coords[:,1]>self.opt.image_size[0])] = -1
        return coords

    def color_jit(self, img_1, img_2):
        '''
        Input:
            img_1, img_2: Tensor CHW
        Output:
            img_1, img_2: Tensor CHW
        '''
        w1 = img_1.shape[2]
        img = torch.cat((img_1, img_2), dim=2)
        img = self.to_pil_image(img.add_(1).div_(2))
        img = self.color_jitter(img)
        img = self.pil_to_tensor(img).mul_(2).sub_(1)
        return img[:,:,:w1], img[:,:,w1:]
    
    def __getitem__(self, index):
        sid1, sid2 = self.id_list[index]
        ######################
        # load data
        ######################
        data = {}
        for item_name in self.opt.data_item_list:
            if item_name == 'img':
                data['img_1'] = self.read_image(sid1)
                data['img_2'] = self.read_image(sid2)
            elif item_name == 'seg':
                data['seg_label_1'] = self.read_seg(sid1)
                data['seg_label_2'] = self.read_seg(sid2)
            elif item_name == 'silh':
                data['silh_label_1'] = self.read_seg(sid1, self.silh_dir)
                data['silh_label_2'] = self.read_seg(sid2, self.silh_dir)
            elif item_name == 'flow':
                corr_2to1, vis_2 = self.read_corr(sid1, sid2) # use corr (instead of flow) to simplify augmentation
                data['corr_2to1'] = (corr_2to1, vis_2)
            elif item_name == 'joint':
                data['joint_c_1'] = np.array(self.joint_label[sid1])
                data['joint_c_2'] = np.array(self.joint_label[sid2])
        ######################
        # augmentation
        ######################
        use_augmentation = self.opt.use_augmentation and self.opt.is_train and self.split == 'train'
        if use_augmentation:
            dx = np.random.randint(-self.opt.aug_shiftx_range, self.opt.aug_shiftx_range) if self.opt.aug_shiftx_range > 0 else 0
            dy = np.random.randint(-self.opt.aug_shifty_range, self.opt.aug_shifty_range) if self.opt.aug_shifty_range > 0 else 0
            sc = self.opt.aug_scale_range**(np.random.rand()*2-1)
            M = np.array([[sc, 0, 0.5*self.opt.image_size[0]*(1-sc)+dx], [0, sc, 0.5*self.opt.image_size[1]*(1-sc)+dy]])
            for item, d in data.iteritems():
                # random shift and scale image_2
                h, w = self.opt.image_size
                if item == 'img_2':
                    data[item] = cv2.warpAffine(d, M, dsize=(w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                elif item in {'seg_label_2', 'silh_label_2'}:
                    data[item] = cv2.warpAffine(d, M, dsize=(w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)[...,np.newaxis]
                elif item == 'corr_2to1':
                    corr, vis = d
                    corr = cv2.warpAffine(corr, M, dsize=(w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    vis = cv2.warpAffine(vis, M, dsize=(w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)[...,np.newaxis]
                    data[item] = (corr, vis)
                elif item == 'joint_c_2':
                    valid = (d[:,0]>=0) & (d[:,1]>=0) & (d[:,0]<self.opt.image_size[1]) & (d[:,1]<self.opt.image_size[0])
                    d_t = d.dot(M[:,0:2].T) + M[:,2:].T
                    valid_t = (d_t[:,0]>=0) & (d_t[:,1]>=0) & (d_t[:,0]<self.opt.image_size[1]) & (d_t[:,1]<self.opt.image_size[0])
                    valid_t = valid_t & valid
                    d_t[~valid_t,:] = -1
                    data[item] = d_t
        ######################
        # pack output data
        ######################
        data_out = {}
        for item, d in data.iteritems():
            if item in {'img_1', 'img_2'}:
                data_out[item] = self.tensor_normalize_std(self.to_tensor(d))
            elif item in {'seg_label_1', 'seg_label_2'}:
                data_out[item] = self.to_tensor(d)
                data_out[item.replace('_label','')] = self.to_tensor(seg_label_to_map(d, nc=self.opt.seg_nc, bin_size=self.opt.seg_bin_size))
            elif item in {'silh_label_1', 'silh_label_2'}:
                data_out[item] = self.to_tensor(d)
                data_out[item.replace('_label', '')] = self.to_tensor(seg_label_to_map(d, nc=self.opt.silh_nc))
            elif item in {'joint_c_1', 'joint_c_2'}:
                data_out[item] = torch.Tensor(d)
                prob_map = kp_to_map(img_sz=(self.opt.image_size[1], self.opt.image_size[0]), label=d, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
                data_out[item.replace('_c', '')] = self.to_tensor(prob_map)
            elif item == 'corr_2to1':
                corr, vis = d
                flow = flow_util.corr_to_flow(corr, vis, order='HWC')
                flow[...,0] = flow[...,0].clip(-self.opt.image_size[1], self.opt.image_size[1]) # remove outliers
                flow[...,1] = flow[...,1].clip(-self.opt.image_size[0], self.opt.image_size[0]) # remove outliers
                data_out['flow_2to1'] = self.to_tensor(flow)
                data_out['vis_2to1'] = self.to_tensor(vis)

        if use_augmentation and self.opt.aug_color_jit:
            data_out['img_1'], data_out['img_2'] = self.color_jit(data_out['img_1'], data_out['img_2'])

        data_out['id_1'] = sid1
        data_out['id_2'] = sid2
        return data_out

