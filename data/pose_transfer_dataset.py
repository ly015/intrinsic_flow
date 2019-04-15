from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *
import cv2
import numpy as np
import os
import util.io as io
from util import flow_util

class PoseTransferDataset(BaseDataset):
    def name(self):
        return 'PoseTransferDataset'
    
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
        self.to_pil_image = transforms.ToPILImage()
        self.pil_to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.2)
    
    def set_len(self, n):
        self._len = n

    def __len__(self):
        if hasattr(self, '_len') and self._len > 0:
            return self._len
        else:
            return len(self.id_list)
    
    def to_tensor(self, np_data):
        return torch.Tensor(np_data.transpose((2,0,1)))
    
    def read_image(self, sid):
        fn = os.path.join(self.img_dir, sid+'.jpg')
        img = cv2.imread(fn).astype(np.float32) / 255.
        img = img[...,[2,1,0]]
        return img
    
    def read_seg(self, sid):
        fn = os.path.join(self.seg_dir, sid+'.bmp')
        seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)[...,np.newaxis]
        return seg
        
    def read_flow(self, sid1, sid2):
        '''
        Output:
            flow_2to1: (h,w,2) correspondence from image 2 to image 1. corr_2to1[y,x] = [u,v], means image2[y,x] -> image1[v,u]
            vis_2: (h,w) visibility mask of image 2.
                0: human pixel with correspondence
                1: human pixel without correspondece
                2: background pixel
        '''
        fn = os.path.join(self.corr_dir, '%s_%s.corr'%(sid2, sid1))
        corr_2to1, vis_2 = flow_util.read_corr(fn)
        vis_2 = vis_2[...,np.newaxis]
        flow_2to1 = flow_util.corr_to_flow(corr_2to1, vis_2, order='HWC')
        if self.opt.vis_smooth_rate > 0:
            vis_2b = cv2.medianBlur(vis_2, self.opt.vis_smooth_rate)[...,np.newaxis]
            m = (vis_2<2).astype(np.uint8)
            vis_2 = vis_2b*m + vis_2*(1-m)
        return flow_2to1, vis_2
    
    def read_corr(self, sid1, sid2):
        '''
        Output:
            corr_2to1: (h, w, 2)
            vis_2: (h, w)
        '''
        try:
            fn = os.path.join(self.corr_dir, '%s_%s.corr'%(sid2, sid1))
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
        img_1 = self.read_image(sid1)
        img_2 = self.read_image(sid2)
        seg_label_1 = self.read_seg(sid1)
        seg_label_2 = self.read_seg(sid2)
        joint_c_1 = np.array(self.pose_label[sid1])
        joint_c_2 = np.array(self.pose_label[sid2])
        corr_2to1, vis_2 = self.read_corr(sid1, sid2)
        h, w = self.opt.image_size
        ######################
        # augmentation
        ######################
        use_augmentation = self.opt.use_augmentation and self.opt.is_train and self.split == 'train'
        if use_augmentation:
            # apply random shift and scale on img_2
            h, w = self.opt.image_size
            dx = np.random.randint(-self.opt.aug_shiftx_range, self.opt.aug_shiftx_range) if self.opt.aug_shiftx_range > 0 else 0
            dy = np.random.randint(-self.opt.aug_shifty_range, self.opt.aug_shifty_range) if self.opt.aug_shifty_range > 0 else 0
            sc = self.opt.aug_scale_range**(np.random.rand()*2-1)
            M = np.array([[sc, 0, 0.5*h*(1-sc)+dx], [0, sc, 0.5*w*(1-sc)+dy]])

            img_2 = cv2.warpAffine(img_2, M, dsize=(w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            seg_label_2 = cv2.warpAffine(seg_label_2, M, dsize(w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)[...,np.newaxis]
            corr_2to1 = cv2.warpAffine(corr_2to1, M, dsize=(w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            vis_2 = cv2.warpAffine(vis_2, M, dsize=(w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

            v = (d[:,0]>=0) & (d[:,1]>=0) & (d[:,0]<w) & (d[:,1]<h)
            jc = joint_c_2.dot(M[:,0:2].T) + M[:,2:].T
            v_t = (jc[:,0]>=0) & (jc[:,1]>=0) & (jc[:,0]<w) & (jc[:,1]<h)
            v_t = v_t & v
            jc[~v_t,:] = -1
            joint_c_2 = jc
        ######################
        # pack output data
        ######################
        joint_1 = kp_to_map(img_sz=(w,h), kps=joint_c_1, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        joint_2 = kp_to_map(img_sz=(w,h), kps=joint_c_2, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        seg_1 = seg_label_to_map(seg_label_1, nc=self.opt.seg_nc)
        seg_2 = seg_label_to_map(seg_label_2, nc=self.opt.seg_nc)
        flow_2to1 = flow_util.corr_to_flow(corr_2to1, vis_2, order='HWC')
        flow_2to1[...,0] = flow_2to1[...,0].clip(-w,w)
        flow_2to1[...,1] = flow_2to1[...,1].clip(-h,h)
        data = {
            'img_1': self.tensor_normalize_std(self.to_tensor(img_1)),
            'img_2': self.tensor_normalize_std(self.to_tensor(img_2)),
            'joint_c_1': torch.from_numpy(joint_c_1).float(),
            'joint_c_2': torch.from_numpy(joint_c_2).float(),
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
        ######################
        # color jit
        ######################
        if use_augmentation and self.opt.aug_color_jit:
            data['img_1'], data['img_2'] = self.color_jit(data['img_1'], data['img_2'])
        
        return data



        


            


