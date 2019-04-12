from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import networks
from base_model import BaseModel
from collections import OrderedDict

import util.io as io
import os
import argparse

class PoseTransferModel(BaseModel):
    '''
    Pose transfer framework that cascade a 3d-flow module and a generation module.
    '''
    def name(self):
        return 'PoseTransferModel'
    
    def initialize(self, opt):
        super(PoseTransferModel, self).initialize(opt)
        ###################################
        # define generator
        ###################################
        if opt.which_model_G == 'unet':
            self.netG = networks.UnetGenerator(
                input_nc = self.get_tensor_dim('+'.join([opt.G_appearance_type, opt.G_pose_type])),
                output_nc = 3,
                nf = opt.G_nf,
                max_nf = opt.G_max_nf,
                num_scales = opt.G_n_scale,
                n_residual_blocks = 2,
                norm = opt.G_norm,
                activation = nn.LeakyReLU(0.1) if opt.G_activation=='leaky_relu' else nn.ReLU(),
                use_dropout = False,
                gpu_ids = opt.gpu_ids
            )
        elif opt.which_model_G == 'dual_unet':
            self.netG = networks.DualUnetGenerator(
                pose_nc = self.get_tensor_dim(opt.G_pose_type),
                appearance_nc = self.get_tensor_dim(opt.G_appearance_type),
                output_nc = 3,
                aux_output_nc = [],
                nf = opt.G_nf,
                max_nf = opt.G_max_nf,
                num_scales = opt.G_n_scale,
                num_warp_scales = opt.G_n_warp_scale,
                n_residual_blocks = 2,
                norm = opt.G_norm,
                vis_mode = opt.G_vis_mode,
                activation = nn.LeakyReLU(0.1) if opt.G_activation=='leaky_relu' else nn.ReLU(),
                use_dropout = False,
                no_end_norm = opt.G_no_end_norm,
                gpu_ids = opt.gpu_ids,
            )
        if opt.gpu_ids:
            self.netG.cuda()
        networks.init_weights(self.netG, init_type=opt.init_type)
        ###################################
        # define external pixel warper
        ###################################
        if opt.G_pix_warp:
            pix_warp_n_scale = opt.G_n_scale
            self.netPW = networks.UnetGenerator_MultiOutput(
                input_nc = self.get_tensor_dim(opt.G_pix_warp_input_type),
                output_nc = [1], # only use one output branch (weight mask)
                nf = 32,
                max_nf = 128,
                num_scales = pix_warp_n_scale,
                n_residual_blocks = 2,
                norm = opt.G_norm,
                activation = nn.ReLU(False),
                use_dropout = False,
                gpu_ids = opt.gpu_ids
            )
            if opt.gpu_ids:
                self.netPW.cuda()
            networks.init_weights(self.netPW, init_type=opt.init_type)
        ###################################
        # define discriminator
        ###################################
        self.use_gan = self.is_train and self.opt.loss_weight_gan > 0
        if self.use_gan:
            self.netD = networks.NLayerDiscriminator(
                input_nc = self.get_tensor_dim(opt.D_input_type_real),
                ndf = opt.D_nf,
                n_layers = opt.D_n_layers,
                use_sigmoid = (opt.gan_type == 'dcgan'),
                output_bias = True,
                gpu_ids = opt.gpu_ids,
            )
            if opt.gpu_ids:
                self.netD.cuda()
            networks.init_weights(self.netD, init_type=opt.init_type)
        ###################################
        # load optical flow model
        ###################################
        if opt.flow_on_the_fly:
            self.netF = load_flow_network(opt.pretrained_flow_id, opt.pretrained_flow_epoch, opt.gpu_ids)
            self.netF.eval()
            if opt.gpu_ids:
                self.netF.cuda()
        ###################################
        # loss and optimizers
        ###################################
        self.crit_psnr = networks.PSNR()
        self.crit_ssim = networks.SSIM()

        if self.is_train:
            self.crit_vgg = networks.VGGLoss(opt.gpu_ids, shifted_style=opt.shifted_style_loss, content_weights=opt.vgg_content_weights)
            if opt.G_pix_warp:
                # only optimze netPW
                self.optim = torch.optim.Adam(self.netPW.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
            else:
                self.optim = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
            self.optimizers = [self.optim]
            if self.use_gan:
                self.crit_gan = networks.GANLoss(use_lsgan=(opt.gan_type=='lsgan'))
                if self.gpu_ids:
                    self.crit_gan.cuda()
                self.optim_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay_D)
                self.optimizers += [self.optim_D]
            
        ###################################
        # load trained model
        ###################################
        if not self.is_train:
            # load trained model for testing
            self.load_network(self.netG, 'netG', opt.which_epoch)
            if opt.G_pix_warp:
                self.load_network(self.netPW, 'netPW', opt.which_epoch)
        elif opt.pretrained_G_id is not None:
            # load pretrained network
            self.load_network(self.netG, 'netG', opt.pretrained_G_epoch, opt.pretrained_G_id)
        elif opt.resume_train:
            # resume training
            self.load_network(self.netG, 'netG', opt.which_epoch)
            self.load_optim(self.optim, 'optim', opt.which_epoch)
            if self.use_gan:
                self.load_network(self.netD, 'netD', opt.which_epoch)
                self.load_optim(self.optim_D, 'optim_D', opt.which_epoch)
            if opt.G_pix_warp:
                self.load_network(self.netPW, 'netPW', opt.which_epoch)
        ###################################
        # schedulers
        ###################################
        if self.is_train:
            self.schedulers = []
            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))
        
    
    def set_input(self, data):
        self.input_list = [
            'img_1',
            'img_2',
            'joint_1',
            'joint_2',
            'joint_c_1',
            'joint_c_2',
            'seg_1',
            'seg_2',
            'seg_label_1',
            'seg_label_2',
            'flow_2to1',
            'vis_2',            
        ]
        for item in self.input_list:
            self.input[item] = self.Tensor(data[item].size()).copy_(data[item])
        
        self.input['id'] = zip(data['id_1'], data['id_2'])
    
    def forward(self, test=False):
        # generate flow
        flow_scale = 20.
        if self.opt.flow_on_the_fly or 'flow' in self.opt.data_item_list:
            if self.opt.flow_on_the_fly:
                with torch.no_grad():
                    input_F = self.get_tensor(self.opt.F_input_type)
                    flow_out, vis_out, _, _ = self.netF(input_F)
                    self.output['vis_out'] = vis_out.argmax(dim=1, keepdim=True).float()
                    self.output['mask_out'] = (self.output['vis_out']<2).float()
                    self.output['flow_out'] = flow_out * flow_scale * self.output['mask_out']
                    self.output['flow_tar'] = self.input['flow_2to1']
                    self.output['vis_tar'] = self.input['vis_2']
                    self.output['mask_tar'] = (self.output['vis_tar']<2).float()
            else:
                self.output['flow_out'] = self.input['flow_2to1']
                self.output['vis_out'] = self.input['vis_2']
                self.output['mask_out'] = (self.output['vis_out']<2).float()
                self.output['flow_tar'] = self.output['flow_out']
                self.output['vis_tar'] = self.output['vis_out']
                self.output['maks_tar'] = self.output['mask_out']
            bsz, _, h, w = self.output['vis_out'].size()
            self.output['vismap_out'] = self.output['vis_out'].new(bsz,3,h,w).scatter_(dim=1, index=self.output['vis_out'].long(), value=1)
            self.output['vismap_tar'] = self.output['vis_tar'].new(bsz,3,h,w).scatter_(dim=1, index=self.output['vis_tar'].long(), value=1)
        
        # warp image
        self.output['img_warp'] = networks.warp_acc_flow(self.input['img_1'], self.output['flow_out'], mask=self.output['mask_out'])
       
        # generate image
        if self.opt.which_model_G == 'unet':
            input_G = self.get_tensor('+'.join([self.opt.G_appearance_type, self.opt.G_pose_type]))
            out = self.netG(input_G)
            self.output['img_out'] = F.tanh(out)
        elif self.opt.which_model_G == 'dual_unet':
            input_G_pose = self.get_tensor(self.opt.G_pose_type)
            input_G_appearance = self.get_tensor(self.opt.G_appearance_type)
            flow_in, vis_in = (self.output['flow_out'], self.output['vis_out']) if self.opt.G_feat_warp else (None, None)

            if not self.opt.G_pix_warp:
                out = self.netG(input_G_pose, input_G_appearance, flow_in, vis_in)
                self.output['img_out'] = F.tanh(out)
            else:
                with torch.no_grad():
                    out = self.netG(input_G_pose, input_G_appearance, flow_in, vis_in)
                self.output['img_out_G'] = F.tanh(out)
                pw_out = self.netPW(self.get_tensor(self.opt.G_pix_warp_input_type))
                self.output['pix_mask'] = F.sigmoid(pw_out[0])
                if self.opt.G_pix_warp_detach:
                    self.output['img_out'] = self.output['img_warp']*self.output['pix_mask'] + self.output['img_out_G'].detach()*(1-self.output['pix_mask'])
                else:
                    self.output['img_out'] = self.output['img_warp']*self.output['pix_mask'] + self.output['img_out_G'] * (1-self.output['pix_mask'])
        self.output['img_tar'] = self.input['img_2']
        self.output['seg_1'] = self.input['seg_1']
        self.output['seg_2'] = self.input['seg_2']
            
    
    def test(self, compute_loss=False, meas_only=False):
        ''' meas_only: only compute measurements (psrn, ssim) when computing loss'''
        with torch.no_grad():
            self.forward(test=True)
            if compute_loss:
                assert self.is_train or meas_only, 'when is_train is False, meas_only must be True'
                self.compute_loss(meas_only=meas_only, compute_ssim=True)
    
    def compute_loss(self, meas_only=False, compute_ssim=False):
        '''compute_ssim: set True to compute ssim (time consuming)'''
        ##############################
        # measurements
        ##############################
        self.output['PSNR'] = self.crit_psnr(self.output['img_out'], self.output['img_tar'])
        if compute_ssim:
            self.output['SSIM'] = self.crit_ssim(self.output['img_out'], self.output['img_tar'])
        if meas_only:
            return
        ##############################
        # losses
        ##############################
        # L1
        self.output['loss_l1'] = F.l1_loss(self.output['img_out'], self.output['img_tar'])
        # Content (Perceptual)
        self.output['loss_content'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], loss_type='content')
        # Style
        if self.opt.loss_weight_style > 0:
            self.output['loss_style'] = self.opt.loss_weight_pix_warp * self.crit_vgg(self.output['img_out'], self.output['img_tar'], loss_type='style')
        # GAN
        if self.use_gan:
            input_D = self.get_tensor(self.opt.D_input_type_fake)
            self.output['loss_G'] = self.crit_gan(self.netD(input_D), True)
   
    def backward(self, check_grad=False):
        if not check_grad:
            loss = 0
            loss += self.output['loss_l1'] * self.opt.loss_weight_l1
            loss += self.output['loss_content'] * self.opt.loss_weight_content
            if self.opt.loss_weight_style > 0:
                loss += self.output['loss_style'] * self.opt.loss_weight_style
            if self.use_gan:
                loss += self.output['loss_G'] * self.opt.loss_weight_gan
            loss.backward()
        else:
            net_to_check = self.netG if not self.opt.G_pix_warp else self.netPW
            with networks.CalcGradNorm(net_to_check) as cgn:
                (self.output['loss_l1'] * self.opt.loss_weight_l1).backward(retain_graph=True)
                self.output['grad_l1'] = cgn.get_grad_norm()
                (self.output['loss_content'] * self.opt.loss_weight_content).backward(retain_graph=True)
                self.output['grad_content'] = cgn.get_grad_norm()
                if self.opt.loss_weight_style > 0:
                    (self.output['loss_style'] * self.opt.loss_weight_style).backward(retain_graph=True)
                    self.output['grad_style'] = cgn.get_grad_norm()
                if self.use_gan:
                    (self.output['loss_G'] * self.opt.loss_weight_gan).backward(retain_graph=True)
                    self.output['grad_G'] = cgn.get_grad_norm()
    
    def backward_D(self):
        input_D_real = self.get_tensor(self.opt.D_input_type_real).detach()
        input_D_fake = self.get_tensor(self.opt.D_input_type_fake).detach()
        self.output['loss_D'] = 0.5 * (self.crit_gan(self.netD(input_D_real), True) +\
                                self.crit_gan(self.netD(input_D_fake), False))
        (self.output['loss_D'] * self.opt.loss_weight_gan).backward()
    
    def optimize_parameters(self, check_grad=False):
        self.output = {}
        # forward
        self.forward()
        # optim netD
        if self.use_gan:
            self.optim_D.zero_grad()
            self.backward_D()
            self.optim_D.step()
        # optim netG
        self.optim.zero_grad()
        self.compute_loss()
        self.backward(check_grad)
        self.optim.step()
    
    def get_tensor_dim(self, tensor_type):
        dim = 0
        tensor_items = tensor_type.split('+')
        for item in tensor_items:
            if item in {'img_1', 'img_2', 'img_out', 'img_warp', 'img_out_G'}:
                dim += 3
            elif item in {'seg_1', 'seg_2'}:
                dim += self.opt.seg_nc
            elif item in {'joint_1', 'joint_2'}:
                dim += self.opt.joint_nc
            elif item  in {'flow_out', 'flow_tar'}:
                dim += 2
            elif item  in {'vis_out', 'vis_tar'}:
                dim += 1
            elif item  in {'vismap_out', 'vismap_tar'}:
                dim += 3
            else:
                raise Exception('invalid tensor_type: %s'%item)
        return dim
    
    def get_tensor(self, tensor_type):
        tensor = []
        tensor_items = tensor_type.split('+')
        for item in tensor_items:
            if item == 'img_1':
                tensor.append(self.input['img_1'])
            elif item == 'img_2':
                tensor.append(self.input['img_2'])
            elif item == 'img_out':
                tensor.append(self.output['img_out'])
            elif item == 'img_out_G':
                tensor.append(self.output['img_out_G'])
            elif item == 'img_warp':
                tensor.append(self.output['img_warp'])
            elif item == 'seg_1':
                tensor.append(self.input['seg_1'])
            elif item == 'seg_2':
                tensor.append(self.input['seg_2'])
            elif item == 'joint_1':
                tensor.append(self.input['joint_1'])
            elif item == 'joint_2':
                tensor.append(self.input['joint_2'])
            elif item == 'flow_out':
                tensor.append(self.output['flow_out'])
            elif item == 'flow_tar':
                tensor.append(self.output['flow_tar'])
            elif item == 'vis_out':
                tensor.append(self.output['vis_out'])
            elif item == 'vis_tar':
                tensor.append(self.output['vis_tar'])
            elif item == 'vismap_out':
                tensor.append(self.output['vismap_out'])
            elif item == 'vismap_tar':
                tensor.append(self.output['vismap_tar'])
            else:
                raise Exception('invalid tensor_type: %s'%item)
        tensor = torch.cat(tensor, dim=1)
        return tensor
    
    def get_current_errors(self):
        error_list = [
            'PSNR',
            'SSIM',
            'loss_l1',
            'loss_content',
            'loss_style',
            'loss_G',
            'loss_D',
            'grad_l1',
            'grad_content',
            'grad_style',
            'grad_G',
        ]
        errors = OrderedDict()
        for item in error_list:
            if item in self.output:
                errors[item] = self.output[item].item()
        return errors
    
    def get_current_visuals(self):
        visual_items = [
            ('img_1', [self.input['img_1'].data.cpu(), 'rgb']),
            ('joint_1', [self.input['joint_1'].data.cpu(), 'pose']),
            ('seg_1', [self.output['seg_1'].data.cpu(), 'seg']),
            ('joint_2', [self.input['joint_2'].data.cpu(), 'pose']),
            ('seg_2', [self.output['seg_2'].data.cpu(), 'seg']),
            ('flow_out', [self.output['flow_out'].data.cpu(), 'flow']),
            ('flow_tar', [self.output['flow_tar'].data.cpu(), 'flow']),
            ('vis_out', [self.output['vis_out'].data.cpu(), 'vis']),
            ('vis_tar', [self.output['vis_tar'].data.cpu(), 'vis']),
        ]

        if self.opt.G_pix_warp:
            visual_items += [
                ('img_warp', [self.output['img_warp'].data.cpu(), 'rgb']),
                ('img_out_G', [self.output['img_out_G'].data.cpu(), 'rgb']),
                ('pix_mask', [self.output['pix_mask'].data.cpu(), 'softmask']),
                ('img_out', [self.output['img_out'].data.cpu(), 'rgb']),
                ('img_tar', [self.output['img_tar'].data.cpu(), 'rgb'])
            ]
        else:
            visual_items += [
                ('img_warp', [self.output['img_warp'].data.cpu(), 'rgb']),
                ('img_out', [self.output['img_out'].data.cpu(), 'rgb']),
                ('img_tar', [self.output['img_tar'].data.cpu(), 'rgb'])
            ]
        
        visuals = OrderedDict(visual_items)
        return visuals

    def save(self, label):
        # save network weights
        self.save_network(self.netG, 'netG', label, self.gpu_ids)
        if self.use_gan:
            self.save_network(self.netD, 'netD', label, self.gpu_ids)
        if self.opt.G_pix_warp:
            self.save_network(self.netPW, 'netPW', label, self.gpu_ids)
        # save optimizer status
        self.save_optim(self.optim, 'optim', label)
        if self.use_gan:
            self.save_optim(self.optim_D, 'optim_D', label)
    
    def train(self):
        # netG and netD will always be in 'train' status
        pass
    
    def eval(self):
        # netG and netD will always be in 'train' status
        pass

##################################################
# helper functions
##################################################
def load_flow_network(model_id, epoch='best', gpu_ids=[]):
    from flow_regression_model import FlowRegressionModel
    opt_dict = io.load_json(os.path.join('checkpoints', model_id, 'train_opt.json'))
    opt = argparse.Namespace(**opt_dict)
    opt.gpu_ids = gpu_ids
    opt.is_train = False # prevent loading discriminator, optimizer...
    opt.which_epoch = epoch
    # create network
    model = FlowRegressionModel()
    model.initialize(opt)
    return model.netF

   
