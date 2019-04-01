from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import networks
from base_model import BaseModel
from collections import OrderedDict

class FlowRegressionModel(BaseModel):
    def name(self):
        return 'FlowRegressionModel'
    
    def initialize(self, opt):
        super(FlowRegressionModel, self).initialize(opt)
        ###################################
        # define flow networks
        ###################################
        if opt.which_model == 'unet':
            self.netF = networks.FlowUnet(
                input_nc = self.get_input_dim(opt.input_type1) + self.get_input_dim(opt.input_type2),
                nf = opt.nf,
                start_scale = opt.start_scale,
                num_scale = opt.num_scale,
                norm_layer = nn.BatchNorm2d,
                gpu_ids = opt.gpu_ids,
            )
        elif opt.which_model == 'unet_v2':
            self.netF = networks.FlowUnet_v2(
                input_nc = self.get_input_dim(opt.input_type1) + self.get_input_dim(opt.input_type2),
                nf = opt.nf,
                max_nf = opt.max_nf,
                start_scale = opt.start_scale,
                num_scales = opt.num_scale,
                norm = opt.norm,
                gpu_ids = opt.gpu_ids,
            )
        if opt.gpu_ids:
            self.netF.cuda()
        networks.init_weights(self.netF, init_type=opt.init_type)
        ###################################
        # loss and optimizers
        ###################################
        self.crit_flow = networks.MultiScaleFlowLoss(start_scale=opt.start_scale, num_scale=opt.num_scale, loss_type=opt.flow_loss_type)
        self.crit_vis = nn.CrossEntropyLoss() #(0-visible, 1-invisible, 2-background)
        if opt.use_ss_flow_loss:
            self.crit_flow_ss = networks.SS_FlowLoss(loss_type='l1')
        if self.is_train:
            self.optimizers = []
            params = []
            if not (self.opt.use_post_refine and self.opt.fix_netF):
                params.append({'params': self.netF.parameters()})
            if self.opt.use_post_refine:
                params.append({'params': self.netPR.parameters()})
            self.optim = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optim)
            
        ###################################
        # load trained model
        ###################################
        if not self.is_train:
            # load trained model for test
            print('load pretrained model')
            self.load_network(self.netF, 'netF', opt.which_epoch)
        elif opt.resume_train:
            # resume training
            print('resume training')
            self.load_network(self.netF, 'netF', opt.last_epoch)
            self.load_optim(self.optim, 'optim', opt.last_epoch)
        ###################################
        # schedulers
        ###################################
        if self.is_train:
            self.schedulers = []
            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))
        
    def set_input(self, data):
        input_list = [
            'img_1',
            'img_2',
            'joint_c_1',
            'joint_c_2',
            'joint_1',
            'joint_2',
            'seg_1',
            'seg_2',
            'seg_label_1',
            'seg_label_2',
            'flow_2to1',
            'vis_2',
        ]
        for name in input_list:
            if name in data:
                self.input[name] = self.Tensor(data[name].size()).copy_(data[name])
        self.input['id'] = zip(data['id_1'], data['id_2'])

    def forward(self):
        input = [self.get_input_tensor(self.opt.input_type1, '1'), self.get_input_tensor(self.opt.input_type2, '2')]
        input = torch.cat(input, dim=1)
        flow_out, vis_out, flow_pyramid_out, flow_feat = self.netF(input)
        flow_scale = 20.
        flow_out = flow_out * flow_scale

        self.output['flow_pyramid_out'] = flow_pyramid_out
        self.output['flow_out'] = flow_out
        self.output['vis_out'] = vis_out
        self.output['flow_tar'] = self.input['flow_2to1']
        self.output['vis_tar'] = self.input['vis_2']
        self.output['flow_feat'] = flow_feat
        self.output['mask_out'] = (self.output['vis_out'].argmax(dim=1, keepdim=True) < 2).float()
        self.output['mask_tar'] = (self.output['vis_tar']<2).float()
        self.output['flow_final'] = self.output['flow_out'] * self.output['mask_out']
        
    
    def test(self, compute_loss=False):
        with torch.no_grad():
            self.forward()
            if compute_loss:
                self.compute_loss()
    
    def compute_loss(self):
        # flow loss
        self.output['loss_flow'], _ = self.crit_flow(self.output['flow_pyramid_out'], self.output['flow_tar'], self.output['mask_tar'])
        # flow_ss loss
        if self.opt.use_ss_flow_loss:
            self.output['loss_flow_ss'] = self.crit_flow_ss(self.output['flow_out'], self.output['flow_tar'], self.input['seg_1'], self.input['seg_2'], self.output['vis_tar'])

        # visibility loss
        self.output['loss_vis'] = self.crit_vis(self.output['vis_out'], self.output['vis_tar'].long().squeeze(dim=1))
        # EPE
        self.output['EPE'] = networks.EPE(self.output['flow_out'], self.output['flow_tar'], self.output['mask_tar'])
   
    def backward(self, check_grad=False):
        
        if not check_grad:
            loss = 0
            loss += self.output['loss_flow'] * self.opt.loss_weight_flow
            loss += self.output['loss_vis'] * self.opt.loss_weight_vis
            if self.opt.use_ss_flow_loss:
                loss += self.output['loss_flow_ss'] * self.opt.loss_weight_flow_ss
            loss.backward()
        else:
            with networks.CalcGradNorm(self.netF) as cgn:
                (self.output['loss_flow']*self.opt.loss_weight_flow).backward(retain_graph=True)
                self.output['grad_flow'] = cgn.get_grad_norm()                        
                (self.output['loss_vis'] * self.opt.loss_weight_vis).backward(retain_graph=True)
                self.output['grad_vis'] = cgn.get_grad_norm()
                if self.opt.use_ss_flow_loss:
                    (self.output['loss_flow_ss'] * self.opt.loss_weight_flow_ss).backward(retain_graph=True)
                    self.output['grad_flow_ss'] = cgn.get_grad_norm()
    
    def optimize_parameters(self, check_grad=False):
        self.output = {}
        self.train()
        self.forward()
        self.optim.zero_grad()
        self.compute_loss()
        self.backward(check_grad)
        self.optim.step()
    
    def get_input_dim(self, input_type):
        dim = 0
        input_items = input_type.split('+')
        input_items.sort()
        for item in input_items:
            if item == 'img':
                dim += 3
            elif item == 'seg':
                dim += self.opt.seg_nc
            elif item == 'joint':
                dim += self.opt.joint_nc
            elif item == 'flow' or item == 'flow_gt':
                dim += 2
            elif item == 'flow_feat':
                dim += self.netF.nf_out
            elif item == 'vis':
                dim += 3
            else:
                raise Exception('invalid input type %s'%item)
        return dim

    def get_input_tensor(self, input_type, index='1'):
        assert index in {'1', '2'}
        tensor = []
        input_items = input_type.split('+')
        input_items.sort()
        for item in input_items:
            if item == 'img':
                tensor.append(self.input['img_%s'%index])
            elif item == 'seg':
                tensor.append(self.input['seg_%s'%index])
            elif item == 'joint':
                tensor.append(self.input['joint_%s'%index])
            else:
                raise Exception('invalid input type %s'%item)
        tensor = torch.cat(tensor, dim=1)
        return tensor
    
    def get_current_errors(self):
        error_list = [
            'EPE',
            'loss_flow',
            'loss_vis',
            'loss_flow_ss',
            'grad_flow',
            'grad_vis',
            'grad_flow_ss',
        ]
        errors = OrderedDict()
        for item in error_list:
            if item in self.output:
                errors[item] = self.output[item].item()
        
        return errors
    
    def get_current_visuals(self):
        visuals = OrderedDict([
            ('img_1', [self.input['img_1'].data.cpu(), 'rgb']),
            ('img_2', [self.input['img_2'].data.cpu(), 'rgb']),
            ('joint_1', [self.input['joint_1'].data.cpu(), 'pose']),
            ('joint_2', [self.input['joint_2'].data.cpu(), 'pose']),
            ('seg_1', [self.input['seg_1'].data.cpu(), 'seg']),
            ('seg_2', [self.input['seg_2'].data.cpu(), 'seg']),
            ('flow_tar', [self.output['flow_tar'].data.cpu(), 'flow']),
            ('flow_out', [self.output['flow_out'].data.cpu(), 'flow']),
            ('vis_tar', [self.output['vis_tar'].data.cpu(), 'vis']),
            ('vis_out', [self.output['vis_out'].data.cpu(), 'vis']),
            ('flow_final', [self.output['flow_final'].data.cpu(), 'flow']),
        ])
        return visuals
    
    def save(self, label):
        # save networks weights
        self.save_network(self.netF, 'netF', label, self.gpu_ids)
        # save optimizer status
        if self.is_train:
            self.save_optim(self.optim, 'optim', label)


    def train(self):
        self.netF.train()
        

    def eval(self):
        self.netF.eval()
