from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import networks_flow
from base_model import BaseModel
from collections import OrderedDict
from misc.pytorch_ssim import ssim

class Flow3dModel(BaseModel):
    def name(self):
        return 'Flow3dModel'
    
    def initialize(self, opt):
        super(Flow3dModel, self).initialize(opt)
        ###################################
        # define flow networks
        ###################################
        if opt.which_model == 'unet':
            self.netF = networks_flow.FlowUnet(
                input_nc = self.get_input_dim(opt.input_type1) + self.get_input_dim(opt.input_type2) + (opt.priori_nc if opt.use_flow_priori else 0),
                nf = opt.nf,
                start_scale = opt.start_scale,
                num_scale = opt.num_scale,
                norm_layer = nn.BatchNorm2d,
                gpu_ids = opt.gpu_ids,
            )
        elif opt.which_model == 'pwc':
            self.netF = networks_flow.PWCNet(
                input_nf = self.get_input_dim(opt.input_feat_type),
                input_nc = self.get_input_dim(opt.input_context_type),
                input_np = self.opt.priori_nc if self.opt.use_flow_priori else 0,
                start_scale = opt.start_scale,
                num_scale = opt.num_scale,
                md = opt.md,
                dense_connect = opt.dense_connect,
                use_context_refine = opt.use_context_refine,
                norm = opt.norm,
                start_dim_level = opt.start_dim_level,
                residual = opt.residual,
                gpu_ids = opt.gpu_ids
            )
        elif opt.which_model == 'unet4':
            self.netF = networks_flow.FlowUnet_v4(
                input_nc = self.get_input_dim(opt.input_type1) + self.get_input_dim(opt.input_type2) + (opt.priori_nc if opt.use_flow_priori else 0),
                nf = opt.nf,
                max_nf = opt.max_nf,
                start_scale = opt.start_scale,
                num_scales = opt.num_scale,
                norm = 'batch',
                gpu_ids = opt.gpu_ids,
            )
        if opt.gpu_ids:
            self.netF.cuda()
        networks_flow.init_weights(self.netF, init_type=opt.init_type)
        ###################################
        # define discriminator
        ###################################
        if self.opt.use_gan and self.is_train:
            self.netD = networks_flow.NLayerDiscriminator(
                input_nc = self.get_input_dim_D(opt.D_input_type),
                ndf = 64,
                n_layers = 3,
                norm_layer = nn.BatchNorm2d,
                use_sigmoid = opt.gan_type=='dcgan',
                output_bias = True,
                gpu_ids=opt.gpu_ids
            )
            if opt.gpu_ids:
                self.netD.cuda()
            networks_flow.init_weights(self.netD, init_type=opt.init_type)
        ###################################
        # define flow refiner
        ###################################
        if opt.use_post_refine:
            self.netPR = networks_flow.FlowRefineNet(
                input_nf = self.get_input_dim(opt.pr_input_type1) + self.get_input_dim(opt.pr_input_type2),
                type_name = opt.which_model_PR,
                norm = opt.norm,
                gpu_ids = opt.gpu_ids
            )
            if opt.gpu_ids:
                self.netPR.cuda()
            networks_flow.init_weights(self.netPR, init_type=opt.init_type)
        ###################################
        # loss and optimizers
        ###################################
        self.crit_flow = networks_flow.MultiScaleFlowLoss(start_scale=opt.start_scale, num_scale=opt.num_scale, loss_type=opt.flow_loss_type)
        self.crit_vis = nn.CrossEntropyLoss() #(0-visible, 1-invisible, 2-background)
        if opt.use_ss_flow_loss:
            self.crit_flow_ss = networks_flow.SS_FlowLoss(loss_type='l1')
        if self.is_train:
            self.optimizers = []
            params = []
            if not (self.opt.use_post_refine and self.opt.fix_netF):
                params.append({'params': self.netF.parameters()})
            if self.opt.use_post_refine:
                params.append({'params': self.netPR.parameters()})
            self.optim = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optim)
            # netD optimizor
            if opt.use_gan:
                self.crit_gan = networks_flow.GANLoss(use_lsgan=opt.gan_type=='lsgan')
                if opt.gpu_ids:
                    self.crit_gan.cuda()
                self.optim_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
                self.optimizers.append(self.optim_D)
        ###################################
        # load trained model
        ###################################
        print('####################')
        if not self.is_train:
            # load trained model for test
            print('load flow3d model')
            self.load_network(self.netF, 'netF', opt.which_epoch)
            if self.opt.use_post_refine:
                self.load_network(self.netPR, 'netPR', opt.which_epoch)
        elif opt.continue_train:
            # resume training
            print('resume training')
            self.load_network(self.netF, 'netF', opt.which_epoch)
            if self.opt.use_post_refine:
                self.load_network(self.netPR, 'netPR', opt.which_epoch)
            self.load_optim(self.optim, 'optim', opt.which_epoch)
            if self.opt.use_gan:
                self.load_network(self.net_D, 'netD', opt.which_epoch)
                self.load_optim(self.optim_D, 'optim_D', opt.which_epoch)
        elif opt.pretrain_id != '':
            # load prerained netF
            # do not load netPR or netD
            self.load_network(self.netF, 'netF', opt.pretrain_epoch, opt.pretrain_id)
        ###################################
        # schedulers
        ###################################
        if self.is_train:
            self.schedulers = []
            for optim in self.optimizers:
                self.schedulers.append(networks_flow.get_scheduler(optim, opt))
        
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
            'seg_mask_1',
            'seg_mask_2',
            'silh_1',
            'silh_2',
            'flow_2to1',
            'vis_2',
            'flow_priori'
        ]
        for name in input_list:
            if name in data:
                self.input[name] = self.Tensor(data[name].size()).copy_(data[name])
        self.input['id'] = zip(data['id_1'], data['id_2'])

        if self.opt.use_flow_priori:
            self.input['flow_priori'] *= 0.05
    

    def forward(self, parts=None):
        if parts is None:
            parts = ['netF']
            if self.opt.use_post_refine:
                parts.append('netPR')
        
        for p in parts:
            if p == 'netF':
                # forward netF
                if self.opt.which_model == 'unet' or self.opt.which_model == 'unet4':
                    input = [self.get_input_tensor(self.opt.input_type1, '1'), self.get_input_tensor(self.opt.input_type2, '2')]
                    if self.opt.use_flow_priori:
                        input.append(self.input['flow_priori'])
                    input = torch.cat(input, dim=1)
                    flow_out, vis_out, flow_pyramid_out, flow_feat = self.netF(input)
                    flow_scale = 20.
                    flow_out = flow_out * flow_scale
                elif self.opt.which_model == 'pwc':
                    feat_1 = self.get_input_tensor(self.opt.input_feat_type, '1')
                    feat_2 = self.get_input_tensor(self.opt.input_feat_type, '2')
                    ctx_1 = self.get_input_tensor(self.opt.input_context_type, '1')
                    if self.opt.use_flow_priori:
                        flow_out, vis_out, flow_pyramid_out, flow_feat = self.netF(feat_1, feat_2, ctx_1, self.input['flow_priori'])
                    else:
                        flow_out, vis_out, flow_pyramid_out, flow_feat = self.netF(feat_1, feat_2, ctx_1)
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
            elif p == 'netPR':
                assert self.opt.use_post_refine
                pr_input = torch.cat((self.get_input_tensor(self.opt.pr_input_type1, '1'), self.get_input_tensor(self.opt.pr_input_type2, '2')), dim=1)
                flow_pr_d, vis_pr_d = self.netPR(pr_input)
                flow_scale = 20.
                self.output['flow_pr_d'] = flow_pr_d
                self.output['vis_pr_d'] = vis_pr_d
                self.output['flow_pr'] = self.output['flow_out'] + flow_pr_d * flow_scale
                self.output['vis_pr'] = self.output['vis_out'] + vis_pr_d
                self.output['flow_pr_final'] = self.output['flow_pr'] * (self.output['vis_pr'].argmax(dim=1, keepdim=True)<2).float()
    
    def test(self, compute_loss=False):
        self.eval()
        with torch.no_grad():
            self.forward()
        if compute_loss:
            self.compute_loss()
        self.train()
    
    def compute_loss(self):
        # flow loss
        if self.opt.output_full_losses:
            self.output['loss_flow'], _, full_losses = self.crit_flow(self.output['flow_pyramid_out'], self.output['flow_tar'], self.output['mask_tar'], output_full_losses=True)
            for i, l in enumerate(full_losses):
                self.output['loss_flow_%d'%(i+1)] = l
        else:
            self.output['loss_flow'], _ = self.crit_flow(self.output['flow_pyramid_out'], self.output['flow_tar'], self.output['mask_tar'])
        # flow_ss loss
        if self.opt.use_ss_flow_loss:
            if self.opt.ss_guide == 'seg':
                ss1, ss2 = self.input['seg_mask_1'], self.input['seg_mask_2']
            elif self.opt.ss_guide == 'silh':
                ss1, ss2 = self.input['silh_1'], self.input['silh_2']
            self.output['loss_flow_ss'] = self.crit_flow_ss(self.output['flow_out'], self.output['flow_tar'], ss1, ss2, self.output['vis_tar'])

        # visibility loss
        self.output['loss_vis'] = self.crit_vis(self.output['vis_out'], self.output['vis_tar'].long().squeeze(dim=1))
        # EPE
        self.output['EPE'] = networks_flow.EPE(self.output['flow_out'], self.output['flow_tar'], self.output['mask_tar'])
        # GAN
        if self.opt.use_gan:
            input_D = self.get_input_D(self.opt.D_input_type, real=False)
            self.output['loss_G'] = self.crit_gan(self.netD(input_D), True)
        # PostRefineNet
        if self.opt.use_post_refine:
            # PR regularization loss
            pr_out = torch.cat((self.output['flow_pr_d'], self.output['vis_pr_d']), dim=1)
            self.output['loss_pr_reg'] = (pr_out*pr_out).mean()
            # PR visibility loss
            mask_tar = (self.input['seg_2']>0.5).long() # 0: background. 1: foreground
            mask_out = torch.cat((self.output['vis_pr'][:,2:,...], self.output['vis_pr'][:,0:2,...].sum(dim=1, keepdim=True)), dim=1) # merge visible/invisible body part into "foreground"
            mask_out_ctr = torch.cat((self.output['vis_out'][:,2:,...], self.output['vis_out'][:,0:2,...].sum(dim=1, keepdim=True)), dim=1)
            self.output['loss_pr_vis'] = self.crit_vis(mask_out, mask_tar.squeeze(dim=1))
            self.output['loss_pr_vis_ctr'] = self.crit_vis(mask_out_ctr, mask_tar.squeeze(dim=1))
            # PR reconstrunction loss
            if self.opt.pr_recon_type == 'seg':
                src = self.input['seg_mask_1']
                tar = self.input['seg_mask_2']
            elif self.opt.pr_recon_type == 'img':
                src = self.input['img_1']
                tar = self.input['img_2']
            out = networks_flow.warp_acc_flow(src, self.output['flow_pr'])# warp "src" according to refined flow
            out_ctr = networks_flow.warp_acc_flow(src, self.output['flow_out'])
            mask_tar = mask_tar.float()
            out = out * mask_tar
            out_ctr = out_ctr * mask_tar
            tar = tar * mask_tar
            if self.opt.pr_recon_meas == 'l2':
                self.output['loss_pr_recon'] = F.mse_loss(out, tar)
                self.output['loss_pr_recon_ctr'] = F.mse_loss(out_ctr, tar) #this loss will not be used to optimize model
            elif self.opt.pr_recon_meas == 'ssim':
                if self.opt.pr_recon_type == 'img':
                    self.output['loss_pr_recon'] = 1.0-ssim(out*0.5+0.5, tar*0.5+0.5)
                    self.output['loss_pr_recon_ctr'] = 1.0-ssim(out_ctr*0.5+0.5, tar*0.5+0.5) #this loss will not be used to optimize model
                else:
                    self.output['loss_pr_recon'] = 1.0-ssim(out, tar)
                    self.output['loss_pr_recon_ctr'] = 1.0-ssim(out_ctr, tar) #this loss will not be used to optimize model
            self.output['pr_warp_out'] = out
            self.output['pr_warp_out_ctr'] = out_ctr
            self.output['pr_warp_gt'] = tar
            
   
    def backward(self, check_grad=False, parts=None):
        if parts is None:
            # note that "netD" is not in the default list of parts
            parts = ['netF']
            if self.opt.use_post_refine:
                parts.append('netPR')

        if not check_grad:
            loss = 0
            for p in parts:
                if p == 'netF':
                    loss += self.output['loss_flow'] * self.opt.loss_weight_flow
                    loss += self.output['loss_vis'] * self.opt.loss_weight_vis
                    if self.opt.use_ss_flow_loss:
                        loss += self.output['loss_flow_ss'] * self.opt.loss_weight_flow_ss
                    if self.opt.use_gan:
                        loss += self.output['loss_G'] * self.opt.loss_weight_gan
                elif p == 'netPR':
                    assert self.opt.use_post_refine
                    loss += self.output['loss_pr_reg'] * self.opt.loss_weight_pr_reg
                    loss += self.output['loss_pr_recon'] * self.opt.loss_weight_pr_recon
                    loss += self.output['loss_pr_vis'] * self.opt.loss_weight_pr_vis
            loss.backward()
        else:
            for p in parts:
                if p == 'netF':
                    with networks_flow.CalcGradNorm(self.netF) as cgn:
                        (self.output['loss_flow']*self.opt.loss_weight_flow).backward(retain_graph=True)
                        self.output['grad_flow'] = cgn.get_grad_norm()                        
                        (self.output['loss_vis'] * self.opt.loss_weight_vis).backward(retain_graph=True)
                        self.output['grad_vis'] = cgn.get_grad_norm()
                        if self.opt.use_ss_flow_loss:
                            (self.output['loss_flow_ss'] * self.opt.loss_weight_flow_ss).backward(retain_graph=True)
                            self.output['grad_flow_ss'] = cgn.get_grad_norm()
                        if self.opt.use_gan:
                            (self.output['loss_G'] * self.opt.loss_weight_gan).backward(retain_graph=True)
                            self.output['grad_G'] = cgn.get_grad_norm()
                elif p == 'netPR':
                    assert self.opt.use_post_refine
                    with networks_flow.CalcGradNorm(self.netPR) as cgn:
                        (self.output['loss_pr_reg'] * self.opt.loss_weight_pr_reg).backward(retain_graph=True)
                        self.output['grad_pr_reg'] = cgn.get_grad_norm()
                        (self.output['loss_pr_recon'] * self.opt.loss_weight_pr_recon).backward(retain_graph=True)
                        self.output['grad_pr_recon'] = cgn.get_grad_norm()
                        (self.output['loss_pr_vis'] * self.opt.loss_weight_pr_vis).backward(retain_graph=True)
                        self.output['grad_pr_vis'] = cgn.get_grad_norm()

    def backward_D(self):
        assert self.opt.use_gan
        input_D_real = self.get_input_D(self.opt.D_input_type, real=True).detach()
        input_D_fake = self.get_input_D(self.opt.D_input_type, real=False).detach()
        self.output['loss_D'] = 0.5 * (self.crit_gan(self.netD(input_D_real), True) + \
                                self.crit_gan(self.netD(input_D_fake), False))
        (self.output['loss_D'] * self.opt.loss_weight_gan).backward()



    def optimize_parameters(self, check_grad=False, fix_netF=False):
        self.output = {}
        self.train()
        if not fix_netF:
            self.forward()
            if self.opt.use_gan:
                # optimize netD first
                self.optim_D.zero_grad()
                self.backward_D()
                self.optim_D.step()
            self.optim.zero_grad()
            self.compute_loss()
            self.backward(check_grad)
        else:
            self.netF.eval()
            with torch.no_grad():
                self.forward(parts=['netF'])
            self.forward(parts=['netPR'])
            if self.opt.use_gan:
                # optimize netD first
                self.optim_D.zero_grad()
                self.backward_D()
                self.optim_D.step()
            self.optim.zero_grad()
            self.compute_loss()
            self.backward(check_grad, parts=['netPR'])
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
            elif item == 'silh':
                dim += self.opt.silh_nc
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
                tensor.append(self.input['seg_mask_%s'%index])
            elif item == 'joint':
                tensor.append(self.input['joint_%s'%index])
            elif item == 'flow':
                tensor.append(self.output['flow_out']*0.05)
            elif item == 'flow_gt':
                tensor.append(self.input['flow_2to1']*0.05)
            elif item == 'flow_feat':
                tensor.append(F.upsample(self.output['flow_feat'], size=self.opt.image_size, mode='bilinear'))
            elif item == 'vis':
                tensor.append(self.output['vis_out'])
            elif item == 'silh':
                tensor.append(self.input['silh_%s'%index])
            else:
                raise Exception('invalid input type %s'%item)
        tensor = torch.cat(tensor, dim=1)
        return tensor
    
    def get_input_dim_D(self, D_input_type):
        if D_input_type == 'warp_err':
            return 3 #(img_2 - img_1w)
        elif D_input_type == 'flow_img':
            return 2+3+1 #(flow + mask + img)
        elif D_input_type == 'flow_silh':
            return 2 + self.opt.silh_nc + 1
    
    def get_input_D(self, D_input_type, real=True):
        flow_scale = 20.
        if D_input_type == 'warp_err':
            if real:
                warp = networks_flow.warp_acc_flow(self.input['img_1'], self.output['flow_tar'])
            else:
                warp = networks_flow.warp_acc_flow(self.input['img_1'], self.output['flow_out'])
            warp_err = (warp - self.input['img_2']) * self.output['mask_tar']
            return warp_err
        elif D_input_type == 'flow_img':
            if real:
                flow = self.output['flow_tar'].div(flow_scale)*self.output['mask_tar']
            else:
                flow = self.output['flow_out'].div(flow_scale)*self.output['mask_tar']
            return torch.cat((flow, self.output['mask_tar'], self.input['img_2']), dim=1)
        elif D_input_type == 'flow_silh':
            if real:
                flow = self.output['flow_tar'].div(flow_scale)*self.output['mask_tar']
            else:
                flow = self.output['flow_out'].div(flow_scale)*self.output['mask_tar']
            return torch.cat((flow, self.output['mask_tar'], self.input['silh_2']), dim=1)
            
    
    def get_current_errors(self):
        error_list = [
            'EPE',
            'loss_flow',
            'loss_vis',
            'loss_flow_ss',
            'loss_G',
            'loss_D',
            'grad_flow',
            'grad_vis',
            'grad_flow_ss',
            'grad_G',
            'loss_pr_recon',
            'loss_pr_recon_ctr',
            'loss_pr_vis',
            'loss_pr_vis_ctr',
            'loss_pr_reg',
            'grad_pr_recon',
            'grad_pr_vis',
            'grad_pr_reg'
        ]
        errors = OrderedDict()
        for item in error_list:
            if item in self.output:
                errors[item] = self.output[item].item()
        
        if self.opt.output_full_losses:
            for i in range(self.opt.num_scale):
                errors['loss_flow_%d'%(i+1)] = self.output['loss_flow_%d'%(i+1)].item()

        return errors
    
    def get_current_visuals(self):
        visuals = OrderedDict([
            ('img_1', [self.input['img_1'].data.cpu(), 'rgb']),
            ('img_2', [self.input['img_2'].data.cpu(), 'rgb']),
            ('joint_1', [self.input['joint_1'].data.cpu(), 'pose']),
            ('joint_2', [self.input['joint_2'].data.cpu(), 'pose']),
            ('seg_1', [self.input['seg_mask_1'].data.cpu(), 'seg']),
            ('seg_2', [self.input['seg_mask_2'].data.cpu(), 'seg']),
            ('silh_1', [self.input['silh_1'].data.cpu(), 'seg']),
            ('silh_2', [self.input['silh_2'].data.cpu(), 'seg']),
            ('flow_tar', [self.output['flow_tar'].data.cpu(), 'flow']),
            ('flow_out', [self.output['flow_out'].data.cpu(), 'flow']),
            ('vis_tar', [self.output['vis_tar'].data.cpu(), 'vis']),
            ('vis_out', [self.output['vis_out'].data.cpu(), 'vis']),
            ('flow_final', [self.output['flow_final'].data.cpu(), 'flow']),
        ])

        if self.opt.vis_warp:
            self.output['img_warp'] = networks_flow.warp_acc_flow(self.input['img_1'], self.output['flow_final']) * self.output['mask_out']
            self.output['img_warp_gt'] = networks_flow.warp_acc_flow(self.input['img_1'], self.output['flow_tar']) * self.output['mask_tar']

            visuals['img_warp_gt'] = [self.output['img_warp_gt'].data.cpu(), 'rgb']
            visuals['img_warp'] = [self.output['img_warp'].data.cpu(), 'rgb']
        
        if self.opt.vis_flow_pyr:
            for i, f in enumerate(self.output['flow_pyramid_out']):
                flow_i = F.upsample(f, size=self.output['flow_out'].size()[2::], mode='bilinear') * self.output['mask_out']
                visuals['flow_out_%d'%(i+1)] = [flow_i.data.cpu(), 'flow']
        
        if self.opt.use_flow_priori:
            priori_x = self.input['flow_priori'][:,0::2,...]
            priori_y = self.input['flow_priori'][:,1::2,...]
            index_x = priori_x.abs().argmax(dim=1, keepdim=True)
            index_y = priori_y.abs().argmax(dim=1, keepdim=True)
            priori_x = priori_x.gather(1, index_x)
            priori_y = priori_y.gather(1, index_y)
            visuals['priori'] = [torch.cat((priori_x, priori_y), dim=1).data.cpu(), 'flow']
        
        if self.opt.use_post_refine:
            if self.opt.pr_recon_type=='img':
                vis_type = 'rgb'
            elif self.opt.pr_recon_type=='seg':
                vis_type = 'seg'
            
            visuals['pr_warp_gt'] = [self.output['pr_warp_gt'].data.cpu(), vis_type]
            visuals['pr_warp_out'] = [self.output['pr_warp_out'].data.cpu(), vis_type]
            visuals['pr_warp_out_ctr'] = [self.output['pr_warp_out_ctr'].data.cpu(), vis_type]
            visuals['flow_pr_final'] = [self.output['flow_pr_final'].data.cpu(), 'flow']
            self.output['img_warp_pr'] = networks_flow.warp_acc_flow(self.input['img_1'], self.output['flow_pr_final']) * (self.output['vis_pr'].argmax(dim=1, keepdim=True)<2).float()
            visuals['img_warp_pr'] = [self.output['img_warp_pr'].data.cpu(), 'rgb']

        return visuals
    
    def save(self, label):
        # save networks weights
        self.save_network(self.netF, 'netF', label, self.gpu_ids)
        if self.opt.use_post_refine:
            self.save_network(self.netPR, 'netPR', label, self.gpu_ids)
        if self.opt.use_gan:
            self.save_network(self.netD, 'netD', label, self.gpu_ids)
        # save optimizer status
        if self.is_train:
            self.save_optim(self.optim, 'optim', label)
            if self.opt.use_gan:
                self.save_optim(self.optim_D, 'optim_D', label)


    def train(self):
        self.netF.train()
        if self.opt.use_post_refine:
            self.netPR.train()
        if self.opt.use_gan:
            self.netD.train()

    def eval(self):
        self.netF.eval()
        if self.opt.use_post_refine:
            self.netPR.eval()
        if self.opt.use_gan:
            self.netD.eval()
