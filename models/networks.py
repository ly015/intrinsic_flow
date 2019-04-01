from __future__ import division
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from modules import *
from skimage.measure import compare_ssim, compare_psnr
import functools

##############################################
# Image generation networks: Unet and DualUnet
##############################################
def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
        norm_layer(out_channels),
    )
    return model

def channel_mapping(in_channels, out_channels, norm_layer=nn.BatchNorm2d, bias=False):
    return conv(in_channels, out_channels, kernel_size=1, norm_layer=norm_layer, bias=bias)

class ResidualBlock(nn.Module):
    '''
    Derived from Variational UNet.
    '''
    def __init__(self, dim, dim_a, norm_layer=nn.BatchNorm2d, use_bias=False, activation=nn.ReLU(False), use_dropout=False, no_end_norm=False):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.activation = activation
        if dim_a <= 0 or dim_a is None:
            # w/o additional input
            if no_end_norm:
                self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=Identity, bias=True)
            else:
                self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=norm_layer, bias=use_bias)
        else:
            # w/ additional input
            self.conv_a = channel_mapping(in_channels=dim_a, out_channels=dim,norm_layer=norm_layer, bias=use_bias)
            if no_end_norm:
                self.conv = conv(in_channels=dim*2, out_channels=dim, kernel_size=3, padding=1, norm_layer=Identity, bias=True)
            else:
                self.conv = conv(in_channels=dim*2, out_channels=dim, kernel_size=3, padding=1, norm_layer=norm_layer, bias=use_bias)
    
    def forward(self, x, a=None):
        if a is None:
            # w/o additional input
            residual = x
        else:
            # w/ additional input
            a = self.conv_a(self.activation(a))
            residual = torch.cat((x, a), dim=1)
        residual = self.conv(self.activation(residual))
        out = x + residual
        if self.use_dropout:
            out = F.dropout(out, p=0.5, training=self.training)
        return out

class GateBlock(nn.Module):
    def __init__(self, dim, dim_a,activation=nn.ReLU(False)):
        super(GateBlock, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=dim_a, out_channels=dim, kernel_size=1)
    def forward(self, x, a):
        '''
        x: (bsz, dim, h, w)
        a: (bsz, dim_a, h, w)
        '''
        a = self.activation(a)
        g = F.sigmoid(self.conv(a))
        return x*g

class UnetGenerator(nn.Module):
    '''
    A variation of Unet that use residual blocks instead of convolution layer at each scale
    '''
    def __init__(self, input_nc, output_nc, nf=64, max_nf=256, num_scales=7, n_residual_blocks=2, norm='batch', activation=nn.ReLU(False), use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nf = nf
        self.max_nf = max_nf
        self.num_scales = num_scales
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
        self.use_dropout = use_dropout

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        
        self.pre_conv = channel_mapping(input_nc, nf, norm_layer, use_bias)
        for l in range(num_scales):
            c_in = min(nf * (l+1), max_nf)
            c_out = min(nf * (l+2), max_nf)
            # encoding layers
            for i in range(n_residual_blocks):
                self.__setattr__('enc_%d_res_%d'%(l, i), ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=False))
            downsample = nn.Sequential(
                activation,
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out)
            )
            self.__setattr__('enc_%d_downsample'%l, downsample)
            # decoding layers
            upsample = nn.Sequential(
                activation,
                nn.Conv2d(c_out, c_in*4, kernel_size=3, padding=1, bias=use_bias),
                nn.PixelShuffle(2),
                norm_layer(c_in)
            )
            self.__setattr__('dec_%d_upsample'%l, upsample)
            for i in range(n_residual_blocks):
                self.__setattr__('dec_%d_res_%d'%(l, i), ResidualBlock(c_in, c_in, norm_layer, use_bias, activation, use_dropout))
        
        self.dec_output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, output_nc, kernel_size=7, padding=0, bias=True)
        )
    
    def forward(self, x, single_device=False):
        if len(self.gpu_ids) > 1 and (not single_device):
            return nn.parallel.data_parallel(self, x, module_kwargs={'single_device':True})
        else:
            hiddens = []
            x = self.pre_conv(x)
            # encode
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x = self.__getattr__('enc_%d_res_%d'%(l,i))(x)
                    hiddens.append(x)
                x = self.__getattr__('enc_%d_downsample'%l)(x)
            # decode
            for l in range(self.num_scales-1,-1,-1):
                x = self.__getattr__('dec_%d_upsample'%l)(x)
                for i in range(self.n_residual_blocks-1,-1,-1):
                    h = hiddens.pop()
                    x = self.__getattr__('dec_%d_res_%d'%(l, i))(x,h)
            out = self.dec_output(x)
            return out

class DualUnetGenerator(nn.Module):
    '''
    A variation of Unet architecture, similar to deformable gan. It contains two encoders: one for target pose and one for appearance. The feature map of appearance encoder will be warped to target pose, guided
    by input flow. There are skip connections from both encoders to the decoder.
    '''
    def __init__(self, pose_nc, appearance_nc, output_nc, aux_output_nc=[], nf=32, max_nf=128, num_scales=7, num_warp_scales=5, n_residual_blocks=2, norm='batch', vis_mode='none', activation=nn.ReLU(False), use_dropout=False, no_end_norm=False, gpu_ids=[]):
        '''
        vis_mode: ['none', 'hard_gate', 'soft_gate', 'residual']
        no_end_norm: remove normalization layer at the start and the end.
        '''
        super(DualUnetGenerator, self).__init__()
        self.pose_nc = pose_nc
        self.appearance_nc = appearance_nc
        self.output_nc = output_nc
        self.nf = nf
        self.max_nf = max_nf
        self.num_scales = num_scales
        self.num_warp_scales = num_warp_scales # at higher scales, warping will not be applied because the resolution of the feature map is too small
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
        self.use_dropout = use_dropout
        self.vis_mode = vis_mode
        self.vis_expand_mult = 2 # expanded multiple when perform vis_expand
        self.aux_output_nc = aux_output_nc
        self.no_end_norm = no_end_norm
        
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        
        ####################################
        # input encoder
        ####################################
        if not no_end_norm:
            self.encp_pre_conv = channel_mapping(pose_nc, nf, norm_layer, use_bias)
            self.enca_pre_conv = channel_mapping(appearance_nc, nf, norm_layer, use_bias)
        else:
            self.encp_pre_conv = channel_mapping(pose_nc, nf, Identity, True)
            self.enca_pre_conv = channel_mapping(appearance_nc, nf, Identity, True)
        for l in range(num_scales):
            c_in = min(nf * (l+1), max_nf)
            c_out = min(nf * (l+2), max_nf)
            ####################################
            # pose encoder
            ####################################
            # resblocks
            for i in range(n_residual_blocks):
                self.__setattr__('encp_%d_res_%d'%(l, i), ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=False))
            # down sample
            p_downsample = nn.Sequential(
                activation,
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out)
            )
            self.__setattr__('encp_%d_downsample'%l, p_downsample)
            ####################################
            # appearance encoder
            ####################################
            for i in range(n_residual_blocks):
                # resblocks
                self.__setattr__('enca_%d_res_%d'%(l, i), ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=False))
                # visibility gating
                if l < num_warp_scales:
                    if vis_mode == 'hard_gate':
                        pass
                    elif vis_mode == 'soft_gate':
                        self.__setattr__('enca_%d_vis_%d'%(l, i), GateBlock(c_in, c_in*self.vis_expand_mult, activation))
                    elif vis_mode == 'residual':
                        self.__setattr__('enca_%d_vis_%d'%(l, i), ResidualBlock(c_in, c_in*self.vis_expand_mult, norm_layer, use_bias, activation, use_dropout=False))
                    elif vis_mode == 'res_no_vis':
                        self.__setattr__('enca_%d_vis_%d'%(l, i), ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=False))
            # down sample
            a_downsample = nn.Sequential(
                activation,
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out)
            )
            self.__setattr__('enca_%d_downsample'%l, p_downsample)
            ####################################
            # decoder
            ####################################
            # resblocks
            if l == num_scales-1:
                self.dec_fuse = channel_mapping(c_out*2, c_out, norm_layer, use_bias) # a fusion layer at the bottle neck
            # upsample
            upsample = nn.Sequential(
                activation,
                nn.Conv2d(c_out, c_in*4, kernel_size=3, padding=1, bias=use_bias),
                nn.PixelShuffle(2),
                norm_layer(c_in)
            )
            self.__setattr__('dec_%d_upsample'%l, upsample)
            for i in range(n_residual_blocks):
                if l == num_scales-1 and i == n_residual_blocks-1:
                    self.__setattr__('dec_%d_res_%d'%(l,i), ResidualBlock(c_in, c_in*2, norm_layer, use_bias, activation, use_dropout, no_end_norm=no_end_norm))
                else:
                    self.__setattr__('dec_%d_res_%d'%(l,i), ResidualBlock(c_in, c_in*2, norm_layer, use_bias, activation, use_dropout))
        ####################################
        # output decoder
        ####################################
        self.dec_output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, output_nc, kernel_size=7, padding=0, bias=True)
        )
        for i, a_nc in enumerate(aux_output_nc):
            dec_aux_output = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(nf, a_nc, kernel_size=7, padding=0, bias=True)
            )
            self.__setattr__('dec_aux_output_%d'%i, dec_aux_output)
    
    def _vis_expand(self, feat, vis):
        '''
        expand feature from n channels to n*vis_expand_mult channels
        '''
        feat_exp = [feat*(vis==i).float() for i in range(self.vis_expand_mult)]
        return torch.cat(feat_exp, dim=1)
        
    def forward(self, x_p, x_a, flow=None, vis=None, output_feats=False, single_device=False):
        '''
        x_p: (bsz, pose_nc, h, w), pose input
        x_a: (bsz, appearance_nc, h, w), appearance input
        vis: (bsz, 1, h, w), 0-visible, 1-invisible, 2-background
        flow: (bsz, 2, h, w) or None. if flow==None, feature warping will not be performed
        '''
        if len(self.gpu_ids)>1 and (not single_device):
            if flow is not None:
                assert vis is not None
                return nn.parallel.data_parallel(self, (x_p, x_a, flow, vis), module_kwargs={'single_device':True, 'output_feats':output_feats})
            else:
                return nn.parallel.data_parallel(self, (x_p, x_a), module_kwargs={'flow':None, 'vis':None, 'single_device':True, 'output_feats':output_feats})
        else:
            use_fw = flow is not None
            if use_fw:
                vis = vis.round()
            hidden_p = []
            hidden_a = []
            # encoding p
            x_p = self.encp_pre_conv(x_p)
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x_p = self.__getattr__('encp_%d_res_%d'%(l,i))(x_p)
                    hidden_p.append(x_p)
                x_p = self.__getattr__('encp_%d_downsample'%l)(x_p)
            # encoding a
            x_a = self.enca_pre_conv(x_a)
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x_a = self.__getattr__('enca_%d_res_%d'%(l,i))(x_a)
                    # feature warping
                    if use_fw and l < self.num_warp_scales:
                        if i == 0: # compute flow and vis once at each scale
                            flow_l = F.avg_pool2d(flow, kernel_size=2**l).div_(2**l) if l > 0 else flow
                            vis_l = -F.max_pool2d(-vis, kernel_size=2**l) if l > 0 else vis# the priority is visible>invisible>background
                        x_w = warp_acc_flow(x_a, flow_l)
                        if self.vis_mode == 'none':
                            pass
                        elif self.vis_mode == 'hard_gate':
                            x_w = x_w * (vis_l<2).float()
                        elif self.vis_mode == 'soft_gate':
                            x_we = self._vis_expand(x_w, vis_l)
                            x_w = self.__getattr__('enca_%d_vis_%d'%(l, i))(x_w, x_we)
                        elif self.vis_mode == 'residual':
                            x_we = self._vis_expand(x_w, vis_l)
                            x_w = self.__getattr__('enca_%d_vis_%d'%(l, i))(x_w, x_we)
                        elif self.vis_mode == 'res_no_vis':
                            x_w = self.__getattr__('enca_%d_vis_%d'%(l, i))(x_w)
                        hidden_a.append(x_w)
                    else:
                        hidden_a.append(x_a)
                x_a = self.__getattr__('enca_%d_downsample'%l)(x_a)
            # bottleneck fusion
            x = self.dec_fuse(torch.cat((x_p, x_a), dim=1))
            feats = [x]
            # decoding
            for l in range(self.num_scales-1, -1, -1):
                x = self.__getattr__('dec_%d_upsample'%l)(x)
                feats = [x] + feats
                for i in range(self.n_residual_blocks-1, -1, -1):
                    h_p = hidden_p.pop()
                    h_a = hidden_a.pop()
                    x = self.__getattr__('dec_%d_res_%d'%(l, i))(x, torch.cat((h_p, h_a), dim=1))
            out = self.dec_output(x)
            if self.aux_output_nc or output_feats:
                aux_out = []
                if self.aux_output_nc:
                    for i in range(len(self.aux_output_nc)):
                        aux_out.append(self.__getattr__('dec_aux_output_%d'%i)(x))
                if output_feats:

                    aux_out.append(feats)
                return out, aux_out
            else:
                return out


class UnetDecoder(nn.Module):
    '''
    Decoder that decodes hierarachical features. Support multi-task output. Used as an external decoder of a DualUnetGenerator network
    '''
    def __init__(self, output_nc=[], nf=32, max_nf=128, num_scales=7, n_residual_blocks=2, norm='batch', activation=nn.ReLU(False), gpu_ids=[]):
        super(UnetDecoder, self).__init__()
        output_nc = output_nc if isinstance(output_nc, list) else [output_nc]
        self.output_nc = output_nc
        self.nf = nf
        self.max_nf = max_nf
        self.num_scales = num_scales
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
    
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        ####################
        # hierarchical decoding layers
        ####################
        for l in range(num_scales):
            c_in = min(nf*(l+1), max_nf) # lower feature dim
            c_out = min(nf*(l+2), max_nf) # higher feature dim

            upsample = nn.Sequential(
                activation,
                nn.Conv2d(c_out, c_in*4, kernel_size=3, padding=1, bias=use_bias),
                nn.PixelShuffle(2),
                norm_layer(c_in)
            )
            self.__setattr__('dec_%d_upsample'%l, upsample)
            for i in range(n_residual_blocks):
                self.__setattr__('dec_%d_res_%d'%(l,i), ResidualBlock(c_in, c_in if i==0 else None, norm_layer, use_bias, activation))
        ####################
        # output decoders
        ####################
        for i, c_out in enumerate(output_nc):
            dec_output_i = nn.Sequential(
                channel_mapping(nf, nf, norm_layer, use_bias),
                activation,
                nn.ReflectionPad2d(3),
                nn.Conv2d(nf, c_out, kernel_size=7)
            )
            self.__setattr__('dec_output_%d'%i, dec_output_i)
    
    def forward(self, feats, single_device=False):
        if len(self.gpu_ids) > 1 and (not single_device):
            nn.parallel.data_parallel(self, feats, module_kwargs={'single_device':True})
        else:
            x, hiddens = feats[-1], feats[:-1]
            # decode
            for l in range(self.num_scales-1, -1, -1):
                x = self.__getattr__('dec_%d_upsample'%l)(x)
                for i in range(self.n_residual_blocks):
                    if i == 0:
                        h = hiddens.pop()
                        x = self.__getattr__('dec_%d_res_%d'%(l,i))(x,h)
                    else:
                        x = self.__getattr__('dec_%d_res_%d'%(l,i))(x)

            out = []
            for i in range(len(self.output_nc)):
                out.append(self.__getattr__('dec_output_%d'%i)(x))
            return out


##############################################
# Flow networks
##############################################
class FlowUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
        super(FlowUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv, downnorm]
            up = [uprelu, upconv, upnorm]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.submodule = submodule
        self.predict_flow = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(outer_nc, 2, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        if self.outermost:
            x_ = self.down(x)
            x_, x_pyramid, flow_pyramid = self.submodule(x_)
            x_ = self.up(x_)
            x_out = x_
        elif self.innermost:
            x_pyramid = []
            flow_pyramid = []
            x_ = self.up(self.down(x))
            x_out = torch.cat((x, x_), dim=1)
        else:
            x_ = self.down(x)
            x_, x_pyramid, flow_pyramid = self.submodule(x_)
            x_ = self.up(x_)
            x_out = torch.cat((x, x_), dim=1)
        
        flow = self.predict_flow(x_)
        x_pyramid = [x_] + x_pyramid
        flow_pyramid = [flow] + flow_pyramid
        return x_out, x_pyramid, flow_pyramid

class FlowUnet(nn.Module):
    def __init__(self, input_nc, nf=16, start_scale=2, num_scale=5, norm='batch', gpu_ids=[], max_nf=512):
        super(FlowUnet, self).__init__()
        self.gpu_ids = gpu_ids
        self.nf = nf
        self.norm = norm
        self.start_scale = 2
        self.num_scale = 5

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        conv_downsample = [
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.1)]
        nc = nf
        for i in range(np.log2(start_scale).astype(np.int)):
            conv_downsample += [
                nn.Conv2d(nc, 2*nc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(2*nc),
                nn.LeakyReLU(0.1)
            ]
            nc = nc*2
        self.conv_downsample = nn.Sequential(*conv_downsample)

        unet_block = None
        for l in range(num_scale)[::-1]:
            outer_nc = min(max_nf, nc*2**l)
            inner_nc = min(max_nf, nc*2**(l+1))
            innermost = (l==num_scale-1)
            outermost = (l==0)
            unet_block = FlowUnetSkipConnectionBlock(outer_nc, inner_nc, input_nc=None, submodule=unet_block, norm_layer=norm_layer, innermost=innermost, outermost=outermost)
       
        self.unet_block = unet_block
        self.nf_out = min(max_nf, nc)
        self.predict_vis = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(min(max_nf, nc), 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input, single_device=False):
        if len(self.gpu_ids) > 1 and (not single_device):
            return nn.parallel.data_parallel(self, input, module_kwargs={'single_device': True})
        else:
            x = self.conv_downsample(input)
            feat_out, x_pyr, flow_pyr = self.unet_block(x)
            vis = self.predict_vis(feat_out)

            flow_out = F.upsample(flow_pyr[0], scale_factor=self.start_scale, mode='bilinear')
            vis = F.upsample(vis, scale_factor=self.start_scale, mode='bilinear')
            return flow_out, vis, flow_pyr, feat_out


class FlowUnet_v2(nn.Module):
    '''
    A variation of Unet that use residual blocks instead of convolution layer at each scale
    '''
    def __init__(self, input_nc, nf=64, max_nf=256, start_scale=2, num_scales=7, n_residual_blocks=2, norm='batch', activation=nn.ReLU(False), use_dropout=False, gpu_ids=[]):
        super(FlowUnet_v2, self).__init__()
        self.input_nc = input_nc
        self.nf = nf
        self.max_nf = max_nf
        self.start_scale = start_scale
        self.num_scales = num_scales
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
        self.use_dropout = use_dropout

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()
        
        start_level = np.log2(start_scale).astype(np.int)
        pre_conv = [channel_mapping(input_nc, nf, norm_layer, use_bias)]
        for i in range(start_level):
            c_in = min(nf*(i+1), max_nf)
            c_out = min(nf*(i+2), max_nf)
            pre_conv += [
                ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=use_dropout),
                activation,
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out)
            ]
        self.pre_conv = nn.Sequential(*pre_conv)

        for l in range(num_scales):
            c_in = min(nf * (start_level+l+1), max_nf)
            c_out = min(nf * (start_level+l+2), max_nf)
            # encoding layers
            for i in range(n_residual_blocks):
                self.__setattr__('enc_%d_res_%d'%(l, i), ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=use_dropout))
            downsample = nn.Sequential(
                activation,
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out)
            )
            self.__setattr__('enc_%d_downsample'%l, downsample)
            # decoding layers
            upsample = nn.Sequential(
                activation,
                nn.Conv2d(c_out, c_in*4, kernel_size=3, padding=1, bias=use_bias),
                nn.PixelShuffle(2),
                norm_layer(c_in)
            )
            self.__setattr__('dec_%d_upsample'%l, upsample)
            for i in range(n_residual_blocks):
                self.__setattr__('dec_%d_res_%d'%(l, i), ResidualBlock(c_in, c_in, norm_layer, use_bias, activation, use_dropout))
            # flow prediction
            pred_flow = nn.Sequential(
                activation,
                nn.Conv2d(c_in, 2, kernel_size=3, padding=1, bias=True)
            )
            self.__setattr__('pred_flow_%d'%l, pred_flow)
        # vis prediction
        self.pred_vis = nn.Sequential(
            activation,
            nn.Conv2d(nf*(1+start_level), 3, kernel_size=3, padding=1, bias=True)
        )
    
    def forward(self, x, single_device=False):
        if len(self.gpu_ids) > 1 and (not single_device):
            return nn.parallel.data_parallel(self, x, module_kwargs={'single_device':True})
        else:
            hiddens = []
            flow_pyr = []
            x = self.pre_conv(x)
            # encode
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x = self.__getattr__('enc_%d_res_%d'%(l,i))(x)
                    hiddens.append(x)
                x = self.__getattr__('enc_%d_downsample'%l)(x)
            # decode
            for l in range(self.num_scales-1,-1,-1):
                x = self.__getattr__('dec_%d_upsample'%l)(x)
                for i in range(self.n_residual_blocks-1,-1,-1):
                    h = hiddens.pop()
                    x = self.__getattr__('dec_%d_res_%d'%(l, i))(x,h)
                flow_pyr = [self.__getattr__('pred_flow_%d'%l)(x)] + flow_pyr
            
            feat_out = x
            flow_out = F.upsample(flow_pyr[0], scale_factor=self.start_scale, mode='bilinear')
            vis_out = F.upsample(self.pred_vis(x), scale_factor=self.start_scale, mode='bilinear')
            
            return flow_out, vis_out, flow_pyr, feat_out


##############################################
# Discriminator network
##############################################
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, output_bias = True, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias = output_bias)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input)
        else:
            return self.model(input)
