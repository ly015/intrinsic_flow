from __future__ import print_function, division
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from base_networks import *
from skimage.measure import compare_ssim, compare_psnr
import functools
from networks_flow import warp_acc_flow
###############################################################################
# losses
###############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = F.mse_loss
        else:
            self.loss = F.binary_cross_entropy
    
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class VGGLoss_v2(nn.Module):
    def __init__(self, gpu_ids, content_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0], style_weights=[1.,1.,1.,1.,1.],shifted_style=False):
        super(VGGLoss_v2, self).__init__()
        self.gpu_ids = gpu_ids
        self.shifted_style = shifted_style
        self.content_weights = content_weights
        self.style_weights = style_weights
        self.shift_delta = [[0,2,4,8,16], [0,2,4,8], [0,2,4], [0,2], [0]]
        # self.style_weights = [0,0,1,0,0] # use relu-3 layer feature to compure style loss
        # define vgg
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x]) # relu1_1
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x]) # relu2_1
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x]) # relu3_1
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x]) # relu4_1
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x]) # relu5_1
        for param in self.parameters():
            param.requires_grad = False

        if len(gpu_ids) > 0:
            self.cuda()

    def compute_feature(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    def forward(self, X, Y, mask=None, loss_type='content', device_mode=None):
        '''
        loss_type: 'content', 'style'
        device_mode: multi, single, sub
        '''
        bsz = X.size(0)
        if device_mode is None:
            device_mode = 'multi' if len(self.gpu_ids) > 1 else 'single'

        if device_mode == 'multi':
            if mask is None:
                return nn.parallel.data_parallel(self, (X, Y), module_kwargs={'loss_type': loss_type, 'device_mode': 'sub', 'mask': None}).mean(dim=0)
            else:
                return nn.parallel.data_parallel(self, (X, Y, mask), module_kwargs={'loss_type': loss_type, 'device_mode': 'sub'}).mean(dim=0)
        else:
            features_x = self.compute_feature(self.normalize(X))
            features_y = self.compute_feature(self.normalize(Y))
            if mask is not None:
                features_x = [feat * F.adaptive_max_pool2d(mask, (feat.size(2), feat.size(3))) for feat in features_x]
                features_y = [feat * F.adaptive_max_pool2d(mask, (feat.size(2), feat.size(3))) for feat in features_y]

            # compute content loss
            if loss_type == 'content':
                loss = 0
                for i, (feat_x, feat_y) in enumerate(zip(features_x, features_y)):
                    loss += self.content_weights[i] * F.l1_loss(feat_x, feat_y, reduce=False).view(bsz, -1).mean(dim=1)
            # compute style loss
            if loss_type == 'style':
                loss = 0
                if self.shifted_style:
                    # with cross_correlation
                    for i, (feat_x, feat_y) in enumerate(zip(features_x, features_y)):
                        if self.style_weights[i] > 0:
                            for delta in self.shift_delta[i]:
                                if delta == 0:
                                    loss += self.style_weights[i] * F.mse_loss(self.gram_matrix(feat_x), self.gram_matrix(feat_y), reduce=False).view(bsz, -1).sum(dim=1)
                                else:
                                    loss += 0.5*self.style_weights[i] * \
                                            (F.mse_loss(self.shifted_gram_matrix(feat_x, delta, 0), self.shifted_gram_matrix(feat_y, delta, 0), reduce=False) \
                                            +F.mse_loss(self.shifted_gram_matrix(feat_x, 0, delta), self.shifted_gram_matrix(feat_y, 0, delta), reduce=False)).view(bsz, -1).sum(dim=1)
                else:
                    # without cross_correlation
                    for i, (feat_x, feat_y) in enumerate(zip(features_x, features_y)):
                        if self.style_weights[i] > 0:
                            loss += self.style_weights[i] * F.mse_loss(self.gram_matrix(feat_x), self.gram_matrix(feat_y), reduce=False).view(bsz, -1).sum(dim=1)
                            # loss += self.style_weights[i] * ((self.gram_matrix(feat_x) - self.gram_matrix(feat_y))**2).view(bsz, -1).mean(dim=1)
                
            if device_mode == 'single':
                loss = loss.mean(dim=0)
            return loss

    def normalize(self, x):
        # normalization parameters of input
        mean_1 = x.new([0.5, 0.5, 0.5]).view(1,3,1,1)
        std_1 = x.new([0.5, 0.5, 0.5]).view(1,3,1,1)
        # normalization parameters of output
        mean_2 = x.new([0.485, 0.456, 0.406]).view(1,3,1,1)
        std_2 = x.new([0.229, 0.224, 0.225]).view(1,3,1,1)

        return (x*std_1 + mean_1 - mean_2)/std_2

    def gram_matrix(self, feat):
        bsz, c, h, w = feat.size()
        feat = feat.view(bsz, c, h*w)
        feat_T = feat.transpose(1,2)
        g = torch.matmul(feat, feat_T) / (c*h*w)
        return g

    def shifted_gram_matrix(self, feat, shift_x, shift_y):
        bsz, c, h, w = feat.size()
        assert shift_x<w and shift_y<h
        feat1 = feat[:,:,shift_y:, shift_x:].contiguous().view(bsz, c, -1)
        feat2 = feat[:,:,:(h-shift_y),:(w-shift_x)].contiguous().view(bsz, c, -1)
        g = torch.matmul(feat1, feat2.transpose(1,2)) / (c*h*w)
        return g


###############################################################################
# metrics
###############################################################################
class PSNR(nn.Module):
    def forward(self, images_1, images_2):
        numpy_imgs_1 = images_1.data.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_1 = ((numpy_imgs_1 + 1.0) * 127.5).clip(0,255).astype(np.uint8)
        numpy_imgs_2 = images_2.data.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_2 = ((numpy_imgs_2 + 1.0) * 127.5).clip(0,255).astype(np.uint8)

        psnr_score = []
        for img_1, img_2 in zip(numpy_imgs_1, numpy_imgs_2):
            psnr_score.append(compare_psnr(img_2, img_1))

        return Variable(images_1.data.new(1).fill_(np.mean(psnr_score)))


class SSIM(nn.Module):
    def forward(self, images_1, images_2):
        numpy_imgs_1 = images_1.data.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_1 = ((numpy_imgs_1 + 1.0) * 127.5).clip(0,255).astype(np.uint8)
        numpy_imgs_2 = images_2.data.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_2 = ((numpy_imgs_2 + 1.0) * 127.5).clip(0,255).astype(np.uint8)

        ssim_score = []
        for img_1, img_2 in zip(numpy_imgs_1, numpy_imgs_2):
            ssim_score.append(compare_ssim(img_1, img_2, multichannel=True))

        return Variable(images_1.data.new(1).fill_(np.mean(ssim_score)))

###############################################################################
# networks
###############################################################################
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_bias, activation=nn.ReLU, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.dim = dim
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       activation()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    def print(self):
        print('ResnetBlock: x_dim=%d'%self.dim)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm='batch', use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', output_tanh=True):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.output_tanh = output_tanh
        activation = nn.ReLU
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        elif norm == 'none':
            norm_layer = Identity
            use_bias = True

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 activation()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      activation()]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation = activation, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      activation()]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if output_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, output_feature=False, single_device=False):
        if len(self.gpu_ids) > 1 and (not single_device):
            return nn.parallel.data_parallel(self, input, module_kwargs={'output_feature': output_feature, 'single_device': True})
        else:
            if not output_feature:
                return self.model(input)
            else:
                feat_idx = len(self.model)-6 if self.output_tanh else len(self.model)-5
                x = input
                for module_idx in range(len(self.model)):
                    x = self.model[module_idx](x)
                    if module_idx == feat_idx:
                        feat = x.clone()
                return x, feat
                
class UnetSkipConnectionBlock_old(nn.Module):
    '''
    This is a wrong version of unet block, which has an incorrect norm layer when outermost==True
    For now we keep this to keep compatible to some trained model, which will be retrained with write unet block in the future.
    '''
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock_old, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = not (norm_layer.func == nn.BatchNorm2d)
        else:
            use_bias = not (norm_layer == nn.BatchNorm2d)

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        dropout_rate = 0.2

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(dropout_rate)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = not (norm_layer.func == nn.BatchNorm2d)
        else:
            use_bias = not (norm_layer == nn.BatchNorm2d)

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        dropout_rate = 0.2

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv, downnorm]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(dropout_rate)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator_v2(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm='batch', use_dropout=False, gpu_ids=[], max_nf=512, output_tanh=True):
        super(UnetGenerator_v2, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_tanh = output_tanh
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'none':
            norm_layer = Identity

        unet_block = None
        for l in range(num_downs)[::-1]:
            outer_nc = min(max_nf, ngf*2**l)
            inner_nc = min(max_nf, ngf*2**(l+1))
            innermost = (l==num_downs-1)
            unet_block = UnetSkipConnectionBlock_old(outer_nc, inner_nc, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, innermost=innermost)

        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3), #0
            norm_layer(ngf), #1
            nn.LeakyReLU(0.2), #2
            unet_block, #3
            nn.LeakyReLU(0.2), #4
            nn.Conv2d(2*ngf, output_nc, kernel_size=7, padding=3)] #5
        if output_tanh:
            model.append(nn.Tanh()) #6

        self.model = nn.Sequential(*model)

    def forward(self, input, output_feature=False, single_device=False):
        if len(self.gpu_ids) > 1 and (not single_device):
            return nn.parallel.data_parallel(self, input, module_kwargs={'output_feature': output_feature, 'single_device': True})
        else:
            if not output_feature:
                return self.model(input)
            else:
                x = input
                for module_idx in range(len(self.model)):
                    x = self.model[module_idx](x)
                    if module_idx == 3:
                        feat = x.clone()
                return x, feat

class UnetGenerator_v1(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm='batch', use_dropout=False, gpu_ids=[], max_nf=512, output_tanh=True, old_unet_block=False):
        super(UnetGenerator_v1, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_tanh = output_tanh
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'none':
            norm_layer = Identity

        unet_block = None
        for l in range(num_downs)[::-1]:
            innermost = (l==num_downs-1)
            outermost = (l==0)
            if not outermost:
                outer_nc = min(max_nf, ngf*2**(l-1))
                inner_nc = min(max_nf, ngf*2**l)
            else:
                outer_nc = output_nc
                inner_nc = min(max_nf, ngf)

            unet_block = UnetSkipConnectionBlock(outer_nc, inner_nc, input_nc=input_nc if outermost else None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, innermost=innermost, outermost=outermost)

        model=[unet_block] #1
        if output_tanh:
            model.append(nn.Tanh()) #4
        self.model = nn.Sequential(*model)

    def forward(self, input, output_feature=False, single_device=False):
        if len(self.gpu_ids) > 1 and (not single_device):
            return nn.parallel.data_parallel(self, input, module_kwargs={'output_feature': output_feature, 'single_device': True})
        else:
            if not output_feature:
                return self.model(input)
            else:
                x = input
                for module_idx in range(len(self.model)):
                    x = self.model[module_idx](x)
                    if module_idx == 0:
                        feat = x.clone()
                return x, feat

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

class VUnetResidualBlock(nn.Module):
    def __init__(self, dim_1, dim_2, norm_layer, use_bias, activation=nn.ReLU(False), use_dropout=False):
        super(VUnetResidualBlock, self).__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.use_dropout = use_dropout
        if norm_layer is None:
            use_bias = True
        if dim_2 <= 0:
            self.conv = nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1, bias=use_bias)
            self.norm_layer = norm_layer(dim_1) if norm_layer is not None else None
        else:
            self.conv = nn.Conv2d(2 * dim_1, dim_1, kernel_size=3, padding=1, bias=use_bias)
            self.norm_layer = norm_layer(dim_1) if norm_layer is not None else None
            self.conv_2 = nn.Conv2d(dim_2, dim_1, kernel_size=1, padding=0, bias=use_bias)
            self.norm_layer_2 = norm_layer(2 * dim_1) if norm_layer is not None else None
        self.activation = activation

    def forward(self, x, a=None):
        if a is None:
            residual = x
        else:
            assert self.dim_2 > 0
            a = self.conv_2(self.activation(a))
            residual = torch.cat((x, a), dim=1)
            if self.norm_layer_2 is not None:
                residual = self.norm_layer_2(residual)
        
        residual = self.activation(residual)
        if self.use_dropout:
            residual = F.dropout(residual, p=0.5, training=True)

        residual = self.conv(residual)
        if self.norm_layer is not None:
            residual = self.norm_layer(residual)

        out = x + residual
        return out

class VariationalUnet(nn.Module):
    def __init__(self, input_nc_dec, input_nc_enc, output_nc, nf, max_nf, input_size, n_latent_scales, bottleneck_factor, box_factor, n_residual_blocks, norm, activation, use_dropout, gpu_ids, output_tanh=True):
        super(VariationalUnet, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_nc = output_nc
        self.input_nc_dec = input_nc_dec
        self.input_size_dec = input_size
        self.n_scales_dec = 1 + int(np.round(np.log2(input_size))) - bottleneck_factor

        self.input_nc_enc = input_nc_enc
        self.input_size_enc = input_size // 2**box_factor
        self.n_scales_enc = self.n_scales_dec - box_factor

        self.n_latent_scales = n_latent_scales
        self.bottleneck_factor = bottleneck_factor
        self.box_factor = box_factor
        self.n_residual_blocks = n_residual_blocks
        self.output_tanh = output_tanh

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            self.use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            self.use_bias = True
        elif norm == 'none':
            norm_layer = Identity
            self.use_bias = True
        # define enc_up network
        c_in = min(nf*2**box_factor, max_nf)
        hidden_c = [] # hidden space dims
        self.enc_up_pre_conv = nn.Sequential(
            nn.Conv2d(input_nc_enc, c_in, kernel_size=1),
            norm_layer(c_in))

        for l in range(self.n_scales_enc):
            spatial_shape = self.input_size_enc / 2**l
            nl = None if spatial_shape == 1 else norm_layer
            for i in range(n_residual_blocks):
                self.__setattr__('enc_up_%d_res_%d' % (l, i), VUnetResidualBlock(c_in, 0, nl, self.use_bias, activation, use_dropout))
                hidden_c.append(c_in)
            if l + 1 < self.n_scales_enc:
                c_out = min(2*c_in, max_nf)
                if spatial_shape <= 2:
                    downsample = nn.Sequential(
                        activation,
                        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=True)
                        )
                else:
                    downsample = nn.Sequential(
                        activation,
                        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=self.use_bias),
                        norm_layer(c_out)
                        )
                self.__setattr__('enc_up_%d_downsample'%l, downsample)
                c_in = c_out
        # define enc_down network
        self.enc_down_pre_conv = nn.Conv2d(c_in, c_in, kernel_size=1)
        for l in range(n_latent_scales):
            spatial_shape = self.input_size_enc / 2**(self.n_scales_enc - l - 1)
            nl = None if spatial_shape == 1 else norm_layer
            for i in range(n_residual_blocks//2):
                c_a = hidden_c.pop()
                self.__setattr__('enc_down_%d_res_%d' % (l, i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))

            self.__setattr__('enc_down_%d_latent'%l, nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, bias=True))

            for i in range(n_residual_blocks//2, n_residual_blocks):
                c_a = c_in + hidden_c.pop()
                self.__setattr__('enc_down_%d_res_%d' % (l, i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))

            if l + 1 < n_latent_scales:
                c_out = hidden_c[-1]
                upsample = nn.Sequential(
                    activation,
                    nn.Conv2d(c_in, c_out*4, kernel_size=3, padding=1, bias=self.use_bias),
                    nn.PixelShuffle(2),
                    norm_layer(c_out)
                    )
                self.__setattr__('enc_down_%d_upsample'%l, upsample)
                c_in = c_out
        # define dec_up network
        c_in = nf
        hidden_c = []
        self.dec_up_pre_conv = nn.Sequential(
            nn.Conv2d(input_nc_dec, c_in, kernel_size=1),
            norm_layer(c_in))
        for l in range(self.n_scales_dec):
            spatial_shape = self.input_size_dec / 2**l
            nl = None if spatial_shape==1 else norm_layer
            for i in range(n_residual_blocks):
                self.__setattr__('dec_up_%d_res_%d'%(l, i), VUnetResidualBlock(c_in, 0, nl, self.use_bias, activation, use_dropout))
                hidden_c.append(c_in)
            if l + 1 < self.n_scales_dec:
                c_out = min(2*c_in, max_nf)
                if spatial_shape <= 2:
                    downsample = nn.Sequential(
                        activation,
                        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=True)
                        )
                else:
                    downsample = nn.Sequential(
                        activation,
                        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=self.use_bias),
                        norm_layer(c_out)
                        )
                self.__setattr__('dec_up_%d_downsample'%l, downsample)
                c_in = c_out
        # define dec_down network
        self.dec_down_pre_conv = nn.Conv2d(c_in, c_in, kernel_size=1)
        for l in range(self.n_scales_dec):
            spatial_shape = self.input_size_dec / 2**(self.n_scales_dec - l - 1)
            nl = None if spatial_shape==1 else norm_layer
            for i in range(n_residual_blocks//2):
                c_a = hidden_c.pop()
                self.__setattr__('dec_down_%d_res_%d' % (l, i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))
            if l < n_latent_scales:
                if spatial_shape == 1:
                    # no spatial correlation
                    self.__setattr__('dec_down_%d_latent'%l, nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, bias=True))
                else:
                    # four autoregressively modeled groups
                    for j in range(4):
                        self.__setattr__('dec_down_%d_latent_%d'%(l,j), nn.Conv2d(c_in*4, c_in, kernel_size=3, padding=1, bias=True))
                        if j + 1 < 4:
                            self.__setattr__('dec_down_%d_ar_%d'%(l,j), VUnetResidualBlock(c_in*4, c_in, None, self.use_bias, activation, use_dropout))
                for i in range(n_residual_blocks//2, n_residual_blocks):
                    if spatial_shape == 1:
                        nin = nn.Conv2d(c_in*2, c_in, kernel_size=1, bias=True)
                    else:
                        nin = nn.Sequential(nn.Conv2d(c_in*2, c_in, kernel_size=1, bias=self.use_bias), norm_layer(c_in))
                    self.__setattr__('dec_down_%d_nin_%d'%(l,i), nin)
                    c_a = hidden_c.pop()
                    self.__setattr__('dec_down_%d_res_%d'%(l,i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))
            else:
                for i in range(n_residual_blocks//2, n_residual_blocks):
                    c_a = hidden_c.pop()
                    self.__setattr__('dec_down_%d_res_%d'%(l,i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))

            if l+1 < self.n_scales_dec:
                c_out = hidden_c[-1]
                upsample = nn.Sequential(
                    activation,
                    nn.Conv2d(c_in, c_out*4, kernel_size=3, padding=1, bias=self.use_bias),
                    nn.PixelShuffle(2),
                    norm_layer(c_out)
                    )
                self.__setattr__('dec_down_%d_upsample'%l, upsample)
                c_in = c_out
        # define final decode layer
        dec_output = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c_in, output_nc, kernel_size=7, padding=0, bias=True)]
        if output_tanh:
            dec_output.append(nn.Tanh())
        self.dec_output = nn.Sequential(*dec_output)

    def enc_up(self, x, c):
        '''
        Input:
            x: appearance input (rgb image)
            c: pose input (heat map)
        Output:
            hs: hidden units
        '''
        # according to the paper, xc=[x, c]. while in the code they use xc=x.
        # xc = torch.cat((x, c), dim=1)
        xc = x
        assert x.size(1) == self.input_nc_enc
        if not xc.size(2)==xc.size(3)==self.input_size_enc:
            xc = F.adaptive_avg_pool2d(xc, self.input_size_enc)

        hs = []
        h = self.enc_up_pre_conv(xc)
        for l in range(self.n_scales_enc):
            for i in range(self.n_residual_blocks):
                h = self.__getattr__('enc_up_%d_res_%d' % (l, i))(h)
                hs.append(h)
            if l + 1 < self.n_scales_enc:
                h = self.__getattr__('enc_up_%d_downsample'%l)(h)
        return hs

    def enc_down(self, gs):
        '''
        Input:
            gs: input hiddent units
        Output:
            hs: hidden units
            qs: posteriors
            zs: samples from posterior
        '''
        hs = []
        qs = []
        zs = []

        h = self.enc_down_pre_conv(gs[-1])
        for l in range(self.n_latent_scales):
            for i in range(self.n_residual_blocks//2):
                h = self.__getattr__('enc_down_%d_res_%d'%(l, i))(h, gs.pop())
                hs.append(h)
            # posterior
            q = self.__getattr__('enc_down_%d_latent'%l)(h)
            qs.append(q)
            # posterior sample
            z = self.latent_sample(q)
            zs.append(z)
            # sample feedback
            for j in range(self.n_residual_blocks//2, self.n_residual_blocks):
                gz = torch.cat((gs.pop(), z), dim=1)
                h = self.__getattr__('enc_down_%d_res_%d'%(l, j))(h, gz)
                hs.append(h)
            # up sample
            if l + 1 < self.n_latent_scales:
                h = self.__getattr__('enc_down_%d_upsample'%l)(h)

        return hs, qs, zs

    def dec_up(self, c):
        '''
        Input:
            c: pose input
        Output:
            hs: hidden units
        '''
        if not c.size(2)==c.size(3)==self.input_size_dec:
            c = F.adaptive_avg_pool2d(c, self.input_size_dec)
        assert c.size(1) == self.input_nc_dec

        hs = []
        h = self.dec_up_pre_conv(c)
        for l in range(self.n_scales_dec):
            for i in range(self.n_residual_blocks):
                h = self.__getattr__('dec_up_%d_res_%d' % (l, i))(h)
                hs.append(h)
            if l + 1 < self.n_scales_dec:
                h = self.__getattr__('dec_up_%d_downsample'%l)(h)
        return hs

    def dec_down(self, gs, zs_posterior, training):
        '''
        Input:
            gs: input hidden units
            zs_posterior: samples from posterior. from LR layer to HR layer
        Output:
            hs: hidden units
            ps: prior
            zs: samples from prior            
        '''
        hs = []
        ps = []
        zs = []
        h = self.dec_down_pre_conv(gs[-1])
        for l in range(self.n_scales_dec):
            for i in range(self.n_residual_blocks//2):
                h = self.__getattr__('dec_down_%d_res_%d'%(l,i))(h, gs.pop())
                hs.append(h)
            if l < self.n_latent_scales:
                spatial_shape = self.input_size_dec / 2**(self.n_scales_dec - l - 1)
                # n_h_channels = hs[-1].size(1)
                if spatial_shape == 1:
                    p = self.__getattr__('dec_down_%d_latent'%l)(h)
                    ps.append(p)
                    z_prior = self.latent_sample(p)
                    zs.append(z_prior)
                else:
                    # four autoregressively modeled groups
                    if training:
                        z_posterior_groups = self.space_to_depth(zs_posterior[0], scale=2) # the order of zs_posterior is from LR to HR
                        split_size = z_posterior_groups.size(1)//4
                        z_posterior_groups = list(z_posterior_groups.split(split_size, dim=1))
                    p_groups = []
                    z_groups = []
                    p_feat = self.space_to_depth(h, scale=2)
                    for i in range(4):
                        p_group = self.__getattr__('dec_down_%d_latent_%d'%(l,i))(p_feat)
                        p_groups.append(p_group)
                        z_group = self.latent_sample(p_group)
                        z_groups.append(z_group)
                        # ar feedback sampled from
                        if training:
                            feedback = z_posterior_groups.pop(0)
                        else:
                            feedback = z_group
                        if i + 1 < 4:
                            p_feat = self.__getattr__('dec_down_%d_ar_%d'%(l,i))(p_feat, feedback)
                    if training:
                        assert not z_posterior_groups
                    p = self.depth_to_space(torch.cat(p_groups, dim=1), scale=2)
                    ps.append(p)
                    z_prior = self.depth_to_space(torch.cat(z_groups, dim=1), scale=2)
                    zs.append(z_prior)
                # vae feedback sampled from
                if training:
                    # posterior
                    z = zs_posterior.pop(0)
                else:
                    # prior
                    z = z_prior
                for i in range(self.n_residual_blocks//2, self.n_residual_blocks):
                    h = torch.cat((h, z), dim=1)
                    h = self.__getattr__('dec_down_%d_nin_%d'%(l,i))(h)
                    h = self.__getattr__('dec_down_%d_res_%d'%(l,i))(h, gs.pop())
                    hs.append(h)
            else:
                for i in range(self.n_residual_blocks//2, self.n_residual_blocks):
                    h = self.__getattr__('dec_down_%d_res_%d'%(l,i))(h, gs.pop())
                    hs.append(h)

            if l + 1 < self.n_scales_dec:
                h = self.__getattr__('dec_down_%d_upsample'%(l))(h)

        assert not gs
        if training:
            assert not zs_posterior
        return hs, ps, zs

    def dec_to_image(self, h):
        return self.dec_output(h)
    
    def latent_sample(self, p):
        mean = p
        stddev = 1.0
        z = p + stddev * p.new(p.size()).normal_()
        return z

    def latent_kl(self, p, q):
        n = p.size(0)
        kl = 0.5 * (p-q)*(p-q)
        kl = kl.view(n, -1)
        kl = kl.sum(dim=1).mean()
        return kl
    
    def depth_to_space(self, x, scale=2):
        ''' from [n,c*scale^2,h,w] to [n,c,h*scale,w*scale]'''
        return F.pixel_shuffle(x, scale)

    def space_to_depth(self, x, scale=2):
        ''' from [n,c,h*scale,w*scale] to [n,c*scale^2,h,w]'''
        n, c, h, w = x.size()
        assert h%scale==0 and w%scale==0
        nh, nw = h//scale, w//scale
        x = x.unfold(2,scale,scale).unfold(3,scale,scale).contiguous()
        x = x.view(n,c,nh,nw,scale*scale).transpose(3,4).transpose(2,3).contiguous()
        x = x.view(n,c*scale*scale,nh,nw)
        return x

    def train_forward_pass(self, x_ref, c_ref, c_tar):
        # encoder
        hs = self.enc_up(x_ref, c_ref)
        es, qs, zs_posterior = self.enc_down(hs)
        # decoder
        gs = self.dec_up(c_tar)
        ds, ps, zs_prior = self.dec_down(gs, zs_posterior, training=True)
        img = self.dec_to_image(ds[-1])
        return img, qs, ps, ds

    def test_forward_pass(self, c_tar):
        # decoder
        gs = self.dec_up(c_tar)
        ds, ps, zs_prior = self.dec_down(gs, [], training=False)
        img = self.dec_to_image(ds[-1])
        return img, ds

    def transfer_pass(self, x_ref, c_ref, c_tar):
        use_mean = True
        # infer latent code
        hs = self.enc_up(x_ref, c_ref)
        es, qs, zs_posterior = self.enc_down(hs)
        zs_mean = [q.clone() for q in qs]
        gs = self.dec_up(c_tar)

        if use_mean:
            ds, ps, zs_prior = self.dec_down(gs, zs_mean, training=True)
        else:
            ds, ps, zs_prior = self.dec_down(gs, zs_posterior, training=True)
        img = self.dec_to_image(ds[-1])
        return img, qs, ps, ds

    def forward(self, x_ref, c_ref, c_tar, mode='train', output_feature=False, single_device=False):
        if len(self.gpu_ids) > 1 and not single_device:
            return nn.parallel.data_parallel(self, (x_ref, c_ref, c_tar), module_kwargs={'mode':mode, 'single_device':True, 'output_feature': output_feature})
        else:
            if mode == 'train':
                img, qs, ps, ds = self.train_forward_pass(x_ref, c_ref, c_tar)
                if output_feature:
                    return img, qs, ps, ds
                else:
                    return img, qs, ps
            elif mode == 'test':
                img, ds = self.test_forward_pass(c_tar)
                if output_feature:
                    return img, ds
                else:
                    return img
            elif mode == 'transfer':
                img, qs, ps, ds = self.transfer_pass(x_ref, c_ref, c_tar)
                if output_feature:
                    return img, qs, ps, ds
                else:
                    return img, qs, ps
            else:
                raise NotImplementedError()

##############################################
# Unet_v3 and FeatureWarpingUnet
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
    Derived from VUnetResidualBlock.
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

class UnetGenerator_v3(nn.Module):
    '''
    A variation of Unet that use residual blocks instead of convolution layer at each scale
    '''
    def __init__(self, input_nc, output_nc, nf=64, max_nf=256, num_scales=7, n_residual_blocks=2, norm='batch', activation=nn.ReLU(False), use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_v3, self).__init__()
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


class UnetGenerator_v4(nn.Module):
    '''
    A variation of Unet that use residual blocks instead of convolution layer at each scale
    '''
    def __init__(self, input_nc, output_nc = [3], nf=64, max_nf=256, num_scales=7, n_residual_blocks=2, norm='batch', activation=nn.ReLU(False), use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_v4, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc if isinstance(output_nc, list) else [output_nc]
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

        for i, c_out in enumerate(output_nc):
            dec_output_i = nn.Sequential(
                channel_mapping(nf, nf, norm_layer, use_bias),
                activation,
                nn.ReflectionPad2d(3),
                nn.Conv2d(nf, c_out, kernel_size=7, padding=0, bias=True)
                )
            self.__setattr__('dec_output_%d'%i, dec_output_i)
    
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
            out = []
            for i in range(len(self.output_nc)):
                out.append(self.__getattr__('dec_output_%d'%i)(x))
            return out

class UnetGenerator_FeatureWarping(nn.Module):
    '''
    A variation of Unet architecture, similar to deformable gan. It contains two encoders: one for target pose and one for appearance. The feature map of appearance encoder will be warped to target pose, guided
    by input flow. There are skip connections from both encoders to the decoder.
    '''
    def __init__(self, pose_nc, appearance_nc, output_nc, aux_output_nc=[], nf=32, max_nf=128, num_scales=7, num_warp_scales=5, n_residual_blocks=2, norm='batch', vis_mode='none', activation=nn.ReLU(False), use_dropout=False, no_end_norm=False, gpu_ids=[]):
        '''
        vis_mode: ['none', 'hard_gate', 'soft_gate', 'residual']
        no_end_norm: remove normalization layer at the start and the end.
        '''
        super(UnetGenerator_FeatureWarping, self).__init__()
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


class UnetGenerator_Decoder(nn.Module):
    '''
    Decoder that decodes hierarachical features. Support multi-task output. Used as an external decoder of a UnetGenerator_FeatureWarping network
    '''
    def __init__(self, output_nc=[], nf=32, max_nf=128, num_scales=7, n_residual_blocks=2, norm='batch', activation=nn.ReLU(False), gpu_ids=[]):
        super(UnetGenerator_Decoder, self).__init__()
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
