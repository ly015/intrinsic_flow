from __future__ import division, print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.optim import lr_scheduler
import functools
import numpy as np
from skimage.measure import compare_ssim, compare_psnr

###############################################################################
# model helper functions
###############################################################################
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def get_norm_layer(norm_type = 'instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine =False)
    elif norm_type == 'none':
        norm_layer = Identity
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

###############################################################################
# parameter initialize
###############################################################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.normal_(m.weight, 0.0, 0.02)
    elif classname.startswith('Linear'):
        init.normal_(m.weight, 0.0, 0.02)
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.02)

    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

def weights_init_normal2(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.normal_(m.weight, 0.0, 0.001)
    elif classname.startswith('Linear'):
        init.normal_(m.weight, 0.0, 0.001)
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.001)

    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.xavier_normal_(m.weight, gain=0.02)
    elif classname.startswith('Linear'):
        init.xavier_normal_(m.weight, gain=0.02)
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.02)
    
    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.startswith('Linear'):
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.02)

    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.orthogonal_(m.weight, gain=1)
    elif classname.startswith('Linear'):
        init.orthogonal_(m.weight, gain=1)
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.02)
    
    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'normal2':
        net.apply(weights_init_normal2)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

###############################################################################
# Optimizer and Scheduler
###############################################################################
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        if opt.resume_train:
            last_epoch = int(opt.last_epoch) -1 
        else:
            last_epoch = -1
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay, gamma=opt.lr_gamma, last_epoch=last_epoch)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

###############################################################################
# Loss Helper
###############################################################################
class SmoothLoss():
    '''
    wrapper of pytorch loss layer.
    '''
    def __init__(self, crit):
        self.crit = crit
        self.max_size = 100000
        self.clear()

    def __call__(self, input_1, input_2, *extra_input):
        loss = self.crit(input_1, input_2, *extra_input)
        self.weight_buffer.append(input_1.size(0))

        if isinstance(loss, Variable):
            self.buffer.append(loss.data.item())
        elif isinstance(loss, torch.Tensor):
            self.buffer.append(loss.data.item())
        else:
            self.buffer.append(loss)

        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size::]
            self.weight_buffer = self.weight_buffer[-self.max_size::]

        return loss

    def clear(self):
        self.buffer = []
        self.weight_buffer = []

    def smooth_loss(self, clear = False):
        if len(self.weight_buffer) == 0:
            loss = 0
        else:
            loss = sum([l * w for l, w in zip(self.buffer, self.weight_buffer)]) / sum(self.weight_buffer)            
        if clear:
            self.clear()
        return loss

class CalcGradNorm(object):
    '''
    example:
        y = model(x)
        with CalcGradNorm(model) as cgn:
            y.backward()
            grad_norm = cgn.get_grad_norm()
    '''
    def __init__(self, module):
        super(CalcGradNorm, self).__init__()
        self.module = module
    
    def __enter__(self):
        self.grad_list = [p.grad.clone() if p.grad is not None else None for p in self.module.parameters()]
        return self
    
    def __exit__(self, type, value, traceback):
        pass
    
    def get_grad_norm(self):
        grad_norm = self.module.parameters().next().new_zeros([])
        new_grad_list = []
        for i, p in enumerate(self.module.parameters()):
            if p.grad is None:
                assert self.grad_list[i] is None, 'gradient information is missing. maybe caused by calling "zero_grad()"'
                new_grad_list.append(None)
            else:
                g = p.grad.clone()
                new_grad_list.append(g)
                if self.grad_list[i] is None:
                    grad_norm += g.norm()
                else:
                    grad_norm += (g-self.grad_list[i]).norm()
        
        self.grad_list = new_grad_list
        return grad_norm.detach()
        


###############################################################################
# Losses and metrics
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


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids, content_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0], style_weights=[1.,1.,1.,1.,1.],shifted_style=False):
        super(VGGLoss, self).__init__()
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

def EPE(input_flow, target_flow, vis_mask):
    '''
    compute endpoint-error
    input_flow: (N,C=2,H,W)
    target_flow: (N,C=2,H,W)
    vis_mask: (N,1,H,W)
    '''
    # return (torch.norm(target_flow-input_flow, p=2, dim=1) * vis_mask).mean()
    bsz = input_flow.size(0)
    epe = (target_flow - input_flow).norm(dim=1, p=2, keepdim=True) * vis_mask #(N, 1, H, W)
    count = vis_mask.view(bsz,-1).sum(dim=1, keepdim=True) #(N, 1)
    return (epe.view(bsz,-1) / (count*bsz+1e-8)).sum()

def L1(input_flow, target_flow, vis_mask):
    '''
    compute l1-loss
    input_flow: (N,C=2,H,W)
    target_flow: (N,C=2,H,W)
    vis_mask: (N,1,H,W)
    '''
    bsz = input_flow.size(0)
    err = (target_flow-input_flow).abs() * vis_mask #(N, 2, H, W)
    count = vis_mask.view(bsz,-1).sum(dim=1, keepdim=True) #(N, 1)
    return (err.view(bsz,-1) / (count*bsz*2+1e-8)).sum()

def L2(input_flow, target_flow, vis_mask):
    '''
    compute l1-loss
    input_flow: (N,C=2,H,W)
    target_flow: (N,C=2,H,W)
    vis_mask: (N,1,H,W)
    '''
    bsz=  input_flow.size(0)
    err = (target_flow-input_flow).norm(dim=1, p=2, keepdim=True)  * vis_mask
    count = vis_mask.view(bsz,-1).sum(dim=1, keepdim=True) #(N, 1)
    return (err.view(bsz,-1) / (count*bsz + 1e-8)).sum()



class MultiScaleFlowLoss(nn.Module):
    def __init__(self, start_scale=2, num_scale=5, loss_type='l1'):
        super(MultiScaleFlowLoss, self).__init__()
        self.start_scale = start_scale
        self.num_scale = num_scale
        self.loss_type = loss_type
        self.loss_weights = [0.005, 0.01, 0.02, 0.08, 0.32] + [0.32]*(num_scale-5)
        self.div_flow = 0.05 # follow flownet and pwc net, but don't konw the reason
        self.avg_pools = [nn.AvgPool2d(self.start_scale*(2**scale), self.start_scale*(2**scale)) for scale in range(num_scale)]
        self.max_pools = [nn.MaxPool2d(self.start_scale*(2**scale), self.start_scale*(2**scale)) for scale in range(num_scale)]
        if loss_type == 'l1':
            self.loss_func = L1
        elif loss_type == 'l2':
            self.loss_func = L2
        
    def forward(self, input_flows, target_flow, vis_mask, output_full_losses=False):
        loss = 0
        epe = 0
        full_losses = []
        target_flow = self.div_flow * target_flow
        for i, input_ in enumerate(input_flows):
            # for pytorch v0.4.0 and earlier
            target_ = self.avg_pools[i](target_flow)
            mask_ = self.max_pools[i](vis_mask)
            assert input_.is_same_size(target_), 'scale %d size mismatch: input(%s) vs. target(%s)' % (i, input_.size(), target_.size())
            loss_i = self.loss_func(input_, target_, mask_)
            loss += self.loss_weights[i] * loss_i
            epe +=  self.loss_weights[i] * EPE(input_, target_, mask_)
            full_losses = full_losses + [loss_i]
        if output_full_losses:
            return loss, epe, full_losses
        else:
            return loss, epe

class SS_FlowLoss(nn.Module):
    '''
    segmentation sensitive flow loss
    this loss function only penalize pixels where the flow points to a wrong segmentation area
    '''
    def __init__(self, loss_type='l1'):
        super(SS_FlowLoss, self).__init__()
        self.div_flow = 0.05
        self.loss_type = loss_type
    
    def forward(self, input_flow, target_flow, seg_1, seg_2, vis_2):
        '''
        input_flow: (bsz, 2, h, w)
        target_flow: (bsz, 2, h, w) note that there is scale factor between input_flow and target_flow, which is self.div_flow
        seg_1, seg_2: (bsz, ns, h, w) channel-0 should be background
        vis_2: (bsz, 1, h, w) visibility map of image_2
        '''
        with torch.no_grad():
            seg_1 = seg_1[:,1::,...]
            seg_2 = seg_2[:,1::,...] # doesn't consider background
            seg_1w = warp_acc_flow(seg_1, input_flow)
            seg_1w = (seg_1w>0).float()
            mask = (seg_2*(1-seg_1w)).sum(dim=1, keepdim=True)
            mask = mask * (vis_2==0).float()
        err = (input_flow - target_flow).mul(self.div_flow) * mask
        if self.loss_type == 'l1':
            loss = err.abs().mean()
        elif self.loss_type == 'l2':
            loss = err.norm(p=2,dim=1).mean()
        return loss


class MeanAP():
    '''
    compute meanAP
    '''

    def __init__(self):
        self.clear()

    def clear(self):
        self.score = None
        self.label = None

    def add(self, new_score, new_label):

        inputs = [new_score, new_label]

        for i in range(len(inputs)):

            if isinstance(inputs[i], list):
                inputs[i] = np.array(inputs[i], dtype = np.float32)

            elif isinstance(inputs[i], np.ndarray):
                inputs[i] = inputs[i].astype(np.float32)

            elif isinstance(inputs[i], torch.Tensor):
                inputs[i] = inputs[i].cpu().numpy().astype(np.float32)

            elif isinstance(inputs[i], Variable):
                inputs[i] = inputs[i].data.cpu().numpy().astype(np.float32)

        new_score, new_label = inputs
        assert new_score.shape == new_label.shape, 'shape mismatch: %s vs. %s' % (new_score.shape, new_label.shape)

        self.score = np.concatenate((self.score, new_score), axis = 0) if self.score is not None else new_score
        self.label = np.concatenate((self.label, new_label), axis = 0) if self.label is not None else new_label

    def compute_mean_ap(self):

        score, label = self.score, self.label

        assert score is not None and label is not None
        assert score.shape == label.shape, 'shape mismatch: %s vs. %s' % (score.shape, label.shape)
        assert(score.ndim == 2)
        M, N = score.shape[0], score.shape[1]

        # compute tp: column n in tp is the n-th class label in descending order of the sample score.
        index = np.argsort(score, axis = 0)[::-1, :]
        tp = label.copy().astype(np.float)
        for i in xrange(N):
            tp[:, i] = tp[index[:,i], i]
        tp = tp.cumsum(axis = 0)

        m_grid, n_grid = np.meshgrid(range(M), range(N), indexing = 'ij')
        tp_add_fp = m_grid + 1    
        num_truths = np.sum(label, axis = 0)
        # compute recall and precise
        rec = tp / (num_truths+1e-8)
        prec = tp / (tp_add_fp+1e-8)

        prec = np.append(np.zeros((1,N), dtype = np.float), prec, axis = 0)
        for i in xrange(M-1, -1, -1):
            prec[i, :] = np.max(prec[i:i+2, :], axis = 0)
        rec_1 = np.append(np.zeros((1,N), dtype = np.float), rec, axis = 0)
        rec_2 = np.append(rec, np.ones((1,N), dtype = np.float), axis = 0)
        AP = np.sum(prec * (rec_2 - rec_1), axis = 0)
        AP[np.isnan(AP)] = -1 # avoid error caused by classes that have no positive sample

        assert((AP <= 1).all())

        AP = AP * 100.
        meanAP = AP[AP >= 0].mean()

        return meanAP, AP

    def compute_recall(self, k = 3):
        '''
        compute recall using method in DeepFashion Paper
        '''
        score, label = self.score, self.label
        tag = np.where((-score).argsort().argsort() < k, 1, 0)
        tag_rec = tag * label

        count_rec = tag_rec.sum(axis = 1)
        count_gt = label.sum(axis = 1)

        # set recall=1 for sample with no positive attribute label
        no_pos_attr = (count_gt == 0).astype(count_gt.dtype)
        count_rec += no_pos_attr
        count_gt += no_pos_attr

        rec = (count_rec / count_gt).mean() * 100.

        return rec

###############################################################################
# image similarity metrics
###############################################################################
class PSNR(nn.Module):
    def forward(self, images_1, images_2):
        numpy_imgs_1 = images_1.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_1 = ((numpy_imgs_1 + 1.0) * 127.5).clip(0,255).astype(np.uint8)
        numpy_imgs_2 = images_2.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_2 = ((numpy_imgs_2 + 1.0) * 127.5).clip(0,255).astype(np.uint8)

        psnr_score = []
        for img_1, img_2 in zip(numpy_imgs_1, numpy_imgs_2):
            psnr_score.append(compare_psnr(img_2, img_1))

        return Variable(images_1.data.new(1).fill_(np.mean(psnr_score)))


class SSIM(nn.Module):
    def forward(self, images_1, images_2, mask=None):
        numpy_imgs_1 = images_1.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_1 = ((numpy_imgs_1 + 1.0) * 127.5).clip(0,255).astype(np.uint8)
        numpy_imgs_2 = images_2.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_2 = ((numpy_imgs_2 + 1.0) * 127.5).clip(0,255).astype(np.uint8)
        if mask is not None:
            mask = mask.cpu().numpy().transpose(0,2,3,1).astype(np.uint8)
            numpy_imgs_1 = numpy_imgs_1 * mask
            numpy_imgs_2 = numpy_imgs_2 * mask

        ssim_score = []
        for img_1, img_2 in zip(numpy_imgs_1, numpy_imgs_2):
            ssim_score.append(compare_ssim(img_1, img_2, multichannel=True))

        return Variable(images_1.data.new(1).fill_(np.mean(ssim_score)))



###############################################################################
# flow-based warping
###############################################################################
def warp_acc_flow(x, flow, mode='bilinear', mask=None, mask_value=-1):
    '''
    warp an image/tensor according to given flow.
    Input:
        x: (bsz, c, h, w)
        flow: (bsz, c, h, w)
        mask: (bsz, 1, h, w). 1 for valid region and 0 for invalid region. invalid region will be fill with "mask_value" in the output images.
    Output:
        y: (bsz, c, h, w)
    '''
    bsz, c, h, w = x.size()
    # mesh grid
    xx = x.new_tensor(range(w)).view(1,-1).repeat(h,1)
    yy = x.new_tensor(range(h)).view(-1,1).repeat(1,w)
    xx = xx.view(1,1,h,w).repeat(bsz,1,1,1)
    yy = yy.view(1,1,h,w).repeat(bsz,1,1,1)
    grid = torch.cat((xx,yy), dim=1).float()
    grid = grid + flow
    # scale to [-1, 1]
    grid[:,0,:,:] = 2.0*grid[:,0,:,:]/max(w-1,1) - 1.0
    grid[:,1,:,:] = 2.0*grid[:,1,:,:]/max(h-1,1) - 1.0

    grid = grid.permute(0,2,3,1)
    output = F.grid_sample(x, grid, mode=mode, padding_mode='zeros')
    # mask = F.grid_sample(x.new_ones(x.size()), grid)
    # mask = torch.where(mask<0.9999, mask.new_zeros(1), mask.new_ones(1))
    # return output * mask
    if mask is not None:
        output = torch.where(mask>0.5, output, output.new_ones(1).mul_(mask_value))
    return output

###############################################################################
# layers
###############################################################################
class Identity(nn.Module):
    def __init__(self, dim=None):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
