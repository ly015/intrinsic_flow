from __future__ import division, print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.optim import lr_scheduler
import functools

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
        if opt.continue_train:
            try:
                last_epoch = int(opt.which_epoch) - 1
            except:
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
# layers
###############################################################################
class Identity(nn.Module):
    def __init__(self, dim=None):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
