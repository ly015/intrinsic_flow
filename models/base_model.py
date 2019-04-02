from __future__ import division, print_function

import torch
import os

class BaseModel(object):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join('checkpoints', opt.id)

        self.input = {}
        self.output = {}

    def set_input(self, data):
        self.input = data

    def forward(self):
        pass

    # used in test time, no backprob
    def test(self):
        pass

    def optimize_parameters(self):
        pass

    
    def get_current_visuals(self):
        return self.input

    
    def get_current_errors(self):
        return {}

    def train(self):
        pass

    def eval(self):
        pass

    def save(self, label):
        pass

    # helper loading function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)

        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    def load_network(self, network, network_label, epoch_label, model_id = None, forced = True):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if model_id is None:
            # for continue training
            save_dir = self.save_dir
        else:
            # for initialize weight
            save_dir = os.path.join('checkpoints', model_id)
        save_path = os.path.join(save_dir, save_filename)
        if (not forced) and (not os.path.isfile(save_path)):
            print('[%s] FAIL to load [%s] parameters from %s' % (self.name(), network_label, save_path))
        else:
            network.load_state_dict(torch.load(save_path))
            print('[%s] load [%s] parameters from %s' % (self.name(), network_label, save_path))

    def save_optim(self, optim, optim_label, epoch_label):
        save_filename = '%s_optim_%s.pth'%(epoch_label, optim_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(optim.state_dict(), save_path)
        
    def load_optim(self, optim, optim_label, epoch_label):
        save_filename = '%s_optim_%s.pth'%(epoch_label, optim_label)
        save_path = os.path.join(self.save_dir, save_filename)
        optim.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)
