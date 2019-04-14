from __future__ import division, print_function

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import util.io as io
import argparse
from PIL import Image
import os
import sys
import tqdm
################################
# options
################################
parser = argparse.ArgumentParser()
# general
parser.add_argument('--train', action='store_true', help='training network')
parser.add_argument('--id', type=str, default='default', help='id of fashion inception network')
parser.add_argument('--n_class', type=int, default=23, help='fashion item class number')
parser.add_argument('--rescale_size', type=int, default=331)
parser.add_argument('--crop_size', type=int, default=299)
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--debug', action='store_true')
# train
parser.add_argument('--fn_split', type=str, default='datasets/DF_Pose/Label/image_split_dfm_new.json')
parser.add_argument('--fn_label', type=str, default='datasets/DF_Pose/Label/class_label.json')
parser.add_argument('--img_dir', type=str, default='datasets/DF_Pose/Img/img/')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_decay', type=float, default=10)
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--display_freq', type=int, default=10)
parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
# test
parser.add_argument('--test_dir', type=str, default=None, help='path of generated images')
parser.add_argument('--n_test_split', type=int, default=10)
parser.add_argument('--ext', type=str, default='jpg', choices=['jpg', 'png'])
# parse input arguments
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
################################
# dataset
################################
class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, split='train'):
        assert split in {'train', 'val', 'test'}
        super(Dataset, self).__init__()
        self.opt = opt
        self.split = split
        if split in {'train', 'val'}:
            self.label = io.load_json(opt.fn_label)['label']
            self.id_list = io.load_json(opt.fn_split)['train' if split=='train' else 'test']
            self.image_dir = opt.img_dir
        else:
            self.label = None
            self.image_dir = opt.test_dir
            self.id_list = [fn[0:-4] for fn in os.listdir(self.image_dir)]
            
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(opt.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(opt.rescale_size),
                transforms.CenterCrop(opt.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
            ])
            
        if opt.debug:
            self.id_list = self.id_list[0:64]
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        sid = self.id_list[index]
        # read image
        with open(os.path.join(self.image_dir, sid+'.'+opt.ext), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)
        # label
        if self.split in {'train', 'val'}:
            l = self.label[sid]
        else:
            l = -1
        data = {'image': img, 'label':l}
        return data

################################
# functions
################################
def create_model(opt):
    model = torchvision.models.inception_v3(pretrained=True)
    model.aux_logits = False
    model.fc = torch.nn.Linear(2048, opt.n_class)
    return model


def train(opt):
    # model
    print('creating inception model')
    model = create_model(opt)
    model.cuda()
    # data
    print('creating data loaders')
    train_dset = Dataset(opt, 'train')
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dset = Dataset(opt, 'val')
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=False)
    # optimizer
    if opt.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    elif opt.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=opt.lr, alpha=0.9, eps=1.0, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=opt.lr_decay, gamma=opt.lr_gamma)
    # experiment dir and log
    exp_dir = 'checkpoints/Fashion_Inception/' + opt.id
    io.mkdir_if_missing(exp_dir)
    io.save_json(vars(opt), os.path.join('options.json'))
    log_file = open(os.path.join(exp_dir, 'train_log.txt'), 'w')
    def log(info):
        tqdm.tqdm.write(info)
        log_file.write(info+'\n')
        sys.stdout.flush()
        
    # training loop
    best_info = {
        'epoch': -1,
        'acc': 0,
    }
    for epoch in tqdm.trange(opt.n_epoch):
        scheduler.step()
        # train
        model.train()
        for idx, data in enumerate(tqdm.tqdm(train_loader, desc='training epoch %d'%epoch)):
            img = data['image'].cuda()
            label = data['label'].cuda().long()
            pred = model(img)
            loss = F.cross_entropy(pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if idx % opt.display_freq == 0:
                log('epoch: %d, iter: %d/%d, lr: %f, loss: %.3f'%(epoch, idx, len(train_loader), optim.param_groups[0]['lr'], loss.item()))
            
        # validate
        model.eval()
        mean_loss = 0
        mean_acc = 0
        for idx, data in enumerate(tqdm.tqdm(val_loader, desc='testing epoch %d'%epoch)):
            img = data['image'].cuda()
            label = data['label'].cuda().long()
            with torch.no_grad():
                pred = model(img)
            loss = F.cross_entropy(pred, label)
            acc = (pred.argmax(dim=1)==label).float().sum() / label.size(0)
            mean_loss += loss
            mean_acc += acc
        mean_loss = mean_loss / len(val_loader)
        mean_acc = mean_acc / len(val_loader)
        log('Epoch %d'%epoch)
        log('loss: %.6f'%mean_loss)
        log('acc: %.2f'%(mean_acc*100))
        if mean_acc > best_info['acc']:
            best_info['acc'] = mean_acc
            best_info['epoch'] = epoch
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best.pth'))
            log('save best epoch: %d' % epoch)


def compute_inception_score(opt):
    # load model
    print('loading model')
    exp_dir = 'checkpoints/Fashion_Inception/' + opt.id
    model = create_model(opt)
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best.pth')))
    model.cuda()
    model.eval()

    # create dataset
    print('creaing data loader')
    test_dset = Dataset(opt, 'test')
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=False)
    print('loaded %d images'%len(test_dset))
    # test loop
    prob_list = []
    for data in tqdm.tqdm(test_loader):
        img = data['image'].cuda()
        with torch.no_grad():
            pred = model(img)
        prob = F.softmax(pred, dim=1)
        prob_list.append(prob.cpu().numpy())
    prob = np.concatenate(prob_list, axis=0)
    # compute inception score
    scores = []
    for i in range(opt.n_test_split):
        part = prob[(i*prob.shape[0]//opt.n_test_split):((i+1)*prob.shape[0]//opt.n_test_split)]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores)



if __name__ == '__main__':
    if opt.train:
        raise NotImplementedError()
        # train(opt)
    else:
        score, std = compute_inception_score(opt)
        print(opt.test_dir)
        print('score: %f, std: %f'%(score, std))

