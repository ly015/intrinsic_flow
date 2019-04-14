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
from models.modules import MeanAP
from collections import OrderedDict


################################
# options
################################
parser = argparse.ArgumentParser()
# general
parser.add_argument('--train', action='store_true', help='training network')
parser.add_argument('--id', type=str, default='default', help='id of fashion attribute model')
#parser.add_argument('--n_attr', type=int, default=463, help='fashion attribute entries number')
parser.add_argument('--n_attr', type=int, default=347, help='fashion attribute entries number')
parser.add_argument('--rescale_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--debug', action='store_true')
# train
parser.add_argument('--fn_split', type=str, default='datasets/DF_Pose/Label/image_split_dfm_new.json')
parser.add_argument('--fn_label', type=str, default='datasets/DF_Pose/Label/attr_label.pkl')
parser.add_argument('--img_dir', type=str, default='datasets/DF_Pose/Img/img/')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
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
        self.label = io.load_data(opt.fn_label)['label']
        if split in {'train', 'val'}:
            self.id_list = io.load_json(opt.fn_split)['train' if split=='train' else 'test']
            self.image_dir = opt.img_dir
        else:
            self.image_dir = opt.test_dir
            self.id_list = [fn[0:-4] for fn in os.listdir(self.image_dir)]
            
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(opt.rescale_size),
                transforms.RandomCrop(opt.crop_size),
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
            # sid = sid1 + '_' + sid2
            sid2 = '_'.join(sid.split('_')[2:])            
            l = self.label[sid2]
        data = {'image': img, 'label':l}
        return data

################################
# functions
################################
def create_model(opt):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, opt.n_attr)
    return model


def train(opt):
    # model
    print('creating network model')
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
    exp_dir = 'checkpoints/Fashion_Attr/' + opt.id
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
        'mean_ap': 0,
    }
    for epoch in tqdm.trange(opt.n_epoch):
        scheduler.step()
        # train
        model.train()
        for idx, data in enumerate(tqdm.tqdm(train_loader, desc='training epoch %d'%epoch)):
            img = data['image'].cuda()
            label = data['label'].cuda().float()
            pred = model(img)
            loss = F.binary_cross_entropy_with_logits(pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if idx % opt.display_freq == 0:
                log('epoch: %d, iter: %d/%d, lr: %f, loss: %.3f'%(epoch, idx, len(train_loader), optim.param_groups[0]['lr'], loss.item()))
            
        # validate
        model.eval()
        crit = MeanAP()
        mean_loss = 0
        for idx, data in enumerate(tqdm.tqdm(val_loader, desc='testing epoch %d'%epoch)):
            img = data['image'].cuda()
            label = data['label'].cuda().float()
            with torch.no_grad():
                pred = model(img)
            mean_loss += F.binary_cross_entropy_with_logits(pred, label)
            crit.add(F.sigmoid(pred), label)
        
        mean_ap, _ = crit.compute_mean_ap()
        rec3 = crit.compute_recall(3)
        rec5 = crit.compute_recall(5)
        rec10 = crit.compute_recall(10)
        rec20 = crit.compute_recall(20)

        log('Epoch %d'%epoch)
        log('loss: %.6f'%(mean_loss/len(val_loader)))
        log('mean_ap: %f'%mean_ap)
        log('recall3: %f, recall5: %f, recall10: %f, recall20: %f'%(rec3, rec5, rec10, rec20))

        if mean_ap > best_info['mean_ap']:
            best_info['mean_ap'] = mean_ap
            best_info['epoch'] = epoch
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best.pth'))
            log('save best epoch: %d' % epoch)


def compute_attribute_score(opt):
    # load model
    print('loading model')
    exp_dir = 'checkpoints/Fashion_Attr/' + opt.id
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
    crit = MeanAP()
    for data in tqdm.tqdm(test_loader):
        img = data['image'].cuda()
        label = data['label'].cuda().float()
        with torch.no_grad():
            pred = model(img)
        crit.add(F.sigmoid(pred), label)
    mean_ap, _ = crit.compute_mean_ap()
    
    rec3 = crit.compute_recall(3)
    rec5 = crit.compute_recall(5)
    rec10 = crit.compute_recall(10)
    rec20 = crit.compute_recall(20)
    rst = OrderedDict([
        ('mean_ap', mean_ap),
        ('rec-3', rec3),
        ('rec-5', rec5),
        ('rec-10', rec10),
        ('rec-20', rec20),
    ])
    
    return rst

def create_attribute_label():
    img_split = io.load_json('datasets/DF_Pose/Label/image_split_dfm_new.json')
    id_list = img_split['train'] + img_split['test']
    attr_entry = io.load_str_list('datasets/DeepFashion/In-shop/Anno/list_attr_cloth.txt')[2:]
    attr_anno = io.load_str_list('datasets/DeepFashion/In-shop/Anno/list_attr_items.txt')
    attr_anno = attr_anno[2:]
    attr_anno = [l.replace('-1', '0').split() for l in attr_anno]
    attr_anno = {l[0]:np.array(l[1:], dtype=np.int) for l in attr_anno}

    label = {}
    for sid in id_list:
        s = sid.index('id')+2
        e = s+8
        sid_ori = 'id_' + sid[s:e]
        label[sid] = attr_anno[sid_ori]

    # remove attribute entries with no positive sample
    label_mat = np.array(label.values())
    valid_idx = label_mat.sum(axis=0) > 0
    print('%d valid attribute entries'%(valid_idx.sum()))
    label = {k:v[valid_idx] for k,v in label.iteritems()}
    attr_entry = [e for i,e in enumerate(attr_entry) if valid_idx[i]]
    attr_label = {'label': label, 'entry': attr_entry}
    
    io.save_data(attr_label, 'datasets/DF_Pose/Label/attr_label.pkl')

if __name__ == '__main__':
    #create_attribute_label()
    #exit(1)
    if opt.train:
        train(opt)
    else:
        rst = compute_attribute_score(opt)
        for k,v in rst.iteritems():
            print('%s: %f'%(k, v))
