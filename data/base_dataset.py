from __future__ import division, print_function
import torch.utils.data as data
import numpy as np
from PIL import Image
import cv2

#####################################
# BaseDataset Class
#####################################

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

#####################################
# Image Transform Modules
#####################################
def kp_to_map(img_sz, kps, mode='gaussian', radius=5):
    '''
    Keypoint cordinates to heatmap map.
    Input:
        img_size (w,h): size of heatmap
        kps (N,2): (x,y) cordinates of N keypoints
        mode: 'gaussian' or 'binary'
        radius: radius of each keypoints in heatmap
    Output:
        m (h,w,N): encoded heatmap
    '''
    w, h = img_sz
    x_grid, y_grid = np.meshgrid(range(w), range(h), indexing = 'xy')
    m = []
    for x, y in kps:
        if x == -1 or y == -1:
            m.append(np.zeros((h, w)).astype(np.float32))
        else:
            if mode == 'gaussian':
                m.append(np.exp(-((x_grid - x)**2 + (y_grid - y)**2)/(radius**2)).astype(np.float32))
            elif mode == 'binary':
                m.append(((x_grid-x)**2 + (y_grid-y)**2 <= radius**2).astype(np.float32))
            else:
                raise NotImplementedError()
    m = np.stack(m, axis=2)
    return m


def seg_label_to_map(seg_label, nc = 7, bin_size=1):
    '''
    Input:
        seg_label: (H,W), 2D segmentation class label
        nc: number of classes
        bin_size: filter isolate pixels which is likely to be noise
    Output:
        seg_map: (H,W,nc)
    '''
    seg_map = [(seg_label == i) for i in range(nc)]
    seg_map = np.concatenate(seg_map, axis=2).astype(np.float32)
    if bin_size > 1:
        h, w = seg_map.shape[0:2]
        dh, dw = h//bin_size, w//bin_size
        seg_map = cv2.resize(seg_map, dsize=(dw,dh), interpolation=cv2.INTER_LINEAR)
        seg_map = cv2.resize(seg_map, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    return seg_map
###############################################################################
