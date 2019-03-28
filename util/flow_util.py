'''
Derived from flownet2.0
'''
import torch
import numpy as np
import cv2


def readFlow(fn):
    """
    Derived from flownet2.0    
    """
    f = open(fn, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def writeFlow(fn, flow):
    """
    Derived from flownet2.0    
    """
    f = open(fn, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()


def write_corr(fn, corr, mask):
    '''
    Save correspondence map (float data) and mask (uint data) to one file.
    Input:
        fn: fine name
        corr: (H, W, 2), float32
        mask: (H, W), uint8
    '''
    assert corr.shape[:2]==mask.shape
    with open(fn, 'wb') as f:
        np.array([corr.shape[1], corr.shape[0]], dtype=np.int32).tofile(f)
        corr.astype(np.float32).tofile(f)
        mask.astype(np.uint8).tofile(f)

def read_corr(fn):
    '''
    Recover correspondence map and mask saved by "write_corr"
    '''
    with open(fn, 'rb') as f:
        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        corr = np.fromfile(f, np.float32, width*height*2).reshape((height, width, 2))
        mask = np.fromfile(f, np.uint8, width*height).reshape((height, width))
    return corr, mask

def visualize_corr(img_1, img_2, corr_1to2, mask_1=None, grid_step=5):
    '''
    Input:
        img_1: (h1, w1, 3)
        img_2: (h2, w2, 3)
        corr_1to2: (h1, w1, 2)
        grid_step: scalar
    Output:
        img_out: (max(h1, w1), w1+w2, 3)
    '''

    h1, w1 = img_1.shape[0:2]
    h2, w2 = img_2.shape[0:2]
    img_out = np.zeros((max(h1, h2), w1+w2, 3), dtype=img_1.dtype)
    img_out[:h1,:w1,:] = img_1
    img_out[:h2,w1:(w1+w2),:] = img_2
    
    mask = ((corr_1to2[...,0]>1) & (corr_1to2[...,0]<w2) & (corr_1to2[...,1]>1) & (corr_1to2[...,1]<h2)).astype(np.uint8)
    if mask_1 is not None:
        mask = mask * ((mask_1==0)|(mask_1==1)).astype(np.uint8)
    
    pt_y1, pt_x1 = np.where(mask==1)
    if grid_step > 1:
        pt_v = (pt_x1%grid_step==0) & (pt_y1%grid_step==0)
        pt_x1 = pt_x1[pt_v]
        pt_y1 = pt_y1[pt_v]

    pt_x2 = corr_1to2[pt_y1,pt_x1,0] + w1
    pt_y2 = corr_1to2[pt_y1,pt_x1,1]
    pt_color = points2color(np.stack([pt_x1, pt_y1], axis=1))

    for x1, y1, x2, y2, c in zip(pt_x1, pt_y1, pt_x2, pt_y2, pt_color):
        c = c.tolist()
        cv2.arrowedLine(img_out, (x1, y1), (x2, y2), c, line_type=cv2.LINE_AA, tipLength=0.02)
    
    return img_out


def points2color(points, method='Lab'):
    '''
    points: (N, 2) point coordinates
    method: {'Lab'}
    '''
    if method == 'Lab':
        range_x = points[:,0].max() - points[:,0].min()
        range_y = points[:,1].max() - points[:,1].min()
        L = np.ones(points.shape[0]) * 255
        A = points[:,0]*255.0/(range_x+0.1)
        B = points[:,1]*255.0/(range_y+0.1)
        C = np.stack([L,A,B], axis=1).astype(np.uint8)
        C = cv2.cvtColor(C.reshape(1,-1,3), cv2.COLOR_LAB2BGR).reshape(-1,3)
        return C
    else:
        raise NotImplementedError()

def warp_image(img, flow):
    h, w = flow.shape[:2]
    m = flow.astype(np.float32)
    m[:,:,0] += np.arange(w)
    m[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, m, None, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
    return res


def flow_to_rgb(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3))
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def corr_to_flow(corr, vis=None, order='NCHW'):
    '''
    order should be one of {'NCHW', 'HWC'}
    '''
    if order == 'NCHW':
        if isinstance(corr, torch.Tensor):
            flow = corr.clone()
            flow[:,0,:,:] -= torch.arange(flow.shape[3], dtype=flow.dtype, device=flow.device) # x-axis
            flow[:,1,:,:] -= torch.arange(flow.shape[2], dtype=flow.dtype, device=flow.device).view(-1,1) #  y-axis
        elif isinstance(corr, np.ndarray):
            flow = corr.copy()
            flow[:,0,:,:] -= np.arange(flow.shape[3])
            flow[:,1,:,:] -= np.arange(flow.shape[2]).reshape(-1,1)
    elif order == 'HWC':
        if isinstance(corr, torch.Tensor):
            flow = corr.clone()
            flow[:,:,0] -= torch.arange(flow.shape[1], dtype=flow.dtype, device=flow.device)
            flow[:,:,1] -= torch.arange(flow.shape[0], dtype=flow.dtype, device=flow.device).view(-1,1)
        elif isinstance(corr, np.ndarray):
            flow = corr.copy()
            flow[:,:,0] -= np.arange(flow.shape[1]).reshape(-1,)
            flow[:,:,1] -= np.arange(flow.shape[0]).reshape(-1,1)
    if vis is not None:
        if isinstance(vis, torch.Tensor):
            vis = (vis<2).float()
        elif isinstance(vis, np.ndarray):
            vis = (vis<2).astype(np.float32)
        flow *= vis
    return flow

