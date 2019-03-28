from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np

# basic image processing interface based on openCV
def imread(filename, mode = 'rgb'):
    if mode == 'grayscale':
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(filename)

def imwrite(im, filename):
    return cv2.imwrite(filename, im)
    

def imshow_cv(im, window_name = 'default'):
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, im)
    # key = cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    # if key ==27:
    #     raise Exception('Esc pressed!')
    # return

def imshow(im):
    '''
    im: numpy.ndarray which stores a image in BGR format.
    '''
    if im.ndim == 3:
        # convert BGR to RGB
        im = im[:,:,[2,1,0]]
        plt.imshow(im)
    else:
        plt.imshow(im, cmap = 'gray')
    plt.show()

def imshowt(im_t):
    '''
    show tensor image
    '''
    im = im_t.cpu().detach().numpy().transpose(1,2,0)
    if im.shape[2] == 3:
        im = im[...,[2,1,0]]
    else:
        im = im[...,0]
    im = (im*127.5+127.5).clip(0,255).astype(np.uint8)
    imshow(im)


def resize(im,  new_shape_t, interpolation = cv2.INTER_LINEAR):
    ''' im_out = resize(im_in, new_shape = (w, h)) '''
    new_shape = list(new_shape_t)
    assert(new_shape[0] > 0 or new_shape[1] > 0)
    if new_shape[0] < 0:
        new_shape[0] = int(new_shape[1] / im.shape[0] * im.shape[1])
    elif new_shape[1] < 0:
        new_shape[1] = int(new_shape[0] / im.shape[1] * im.shape[0])

    return cv2.resize(im, tuple(new_shape), interpolation = interpolation)
    
def crop(im, bbox):
    ''' im_patch = crop(im, bbox = (x1, y1, x2, y2))'''
    x1, y1, x2, y2 = bbox
    h, w = im.shape[0:2]
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(max(min(x2, w-1), x1))
    y2 = int(max(min(y2, h-1), y1))

    return im[y1:(y2+1), x1:(x2+1)].copy()

def pad_square(im, size, padding_value = 0, mode = 'center'):
    '''
    size (tuple or int):
        - (w, h)
        - sz
    '''
    assert mode in {'center', 'lefttop'}

    h, w = im.shape[0:2]
    if isinstance(size, tuple):
        wt, ht = size
    else:
        wt = ht = size

    assert w <= wt and h <= ht

    if im.ndim == 3:
        im_pad = np.ones((ht, wt, im.shape[2]), dtype = im.dtype) * padding_value
    else:
        im_pad = np.ones((ht, wt), dtype = im.dtype) * padding_value

    if mode == 'center':
        x1 = int((wt - w)/2)
        y1 = int((ht - h)/2)
        x2 = x1 + w
        y2 = y1 + h
        im_pad[y1:y2, x1:x2] = im
    elif mode == 'lefttop':
        im_pad[0:h, 0:w] = im

    return im_pad



def upsample(im, size, fill = True):
    '''
    up sample the image to have size as its long-side length
    if fill == ture, the up sampled image will be padded to size x size with 0. 
    '''
    new_shape = list(im.shape[0:2])
    if new_shape[0]>new_shape[1]:
        new_shape[0:2] = [size, new_shape[1] / new_shape[0] * size]
    else:
        new_shape[0:2] = [new_shape[0]/new_shape[1] * size, size]

    new_shape = np.array(new_shape, dtype = np.int)
    if fill == True:
        im_out = np.zeros((size, size, 3), dtype = im.dtype)
        im_out[0:new_shape[0], 0:new_shape[1], :] = resize(im, new_shape[::-1])
    else:
        im_out = resize(im, new_shape[::-1])

    return im_out

color_map = {'r':(255,0,0), 'g':(0,255,0), 'b':(0,0,255),\
             'y':(255,255,0), 'p':(255,0,255), 'c':(0,255,255),\
             'rg':(255,255,0), 'rb':(255,0,255), 'gb':(0,255,255),\
             'w':(255,255,255), 'k':(0,0,0), 'wk':(100,100,100)}
def _color_transfer(color):
    if isinstance(color, tuple):
        return color
    else:
        assert(isinstance(color, str))
        if color in color_map:
            return color_map[color]
        else:
            return (0,0,0)

def draw_rectangle(im, position, color, thickness = 1):
    ''' draw_rectangle(im, position = (x1, y1, x2, y2), color = (r, g, b))
        (x1, y1), (x2, y2) are left-top and right-bottom point'''
    color = _color_transfer(color)
    position = np.array(position, dtype=int)
    cv2.rectangle(im, tuple(position[0:2]), tuple(position[2:4]), tuple(color[::-1]), thickness)
    return im

def draw_text(im, str_text, position, color, font_size = 1):
    ''' draw_text(im, str_text, position = (x, y), color = (r, g, b), font_size = 1.0)
        (x, y) is bottom-left point '''
    color = _color_transfer(color)
    position = tuple(np.array(position, dtype = np.int))
    cv2.putText(im, str_text.encode('ascii', 'replace'), position, cv2.FONT_HERSHEY_SIMPLEX, font_size, tuple(color[::-1]))
    return im


def add_tag(im, tag_list, color_list = None):
    if color_list == None or len(color_list) != len(tag_list):
        color_list = [(0,0,0) for i in xrange(len(tag_list))]
    else:
        color_list = [_color_transfer(c) for c in color_list]

    height, width = im.shape[0], im.shape[1]
    # new_height = max(height, len(tag_list) * 20)
    # new_width = width + max(200, max([len(t) for t in tag_list]) * 10 + 10)
    # new_im = np.zeros((new_height, new_width, 3), dtype = im.dtype)
    # new_im[0:height, 0:width, :] = im
    # for i, tag in enumerate(tag_list):
    #   draw_text(new_im, tag, (width + 10, i * 20 + 20), (0,0,0), font_size = 1)
    new_height = height + 20 * len(tag_list) + 10
    new_width = max(width, max([len(t) for t in tag_list])* 10 + 10)
    new_im = np.ones((new_height, new_width, 3), dtype = im.dtype) * 255
    new_im[0:height, 0:width, :] = im
    for i, tag in enumerate(tag_list):
        draw_text(new_im, tag, (10, height + (i+1) * 20), color = color_list[i], font_size = 0.5)
    return new_im

# all boxes are [xmin, ymin, xmax, ymax] format, 0-indexed, including xmax and ymax
def compute_iou(boxes, target):
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    if isinstance(target, list):
        target = np.array(target)

    assert(target.ndim == 1 and boxes.ndim == 2)
    A_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    A_target = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)
    assert(np.all(A_boxes >= 0))
    assert(np.all(A_target >= 0))
    I_x1 = np.maximum(boxes[:, 0], target[0])
    I_y1 = np.maximum(boxes[:, 1], target[1])
    I_x2 = np.minimum(boxes[:, 2], target[2])
    I_y2 = np.minimum(boxes[:, 3], target[3])
    A_I = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)
    IoUs = A_I / (A_boxes + A_target - A_I)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs

def stitch(image_list, axis = 0):
    '''
    image_list: a list of images (3-d numpy.ndarray)
    axis: 0 for horizontal and 1 for vertical
    '''
    if axis == 0:
        w = sum([im.shape[1] for im in image_list])
        h = max([im.shape[0] for im in image_list])
        new_im = np.zeros((h, w, 3), dtype = np.float32)
        n = 0
        for im in image_list:
            new_im[0:im.shape[0],n:(n+im.shape[1]),:] = im
            n += im.shape[1]
    else:
        w = max([im.shape[1] for im in image_list])
        h = sum([im.shape[0] for im in image_list])
        new_im = np.zeros((h, w, 3), dtype = np.float32)
        n = 0
        for im in image_list:
            new_im[n:(n+im.shape[0]),0:im.shape[1],:] = im
            n += im.shape[0]
    return new_im

def combine_bbox(bbox_list):
    bbox_mat = np.array(bbox_list, dtype = np.float32)
    bbox = np.min(bbox_mat, axis = 0)
    bbox[2:4] = np.max(bbox_mat[:,2:4], axis = 0)
    return bbox.tolist()


def align_image(im, p_src, p_tar, sz_tar, flags = cv2.INTER_CUBIC):
    '''
    align image by affine transform
    
    input:
        im: input image
        p_src: list of source key points
        p_tar: list of target key points
        sz_tar: target image size
    '''

    p_src = np.array(p_src, dtype = np.float32)
    p_tar = np.array(p_tar, dtype = np.float32)
    
    assert p_src.shape == p_tar.shape
    num_p = p_src.shape[0]

    X = np.zeros((num_p * 2, 4), dtype = np.float32)
    U = np.zeros((num_p * 2, 1), dtype = np.float32)

    X[0:num_p, 0:2] = p_src
    X[0:num_p, 2] = 1
    X[num_p::, 0] = p_src[:,1]
    X[num_p::, 1] = -p_src[:,0]
    X[num_p::, 3] = 1

    U[0:num_p, 0] = p_tar[:,0]
    U[num_p::, 0] = p_tar[:,1]

    M = np.linalg.pinv(X).dot(U).flatten()
    trans_mat = np.array([[M[0], M[1], M[2]], [-M[1], M[0], M[3]]], dtype = np.float32)

    im_out = cv2.warpAffine(im, trans_mat, dsize = sz_tar, flags = flags, borderMode = cv2.BORDER_REPLICATE)

    return im_out, trans_mat


def transform_coordinate(p_src, t_mat):
    '''
    Input:
        p_src: Nx2, 2-D coordinate array of source points
        t_mat: 2x2 or 2x3, transform matrix

    Output:
        p_tar: Nx2, 2-D coordinate array of target points
    '''

    p_src = p_src[:, np.newaxis, :]
    p_tar = cv2.transform(p_src, t_mat).squeeze()

    return p_tar
