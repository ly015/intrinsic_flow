'''
Created by Yining LI. 30/07/2018
Script to infer 3d pose by HMR model.
'''

from __future__ import absolute_import, division, print_function
import sys
from absl import flags
import numpy as np
import skimage.io as io

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel
import json
import tqdm
import os
import cPickle


def create_image_list(num_sample):
    with open('/data2/ynli/datasets/DF_Pose/Label/image_split_dfm.json') as f:
        split = json.load(f)
    id_list = split['test'] + split['train']
    if num_sample > 0:
        id_list = id_list[:num_sample]
    with open('/data2/ynli/datasets/DF_Pose/Label/pose_label_dfm.pkl', 'rb') as f:
        pose_label = cPickle.load(f)

    image_list = []
    for sid in tqdm.tqdm(id_list, desc='creating image list'):
        pose = pose_label[sid]
        pose_fmt = []
        for x,y in pose:
            if x>0 and y>0:
                pose_fmt += [x,y,1]
            else:
                pose_fmt += [x,y,0]
        image_list.append({
            'id': sid,
            'path': '/data2/ynli/datasets/DF_Pose/Img/img/' + sid + '.jpg',
            'pose_keypoints': pose_fmt
            })

    return image_list

def create_image_list_old(num_sample):
    with open('/data2/ynli/datasets/DF_Pose/Label/pair_split_dfm.json') as f:
        split = json.load(f)
    pair_list = split['test'] + split['train']
    if num_sample > 0:
        pair_list = pair_list[:num_sample]
    id_list = sum([[p1, p2] for p1, p2 in pair_list], [])
    id_list = list(set(id_list))

    with open('/data2/ynli/datasets/DF_Pose/Label/pose_label_dfm.pkl', 'rb') as f:
        pose_label = cPickle.load(f)

    image_list = []
    for sid in tqdm.tqdm(id_list, desc='creating image list'):
        pose = pose_label[sid]
        pose_fmt = []
        for x,y in pose:
            if x>0 and y>0:
                pose_fmt += [x,y,1]
            else:
                pose_fmt += [x,y,0]
        image_list.append({
            'id': sid,
            'path': '/data2/ynli/datasets/DF_Pose/Img/img/' + sid + '.jpg',
            'pose_keypoints': pose_fmt
            })

    return image_list


def get_bbox(pose_keypoints):
    kp = np.array(pose_keypoints).reshape(-1,3)
    vis = kp[:,2] > 0
    vis_kp = kp[vis, :2]
    if vis_kp.size == 0:
        return None, None
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        center = scale = None
    else:
        center = (min_pt + max_pt) / 2.
        scale = 150. / person_height

    return scale, center

def preprocess_image(img, pose_keypoints, use_pose=True):

    if use_pose:
        scale, center = get_bbox(pose_keypoints)

    if (not use_pose) or (scale is None):
        scale = (float(config.img_size) / np.max(img.shape[:2]))
        center = np.round(np.array(img.shape[:2])/2).astype(np.int)
        center = center[::-1]

    crop, proc_param = img_util.scale_and_crop(img, scale, center, config.img_size)
    crop = 2*((crop / 255.) - 0.5)

    return crop, proc_param

def draw_2d_verts(img, verts2d, verts_z = None):
    if verts_z is None:
        verts_z = np.ones(verts2d.shape[0])
    else:
        verts_z = (verts_z - verts_z.min()) / (verts_z.max() - verts_z.min())

    def _color(z):
        return np.array([255*z, 255*(1-z), 0], dtype=np.uint8)
            
    img = img.copy()
    verts2d = verts2d.astype(np.int)
    for (x,y), z in zip(verts2d, verts_z):
        if 0<=x<img.shape[1] and 0<=y<img.shape[0]:
            img[y,x] = _color(z)
    return img

def visualize(img, proc_param, joints_crop, verts_crop, cam_crop):
    # render using opendr
    cam_render, verts_render, joints_render = vis_util.get_original(proc_param, verts_crop, cam_crop, joints_crop, img_size=img.shape[:2])
    skel_img = vis_util.draw_skeleton(img, joints_render)
    rend_img_overlay = renderer(verts_render, cam=cam_render, img=img, do_alpha=True)
    rend_img = renderer(verts_render, cam=cam_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(verts_render, 60, cam=cam_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(verts_render, -60, cam=cam_render, img_size=img.shape[:2])
    # draw verts projected on original image
    verts2d = project_3d_to_original2d_v2(cam_crop, verts_crop, proc_param)
    verts_z = verts_crop[:,2]
    verts2d_img = draw_2d_verts(np.zeros(img.shape, dtype=np.uint8), verts2d, verts_z)
    verts2d_img_overlay = draw_2d_verts(img, verts2d, verts_z)
    verts2d_rend_overlay = draw_2d_verts(rend_img, verts2d, verts_z)

    row1 = np.hstack([img, skel_img, rend_img_overlay[...,:3]])
    row2 = np.hstack([rend_img, rend_img_vp1[...,:3], rend_img_vp2[...,:3]])
    row3 = np.hstack([verts2d_img, verts2d_img_overlay, verts2d_rend_overlay])
    vis = np.vstack([row1, row2, row3])
    return vis


def project_3d_to_original2d(cam, verts, proc_param):
    '''
    project 3d points to original 2d coordinate space.
    Input:
        cam: (1, 3) camera parameters (f, cx, cy) output by model.
        verts: 3d verts output by model.
        proc_param: preprocessing parameters. this is for converting points from crop (model input) to original image.
    Output:
    '''

    crop_size = proc_param['img_size']
    undo_scale = 1./proc_param['scale']
    start_pt = proc_param['start_pt'].reshape(1,2)

    cam = cam.reshape(3)
    verts = verts.reshape(-1, 3)
    verts2d_crop = (verts[:,0:2] + cam[1:3]) * cam[0]
    verts2d_crop = (1. + verts2d_crop) * crop_size * 0.5
    verts2d = (verts2d_crop + start_pt - int(crop_size/2)) * undo_scale
    return verts2d

def project_3d_to_original2d_v2(cam, verts, proc_param):
    '''
    project 3d points to original 2d coordinate space.
    NOTE: this version is coincident with render code
    Input:
        cam: (1, 3) camera parameters (f, cx, cy) output by model.
        verts: (6890, 3) 3d verts output by model.
        proc_param: preprocessing parameters. this is for converting points from crop (model input) to original image.
    Output:
        verts2d: (6890, 2) projected 2d verts
    '''
    undo_scale = 1./proc_param['scale']
    start_pt = proc_param['start_pt'].reshape(1,2)
    img_size = proc_param['img_size']
    flength = 500.
    cam = cam.reshape(3)
    verts = verts.reshape(-1, 3)
    xy_3d = verts[:,:2] #(6890,2)
    z_3d = verts[:,2:3] #(6890, 1)
    
    verts2d = (xy_3d + cam[1:3])* flength / (z_3d + flength/(cam[0]*0.5*img_size))
    verts2d = (verts2d + start_pt) * undo_scale
    return verts2d

def main(num_chunk=-1, i_chunk=-1):
    # config
    num_sample = -1
    use_pose = True
    output_vis = True
    # output_dir = '/data2/ynli/Fashion/fashionHD/temp/3d_hmr/hmr_df_openpose/'
    # output_dir = '/data2/ynli/Fashion/fashionHD/temp/3d_hmr/hmr_df/'
    # output_dir = '/data2/ynli/Fashion/fashionHD/datasets/DF_Pose/3d/hmr_dfm/'
    #output_dir = '/data2/ynli/Fashion/fashionHD/datasets/DF_Pose/3d/hmr_dfm_v2/'
    output_dir = '/data2/ynli/Fashion/fashionHD/datasets/DF_Pose/3d/hmr_dfm_v3/'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir+'pred/'):
        os.mkdir(output_dir+'pred/')
    if not os.path.isdir(output_dir+'vis/'):
        os.mkdir(output_dir+'vis/')
    
    image_list = create_image_list(num_sample)
    if num_chunk > 0 and i_chunk >= 0:
        image_list = image_list[i_chunk::num_chunk]

    # load model
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    for img_data in tqdm.tqdm(image_list):
        sid = img_data['id']
        # skip samples whose 3d parameters has already been calculated
        # if os.path.isfile(output_dir+'pred/'+sid+'.pkl'):
        #     continue

        img = io.imread(img_data['path'])
        pose_keypoints = img_data['pose_keypoints']

        input_img, proc_param = preprocess_image(img, pose_keypoints, use_pose)
        input_img = np.expand_dims(input_img, 0)
        joints_crop, verts_crop, cam_crop, joints3d_crop, theta = model.predict(input_img, get_theta=True)
        joints_crop, verts_crop, cam_crop = joints_crop[0], verts_crop[0], cam_crop[0]
        
        # compute 2d vertex coordinates and depth
        verts2d = project_3d_to_original2d_v2(cam_crop, verts_crop, proc_param)
        verts_z = verts_crop[:,2]
        
        # compute face visibility map by opendr
        cam_render, verts_render, joints_render = vis_util.get_original(proc_param, verts_crop, cam_crop, joints_crop, img_size=img.shape[:2])
        _, visibility = renderer(verts_render, cam=cam_render, img_size=img.shape[:2], return_visibility_image=True)

        pred = {
            'id': sid,
            'theta': theta,
            'proc_param': proc_param,
            'verts2d': verts2d,
            'verts_z': verts_z,
            'visibility': visibility, # a map with same size of img, each pixel is its corresponding SMPL face index (or 4294967295 if it's corresponding to no face)
            'joints': joints_render,
            # 'cam_render': cam_render,
            # 'verts_render': verts_render,
            # 'verts': verts_crop,
        }

        with open(output_dir+'pred/'+sid+'.pkl', 'wb') as f:
            cPickle.dump(pred, f, cPickle.HIGHEST_PROTOCOL)
        if output_vis:
            vis = visualize(img, proc_param, joints_crop, verts_crop, cam_crop)
            io.imsave(output_dir+'vis/'+sid+'.jpg', vis)

if __name__ == '__main__':
    gpu_ids = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    import tensorflow as tf

    num_chunk = 16
    i_chunk = int(gpu_ids)

    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
    main(num_chunk=num_chunk, i_chunk=i_chunk)
