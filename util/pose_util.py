import numpy as np
from skimage.draw import circle, line_aa, polygon
from skimage.morphology import dilation, erosion, square

joint2idx = {
    'nose': 0,
    'neck': 1,
    'rshoulder':2,
    'relbow': 3,
    'rwrist': 4,
    'lshoulder': 5,
    'lelbow': 6,
    'lwrist': 7,
    'rhip': 8,
    'rknee': 9,
    'rankle': 10,
    'lhip': 11,
    'lknee': 12,
    'lankle': 13,
    'reye': 14,
    'leye': 15,
    'rear': 16,
    'lear': 17,
}

def get_joint_coord(label, joint_list):
    indices = [joint2idx[j] for j in joint_list]
    if isinstance(label, list):
        label = np.float32(label)
    return label[indices, :]



##############################################################################################
# Derived from Deformable GAN (https://github.com/AliaksandrSiarohin/pose-gan)
##############################################################################################
LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,1], [5,1]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']
MISSING_VALUE = -1

def map_to_coords(pose_map, threshold=0.1):
    '''
    Input:
        pose_map: (h, w, channel)
    Output:
        coord:
    '''
    all_peaks = [[] for i in range(18)]
    pose_map = pose_map[...,:18]

    y,x,z = np.where(np.logical_and(pose_map==pose_map.max(axis=(0,1)), pose_map>threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(18):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)

def draw_pose_from_coords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True
    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask

def draw_pose_from_map(pose_map, threshold=0.1, radius=2, draw_joints=True):
    img_size = pose_map.shape[0:2]
    coords = map_to_coords(pose_map, threshold)
    return draw_pose_from_coords(coords, img_size, radius, draw_joints)


def relative_scale_from_pose(pose, pose_std, skeleton_def=None):
    if skeleton_def is None:
        skeleton_def = [(2,3), (5,6), (2,8), (5,11)]
    scale = []
    for t1, t2 in skeleton_def:
        if np.all([pose[t1]>=0, pose[t2]>=0, pose_std[t1]>=0, pose_std[t2]>=0]):
            s = np.linalg.norm(pose[t1]-pose[t2]) / np.linalg.norm(pose_std[t1]-pose_std[t2])
            scale.append(s)
    if scale:
        return np.mean(scale)
    else:
        return -1

def relative_vertical_offset_from_pose(pose, pose_std, keypoint_def=None):
    if keypoint_def is None:
        keypoint_def = [0,1,2,5,8,11]

    valid = (pose[keypoint_def, 1] >=0) & (pose_std[keypoint_def, 1]>=0)
    if valid.any():
        offset = pose[[keypoint_def],1] - pose_std[[keypoint_def], 1]
        return np.mean(offset[valid])
    else:
        return None

def get_pose_mask(pose, img_size, point_radius=4):
    mask = np.zeros(shape=img_size, dtype=bool)
    limbs = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],\
            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],\
            [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    limbs = np.array(limbs) - 1
    for f,t in limbs:
        from_missing = pose[f][0] < 0 or pose[f][1] < 0
        to_missing = pose[t][0] < 0 or pose[t][1] < 0
        if from_missing or to_missing:
            continue
        norm_vec = pose[f] - pose[t]
        norm_vec = np.array([-norm_vec[1], norm_vec[0]])
        norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec+1e-8)
        
        vetexes = np.array([
            pose[f] + norm_vec,
            pose[f] - norm_vec,
            pose[t] - norm_vec,
            pose[t] + norm_vec])

        yy, xx = polygon(vetexes[:,0], vetexes[:,1], shape=img_size)
        mask[yy, xx] = True

    for i, joint in enumerate(pose):
        if pose[i][0] < 0 or pose[i][1] < 0:
            continue
        yy, xx = circle(joint[0], joint[1], radius=point_radius, shape=img_size)
        mask[yy, xx] = True

    mask = dilation(mask, square(5))
    mask = erosion(mask, square(5))

    return mask

def get_pose_mask_batch(pose, img_size, point_radius=4):
    '''
    Input:
        pose (tensor): (N,18,2) key points
        img_size (tuple): (h, w)
        point_radius (int): width of skeleton mask
    Output:
        mask (tensor): (N, 1, h, w)
    '''
    mask = []
    pose_np = pose.cpu().numpy()
    for p in pose_np:
        m = get_pose_mask(p, img_size, point_radius)
        mask.append(m)
    mask = np.expand_dims(np.stack(mask), axis=1)
    mask = mask.astype(np.float32)
    return pose.new(mask)
