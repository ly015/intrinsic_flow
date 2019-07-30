from __future__ import division, print_function
import sys
sys.path.append('./')
import util.io as io
import numpy as np
import tqdm
import imageio
import cv2

def divide_vert_into_bodypart():
    '''
    Devide 6890 verts of a SMPL model into 24 parts, each part corresponding to a joint. A vert will be assigned to the joint with the largest vert-to-angle weight.
    May need to run this function under HMR environment
    '''
    smpl_dict = io.load_data('scripts/3d/neutral_smpl_with_cocoplus_reg.pkl')
    weights = smpl_dict['weights']
    vert2part = weights.argmax(axis=1).tolist()
    io.save_json(vert2part, 'scripts/3d/smpl_vert_to_bodypart.json')


def create_silhouette():
    # DeepFashion
    #smpl_pred_dir = 'datasets/DF_Pose/3d/hmr_dfm_v2/pred/'
    #output_dir = 'datasets/DF_Pose/Img/silhouette24/'
    #image_split = io.load_json('datasets/DF_Pose/Label/image_split_dfm.json')
    
    # Market-1501
    smpl_pred_dir = 'datasets/market1501/3d/hmr/pred/'
    output_dir = 'datasets/market1501/Images/silhouette24/'
    image_split = io.load_json('datasets/market1501/Label/image_split.json')

    faces = np.load('scripts/3d/smpl_faces.npy')
    vert2part = io.load_json('scripts/3d/smpl_vert_to_bodypart.json')

    def _func(face_id):
        if face_id == 4294967295:
            return 0
        else:
            verts = faces[face_id]
            part_id = vert2part[verts[0]] + 1
            return part_id
    
    _vfunc = np.vectorize(_func)

    io.mkdir_if_missing(output_dir)
    id_list = image_split['train'] + image_split['test']
    for sid in tqdm.tqdm(id_list):
        pred = io.load_data(smpl_pred_dir + '%s.pkl'%sid)
        vis = pred['visibility']
        silh = _vfunc(vis).astype(np.uint8)
        silh = cv2.medianBlur(silh, 5)
        imageio.imwrite(output_dir + '%s.bmp'%sid, silh)


def merge_silhouette():
    # for DeepFashion
    #image_split = io.load_json('datasets/DF_Pose/Label/image_split_dfm.json')
    #input_dir = 'datasets/DF_Pose/Img/silhouette24/'
    #output_dir = 'datasets/DF_Pose/Img/silhouette6/'

    # for Market-1501
    image_split = io.load_json('datasets/market1501/Label/image_split.json')
    input_dir = 'datasets/market1501/Images/silhouette24/'
    output_dir = 'datasets/market1501/Images/silhouette6/'

    id_list = image_split['train'] + image_split['test']
    io.mkdir_if_missing(output_dir)

    head = {13,16}
    torso = {1,4,7,10}
    larm = {14,17,19,21,23}
    rarm = {15,18,20,22,24}
    lleg = {2,5,8,11}
    rleg = {3,6,9,12}
        
    def _func_merge(x):
        if x == 0:
            return 0
        elif x in head:
            return 1
        elif x in torso:
            return 2
        elif x in larm:
            return 3
        elif x in rarm:
            return 4
        elif x in lleg:
            return 5
        elif x in rleg:
            return 6
    
    _vfunc_merge = np.vectorize(_func_merge, otypes=[np.uint8])

    for sid in tqdm.tqdm(id_list):
        silh = imageio.imread(input_dir+'%s.bmp'%sid)
        silhm = _vfunc_merge(silh)
        imageio.imwrite(output_dir+'%s.bmp'%sid, silhm)

def create_hmr_pose_label():
    '''
        name: [openpose_index, hmr_index]:
        
        nose: 0,14
        neck: 1, 12
        right_shoulder: 2, 8
        right_elow: 3, 7
        right_wrist: 4, 6
        left_shoulder: 5, 9
        left_elbow: 6, 10
        left_wrist: 7, 11
        right_hip: 8, 2
        right_knee: 9, 1
        right_ankle: 10, 0
        left_hip: 11, 3
        left_knee: 12, 4
        left_ankle: 13, 5
        right_eye: 14, 16
        left_eye: 15, 15
        right_ear: 16, 18
        left_ear: 17, 17
        head_top: -, 13
    '''
    # DeepFashion
    #image_split = io.load_json('datasets/DF_Pose/Label/image_split_dfm.json')
    #smpl_pred_dir = 'datasets/DF_Pose/3d/hmr_dfm_v3/pred/'
    #fn_out = 'datasets/DF_pose/Label/pose_label_hmr.pkl'

    # Market-1501
    image_split = io.load_json('datasets/market1501/Label/image_split.json')
    smpl_pred_dir = 'datasets/market1501/3d/hmr/pred/'
    fn_out = 'datasets/market1501/Label/pose_label_hmr.pkl'

    id_list = image_split['train'] + image_split['test']
    kp_order_map = [14,12,8,7,6,9,10,11,2,1,0,3,4,5,16,15,18,17]
    
    pose_label_hmr = {}
    for sid in tqdm.tqdm(id_list):
        pred = io.load_data(smpl_pred_dir + '%s.pkl'%sid)
        joints = pred['joints']
        pts = joints[kp_order_map]
        pts[(pts[:,0]<0)|(pts[:,0]>255)|(pts[:,1]<0)|(pts[:,1]>255)]=-1
        pose_label_hmr[sid] = pts.tolist()
    io.save_data(pose_label_hmr, fn_out)

def create_hmr_pose_label_adapt():
    '''
    This is to create a version of hmr_pose joint, which is adapted to openpose joint:
    - compute "neck" using a regressor, trained useing openpose joints (due to the different definition of neck point"
    - invalidate the joint points which is invalid in dfm_pose
    '''
    from sklearn.linear_model import RidgeCV

    # DeepFashion
    #joint_label = io.load_data('datasets/DF_Pose/Label/pose_label_dfm.pkl')
    #joint_label_hmr = io.load_data('datasets/DF_Pose/Label/pose_label_hmr.pkl')
    #fn_out = 'datasets/DF_Pose/Label/pose_label_hmr_adapt.pkl'

    # Market-1501
    joint_label = io.load_data('datasets/market1501/Label/pose_label.pkl')
    joint_label_hmr = io.load_data('datasets/market1501/Label/pose_label_hmr.pkl')
    fn_out = 'datasets/market1501/Label/pose_label_hmr_adapt.pkl'

    # train a linear regressor, which predict neck point location from left/right shoulder locations
    print('training regressor...')
    pts_dfm = np.array(joint_label.values()) #(N,18,2)
    v = (pts_dfm[:,[1,2,5],:].reshape(-1,6) >= 0).all(axis=1)
    x_train = (pts_dfm[v])[:,[2,5]].reshape(-1,4) #shoulder points
    y_train = (pts_dfm[v])[:,1].reshape(-1,2) #neck points
    reg = RidgeCV(normalize=False)
    reg.fit(x_train, y_train)

    pts_hmr = np.array(joint_label.values())
    x_test = pts_hmr[:,[2,5],:].reshape(-1,4)
    y_test = reg.predict(x_test).reshape(-1,2)

    # generate adapted joint label
    joint_label_adapt = {}
    for idx, sid in enumerate(tqdm.tqdm(joint_label_hmr.keys())):
        p_h = np.array(joint_label_hmr[sid])
        p_d = np.array(joint_label[sid])
        if (p_h[[2,5],:] >= 0).all():
            p_h[1,:] = y_test[idx]
        
        inv = (p_d < 0).any(axis=1) | (p_h < 0).any(axis=1) | (p_h > 255).any(axis=1) # invalid joint points in joint_dfm will also be marked as invalid in joint_hmr
        p_h[inv,:] = -1
        joint_label_adapt[sid] = p_h.tolist()
    
    io.save_data(joint_label_adapt, fn_out)
    
if __name__ == '__main__':
    # divide_vert_into_bodypart()
    #create_silhouette()
    #merge_silhouette()
    #create_hmr_pose_label()
    create_hmr_pose_label_adapt()
