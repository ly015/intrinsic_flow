# Dense Intrinsic Appearance Flow for Human Pose Transfer

This is a pytorch implementation of the CVPR 2019 paper [Dense Intrinsic Appearance Flow for Human Pose Transfer](http://mmlab.ie.cuhk.edu.hk/projects/pose-transfer/).

![fig_intro](imgs/fig_intro.png)

## Requirements
- python 2.7
- pytorch (0.4.0)
- numpy
- opencv
- scikit-image
- tqdm
- imageio

Install dependencies:
```
pip install -r requirements.txt
```

## Resources
#### Datasets
Download and unzip preprocessed datasets with the following scripts.
```
bash scripts/download_deepfashion.sh
bash scripts/download_market1501.sh
```
Or you can manually download them from the following links:
- DeepFashion (23GB): [Google Drive](https://drive.google.com/file/d/1LbibHhhF7xA7G3hHoHj9I-MvCByzdkvr/view?usp=sharing)
- Market-1501 (9GB): [Google Drive](https://drive.google.com/file/d/16zZJ5f5qOJcgg-cPfmAdso8al-MSWiwu/view?usp=sharing)

#### Pretrained Models
Download pretrained models with the following scripts.
```
bash scripts/download_models.sh
```
Pretrained models below will be downloaded into the folder ./checkpoints. You can manually donwload them from [here](https://drive.google.com/file/d/1QHcb1QBGVmGginpsYmer-Q5Aq0HNtBHv/view?usp=sharing).

| Deepfashion | Market-1501 | Others |
|-------------|-------------|--------|
|<ul><li>`PoseTransfer_0.1` (w/o. dual encoder)</li><li>`PoseTransfer_0.2` (w/o. flow)</li><li>`PoseTransfer_0.3` (w/o. vis)</li><li>`PoseTransfer_0.4` (w/o. pxiel warping)</li><li>`PoseTransfer_0.5` (full)</li></ul>|<ul><li>`PoseTransfer_m0.1` (w/o. dual encoder)</li><li>`PoseTransfer_m0.2` (w/o. flow)</li><li>`PoseTransfer_m0.3` (w/o. vis)</li><li>`PoseTransfer_m0.4` (w/o. pxiel warping)</li><li>`PoseTransfer_m0.5` (full)</li></ul>|<ul><li>`Fasion_Inception`(compute FashionIS)</li><li>`Fasion_Attr`(compute AttrRec-k)</li></ul>|

## Testing
#### DeepFashion
1. Run scripts/test_pose_transfer.py to generate images and compute SSIM score.
```
python scripts/test_pose_transfer.py --gpu_ids 0 --id PoseTransfer_0.5 --which_epoch best --save_output
```
2. Compute inception score with the following script. (Note that this script is derived from [improved-gan](https://github.com/openai/improved-gan) and needs Tensorflow)
```
# python scripts/inception_score.py image_dir gpu_ids
python scripts/inception_score.py checkpoints/PoseTransfer_0.5/output/ 0
```
3. Compute fashionIS and AttrRec-k with the following scripts. 
```
# FashionIS
python scripts/fashion_inception_score.py --test_dir checkpoints/PoseTransfer_0.5/output/

# AttrRec-k
python scripts/fashion_attribute_score.py --test_dir checkpoints/PoseTransfer_0.5/output/
```

#### Market-1501
1. Run scripts/test_pose_transfer.py to generate images and compute SSIM/masked-SSIM score.
```
python scripts/test_pose_transfer.py --gpu_ids 0 --id PoseTransfer_m0.5 --which_epoch best --save_output --masked
```
2. Compute inception score or masked inception score with following scripts.
```
# IS
python scripts/inception_score.py checkpoints/PoseTransfer_m0.5/output/ 0

# masked-IS (only for market-1501)
python scripts/masked_inception_score.py checkpoints/PoseTransfer_m0.5/output/ 0
```

## Training
#### DeepFashion
1. Train flow regression module. (See all options in ./options/flow_regression_options.py)
```
python scripts/train_flow_regression_module.py --id id_flow --gpu_ids 0 --which_model unet --dataset_name deepfashion
```
You can alternativelly set `--which_model unet_v2` to use a improved version of network architecture with fewer parameters (only tested on Market-1501).

2. Train human pose transfer models. Set `--pretrained_flow_id` and `--pretrained_flow_epoch` to load the flow regression module. (See all options in ./options/pose_transfer_options.py)
```
# w/o. dual encoder
python scripts/train_pose_transfer_model.py --id id_pose_1 --gpu_ids 1 --dataset_name deepfashion --which_model_G unet

# w/o. flow
python scripts/train_pose_transfer_model.py --id id_pose_2 --gpu_ids 2 --dataset_name deepfashion --which_model_G dual_unet --G_feat_warp 0

# w/o. visibility
python scripts/train_pose_transfer_model.py --id id_pose_3 --gpu_ids 3 --dataset_name deepfashion --which_model_G dual_unet --G_feat_warp 1 --G_vis_mode none

# w/o. pixel warping
python scripts/train_pose_transfer_model.py --id id_pose_4 --gpu_ids 4 --dataset_name deepfashion --which_model_G dual_unet --G_feat_warp 1 --G_vis_mode residual

# full (need a pretrained pose transfer model without pixel warping)
python scripts/train_pose_transfer_model.py --id id_pose_5 --gpu_ids 5 --dataset_name deepfashion --G_pix_warp 1 --which_model_G dual_unet --pretrained_G_id id_pose_4 --pretrained_G_epoch 8
```

#### Market-1501
Set `--dataset_name market` to train models on Market-1501 dataset. Data related parameters will be automatically adjusted (see `.auto_set()` in ./options/flow_regression_options.py and ./options/pose_transfer_options.py for details).

## Citation
```
@inproceedings{li2019dense,
  author = {Li, Yining and Huang, Chen and Loy, Chen Change},
  title = {Dense Intrinsic Appearance Flow for Human Pose Transfer},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2019}}
```



