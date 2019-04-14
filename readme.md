# Dense Intrinsic Appearance Flow for Human Pose Transfer

This is a pytorch implementation of the CVPR 2019 paper [Dense Intrinsic Appearance Flow for Human Pose Transfer] (http://mmlab.ie.cuhk.edu.hk/projects/pose-transfer/).

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

### Datasets
Download and unzip preprocessed datasets with the following scripts:
```
```
Or manually download them from the following links.
- DeepFashion ()
- Market-1501 (9GB)

### Pretrained Models
Download pretrained models with the following scripts:
'''
'''

## Testing
### DeepFashion
1. Run scripts/test_pose_transfer.py to generate images and compute SSIM score.
'''
python scripts/test_pose_transfer.py --gpu_ids 0 --id 0.5 --which_epoch best --save_output
'''
2. Compute inception score with the following script:
'''
# python scripts/inception_score.py image_dir gpu_ids
python scripts/inception_score.py checkpoints/PoseTransfer_0.5/output/ 0
'''
3. Compute fashionIS and AttrRec-k with the following scripts:
'''
# FashionIS
python 
'''
### Market-1501

## Training




