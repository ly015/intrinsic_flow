from base_options import BaseOptions

class BasePoseTransferOptions(BaseOptions):
    def initialize(self):
        super(BasePoseTransferOptions, self).initialize()
        parser = self.parser
        ##############################
        # General Setting
        ##############################
        # parser.add_argument('--pavi', action = 'store_true', help = 'activate pavi log')
        parser.add_argument('--resume_epoch', type=int, default=-1, help='set which epoch to resume training.')
        ##############################
        # Model Setting
        ##############################
        parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'network initialization method [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--no_dropout', type=int, default=1, choices=[0,1], help='no dropout in generator')
        # netG (general)
        parser.add_argument('--pretrained_G_id', type=str, default=None)
        parser.add_argument('--pretrained_G_epoch', type=str, default='best')
        parser.add_argument('--which_model_G', type=str, default='unetfw', choices=['unet', 'resnet', 'unet1', 'vunet', 'unet3', 'unetfw'], help='generator network architecture')
        parser.add_argument('--norm', type=str, default='batch', choices=['none', 'batch', 'instance'], help='type of normalization layers')
        parser.add_argument('--G_input_type', type=str, default='joint_2+img_1+img_warp+vismap_out', help='data items to feed into netG, concatenated by "+"')
        # netG (unet, unet1 and resnet)
        parser.add_argument('--G_nf', type=int, default=64, help='feature number of first conv layer in netG')
        parser.add_argument('--G_max_nf', type=int, default=512, help='max feature number of layers in netG, only for unet now')
        # netG (vunet)
        parser.add_argument('--G_vunet_pose_type1', type=str, default='joint_1', help='data items to feed into netG (vunet) as pose representation of image ref')
        parser.add_argument('--G_vunet_pose_type2', type=str, default='joint_2', help='data items to feed into netG (vunet) as pose representation of image tar')
        parser.add_argument('--G_vunet_appearance_type', type=str, default='img_1', help='data items to feed into netG (vunet) as pose representation of image ref')
        parser.add_argument('--vunet_nf', type=int, default=32, help='vunet setting: channel number of the first conv layer')
        parser.add_argument('--vunet_max_nf', type=int, default=128, help='vunet setting: max channel number of mid-level conv layers')
        parser.add_argument('--vunet_n_latent_scales', type=int, default=2, help='vunet setting: layer number of latent space')
        parser.add_argument('--vunet_bottleneck_factor', type=int, default=2, help='vunet setting: the bottleneck resolution will be 2**#')
        parser.add_argument('--vunet_box_factor', type=int, default=2, help='vunet setting: the size of pose input will be reduced 2**# times')
        parser.add_argument('--vunet_activation', type=str, default='relu', choices=['relu', 'elu'], help='activation type')
        # netG (unet3)
        parser.add_argument('--G_unet3_nf', type=int, default=32)
        parser.add_argument('--G_unet3_max_nf', type=int, default=128)
        parser.add_argument('--G_unet3_n_scales', type=int, default=7)
        # netG (unet_fw)
        parser.add_argument('--G_unetfw_usefw', type=int, default=1, choices=[0,1], help='set 1 to use feature warping; otherwise the model is a simple unet with 2 encoders for pose and appearance respectively')
        parser.add_argument('--G_unetfw_pose_type', type=str, default='joint_2')
        parser.add_argument('--G_unetfw_appearance_type', type=str, default='img_1')
        parser.add_argument('--G_unetfw_nf', type=int, default=32)
        parser.add_argument('--G_unetfw_max_nf', type=int, default=128)
        parser.add_argument('--G_unetfw_n_scale', type=int, default=7, help='number of scales in unet')
        parser.add_argument('--G_unetfw_n_warp_scale', type=int, default=5, help='at scales higher than this, feature warping will not be performed (because the resolution of feature map is too small)')
        parser.add_argument('--G_unetfw_vis_mode', type=str, default='residual', choices=['none', 'hard_gate', 'soft_gate', 'residual', 'res_no_vis'])
        
        parser.add_argument('--G_pix_warp', type=str, default='none', choices=['none', 'mask', 'mask+flow', 'ext_mask', 'ext_mask+flow', 'exth_mask', 'exth_mask+flow'], help='combine generated image_2 and warped image_1 to synthesize final output. "mask": netG output a soft-mask to combine img_gen and img_warp; "mask+flow": netG output a soft-mask and a flow residual')
        parser.add_argument('--epw_input_type', type=str, default='img_out_G+img_warp+vis_out+flow_out')
        parser.add_argument('--G_pix_warp_detachG', type=int, default=1, choices=[0,1], help='generated image will be detached when it is used to combine with warped image. Thus the gradient from combined image will only propagate backward to soft-mask')
        parser.add_argument('--G_pix_warp_flow_scale', type=float, default=3)
        parser.add_argument('--G_activation', type=str, default='relu', choices=['relu', 'leaky_relu'], help='choose activation function for netG')
        parser.add_argument('--G_unetfw_no_end_norm', type=int, default=0, choices=[0,1], help='if set as 1, convolution at the start and the end of netG will not followed by norm_layer like BN.')
        # netD
        parser.add_argument('--D_nf', type=int, default=64, help='feature number of first conv layer in netD')
        parser.add_argument('--D_n_layers', type=int, default=3, help='number of conv layers in netD (patch gan)')
        parser.add_argument('--gan_type', type=str, default='dcgan', choices=['lsgan', 'dcgan'], help='gan loss type')
        parser.add_argument('--D_input_type_real', type=str, default='img_1+img_2+joint_2', help='input data items to netD')
        parser.add_argument('--D_input_type_fake', type=str, default='img_1+img_out+joint_2', help='input data items to netD')
        # netL (layout predictor, or silh predictor)
        parser.add_argument('--silh_on_the_fly', type=int, default=0, choices=[0,1], help='use a silh generator to generate silh on-the-fly')
        parser.add_argument('--L_input_type_1', type=str, default='joint_1', help='input data items for netL(layout) when silh is generated on-the-fly')
        parser.add_argument('--L_input_type_2', type=str, default='joint_2', help='input data items for netL(layout) when silh is generated on-the-fly')
        parser.add_argument('--pretrained_layout_id', type=str, default='Layout_4.0.0')
        parser.add_argument('--pretrained_layout_epoch', type=str, default='best')
        # netF
        parser.add_argument('--flow_on_the_fly', type=int, default=1, choices=[0,1], help='use a flow3d model to generate flow on-the-fly')
        #parser.add_argument('--F_input_type', type=str, default='joint_1+silh_1+joint_2+silh_2', help='input data items for netF(flow) which flow is generated on-the-fly')
        parser.add_argument('--F_input_type', type=str, default='joint_1+joint_2', help='input data items for netF(flow) which flow is generated on-the-fly')
        #parser.add_argument('--pretrained_flow3d_id', type=str, default='Flow3d_14.0.0')
        #parser.add_argument('--pretrained_flow3d_id', type=str, default='Flow3d_14.0.2')
        parser.add_argument('--pretrained_flow3d_id', type=str, default='Flow3d_15.0.0')
        parser.add_argument('--pretrained_flow3d_epoch', type=str, default='best')
        ##############################
        # Pose Setting
        ##############################
        parser.add_argument('--joint_nc', type=int, default=18, help='2d joint number. 18 for openpose joint')
        parser.add_argument('--joint_mode', type=str, default='binary', choices=['binary', 'gaussian'])
        parser.add_argument('--joint_radius', type=int, default=8, help='radius of joint map')
        parser.add_argument('--seg_nc', type=int, default=8, help='number of segmentation classes, 7 for ATR, 8 for LIP, 20 for LIP-FULL')
        parser.add_argument('--silh_nc', type=int, default=7, help='number of silhouette channels')
        ##############################
        # data setting (dataset_mode == general_pair)
        ##############################
        parser.add_argument('--dataset_mode', type=str, default='general_pair', help='type of dataset. see data/data_loader.py')
        parser.add_argument('--dataset_name', type=str, default='dfm', choices=['dfm', 'dfm_aug', 'dfm_easy', 'market'])
        parser.add_argument('--image_size', type=int, nargs='+', default=[256,256])
        parser.add_argument('--batch_size', type = int, default = 8, help = 'batch size')
        parser.add_argument('--data_item_list', type=list, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--data_root', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--fn_split', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--img_dir', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--seg_dir', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--silh_dir', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--corr_dir', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--fn_joint', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--fn_view', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--debug', action='store_true', help='debug')

        parser.add_argument('--use_augmentation', type=int, default=0, choices=[0,1])
        parser.add_argument('--aug_scale_range', type=float, default=1.2)
        parser.add_argument('--aug_shiftx_range', type=int, default=10)
        parser.add_argument('--aug_shifty_range', type=int, default=10)
        parser.add_argument('--aug_color_jit', type=int, default=0, choices=[0,1])
        parser.add_argument('--vis_smooth_rate', type=int, default=5, help='use a median filter of size # to smooth the visiblity map')

    def auto_set(self):
        super(BasePoseTransferOptions, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        ###########################################
        if not opt.id.startswith('CTransfer_'):
            opt.id = 'CTransfer_' + opt.id
        ###########################################
        # Set dataset path
        ###########################################
        if opt.dataset_name == 'market':
            opt.G_unet3_n_scales = 5
            opt.G_unetfw_n_scale = 5
            opt.G_unetfw_n_warp_scale = 4

            opt.joint_radius = 4

            opt.data_root = 'datasets/market1501'
            opt.fn_split = 'Label/pair_split.json'
            opt.img_dir = 'Images/img/'
            opt.seg_dir = 'Images/silhouette6/'
            opt.silh_dir = 'Images/silhouette6/'
            opt.fn_joint = 'Label/pose_label.pkl'
            opt.fn_view = 'Label/view_label.json'
            opt.corr_dir = '3d/hmr/corr/'

            opt.pretrained_flow3d_id = 'Flow3d_m1.0.0'
            opt.image_size=[128,64]

            opt.use_augmentation = 1
        else:
            if opt.data_root is None:
                opt.data_root = 'datasets/DF_Pose/'
            if opt.img_dir is None:
                opt.img_dir = 'Img/img/'
            if opt.seg_dir is None:
                opt.seg_dir = 'Img/seg-lip/'
            if opt.silh_dir is None:
                #opt.silh_dir = 'External/silh_3.0.0/'
                opt.silh_dir = 'External/silh_4.0.0/'
                #opt.seg_dir = 'Img/seg-lip-full/'
                #opt.silh_dir = 'Img/silhouette6/'
            if opt.fn_joint is None:
                opt.fn_joint = 'Label/pose_label_dfm.pkl' # joint coordinates detected by openpose
                # opt.fn_joint = 'Label/pose_label_hmr.pkl' # joint coordinates prjected from a SMPL model fitted using HMR
            if opt.fn_view is None:
                opt.fn_view = 'Label/view_label_dfm.json'
            if opt.corr_dir is None:
                opt.corr_dir = '3d/hmr_dfm_v2/corr/'

            if opt.dataset_name == 'dfm':
                if opt.fn_split is None:
                    #opt.fn_split = 'Label/pair_split_dfm_new.json'
                    opt.fn_split = 'Label/pair_split_dfm_old.json'
                    #opt.fn_split = 'Label/pair_split_dfm.json'
            elif opt.dataset_name == 'dfm_easy':
                if opt.fn_split is None:
                    opt.fn_split = 'Label/pair_split_dfm_easy-front-side.json'

        ###########################################
        # Set data_item_list
        ###########################################
        opt.data_item_list = ['img', 'silh', 'joint', 'flow']

    
class TrainPoseTransferOptions(BasePoseTransferOptions):
    def initialize(self):
        super(TrainPoseTransferOptions, self).initialize()
        self.is_train = True
        parser = self.parser
        # basic
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        parser.add_argument('--small_val_set', type=int, default=1, choices=[0,1], help='use 1/5 test samples as validation set')
        # optimizer
        parser.add_argument('--lr', type = float, default = 2e-4, help = 'initial learning rate')
        # parser.add_argument('--lr_warmup', type=int, default = 0, help='if lr_warmup > 0, lr will increate from 0.01xlr to 1.0xlr in first lr_warmup steps')
        parser.add_argument('--beta1', type = float, default = 0.5, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
        parser.add_argument('--lr_D', type=float, default=2e-5)
        parser.add_argument('--weight_decay_D', type=float, default=4e-4)
        # scheduler
        parser.add_argument('--lr_policy', type=str, default='step', choices = ['step', 'plateau', 'lambda'], help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--niter', type = int, default=30, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=10, help='multiply by a gamma every lr_decay_interval epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.1, help='lr decay rate')
        parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--save_epoch_freq', type = int, default = 1, help='frequency of saving model to disk' )
        parser.add_argument('--vis_epoch_freq', type = int, default = 1, help='frequency of visualizing generated images')
        parser.add_argument('--check_grad_freq', type = int, default = 100, help = 'frequency of checking gradient of each loss')
        parser.add_argument('--nvis', type = int, default = 64, help='number of visualized images')
        # loss setting
        parser.add_argument('--loss_weight_l1', type=float, default=1.)
        parser.add_argument('--loss_weight_content', type=float, default=1.)
        #parser.add_argument('--loss_weight_style', type=float, default=0.001)
        parser.add_argument('--loss_weight_style', type=float, default=0)
        parser.add_argument('--loss_weight_gan', type=float, default=0.01)
        #parser.add_argument('--loss_weight_style', type=float, default=0.)
        #parser.add_argument('--loss_weight_gan', type=float, default=0.)
        parser.add_argument('--shifted_style_loss', type=int, default=1, choices=[0,1])
        parser.add_argument('--loss_weight_flow_residual', type=float, default=0.)
        parser.add_argument('--loss_weight_kl', type=float, default=1.0e-6)
        #parser.add_argument('--vgg_content_weights', type=float, nargs='+', default=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0])
        parser.add_argument('--vgg_content_weights', type=float, nargs='+', default=[0.125, 0.125, 0.125, 0.125, 0.125])
        parser.add_argument('--vgg_content_mode', type=str, default='balance', choices=['balance', 'imbalance', 'special'])
        parser.add_argument('--loss_weight_pix_warp', type=float, default=0.5, help='loss = w*loss(img_out)+(1-w)*loss(img_out_G)')

    def auto_set(self):
        super(TrainCascadeTransferOptions, self).auto_set()
        opt = self.opt

        if opt.vgg_content_mode == 'balance':
            opt.vgg_content_weights = [0.125, 0.125, 0.125, 0.125, 0.125]
        elif opt.vgg_content_mode == 'imbalance':
            opt.vgg_content_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
    
class TestPoseTransferOptions(BasePoseTransferOptions):
    def initialize(self):
        super(TestPoseTransferOptions, self).initialize()
        self.is_train = False
        parser = self.parser
        parser.add_argument('--nbatch', type=int, default=-1, help='set number of minibatch used for test')
        parser.add_argument('--save_output', action='store_true', help='save output images in the folder exp_dir/test/')
        parser.add_argument('--output_dir', type=str, default='output', help='path to save generated images')
        parser.add_argument('--save_seg', action='store_true', help='save segmentation outputs in the folder exp_dir/test_seg/')
        parser.add_argument('--visualize', action='store_true', help='visualize result')
        parser.add_argument('--test_nvis', type = int, default = 64, help='number of visualized images')
        parser.add_argument('--data_split', type=str, default='test')
        parser.add_argument('--small_val_set', type=int, default=1, choices=[0,1], help='use 1/5 test samples as validation set')

        parser.add_argument('--visualize_batch', action='store_true')
        parser.add_argument('--test_batch_nvis', type=int, default=20)
        parser.add_argument('--mask', action='store_true', help='use skeloton mask')

        # cross domain options
        parser.add_argument('--cross_domain', action='store_true')
        parser.add_argument('--cross_domain_dataset_info_fn', type=str, default='datasets/DF_Pose/CrossDomain/DeepFashion_to_Market1501.json')

    def auto_set(self):
        super(TestCascadeTransferOptions, self).auto_set()
        opt = self.opt
        if opt.cross_domain:
            self.opt.dataset_mode = 'general_cross_domain'
    
        


