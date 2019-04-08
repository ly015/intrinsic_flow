from base_options import BaseOptions

class BasePoseTransferOptions(BaseOptions):
    def initialize(self):
        super(BasePoseTransferOptions, self).initialize()
        parser = self.parser
        ##############################
        # Model Setting
        ##############################
        parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'network initialization method [normal|xavier|kaiming|orthogonal]')
        # parser.add_argument('--no_dropout', type=int, default=1, choices=[0,1], help='no dropout in generator')
        # netG (general)
        parser.add_argument('--which_model_G', type=str, default='dual_unet', choices=['unet', 'dual_unet'], help='generator network architecture')
        parser.add_argument('--pretrained_G_id', type=str, default=None)
        parser.add_argument('--pretrained_G_epoch', type=str, default='best') 
        parser.add_argument('--G_nf', type=int, default=32, help='feature dimension at the bottom layer')
        parser.add_argument('--G_max_nf', type=int, default=128, help='max feature dimension')
        parser.add_argument('--G_n_scale', type=int, default=7, help='scale level number')
        parser.add_argument('--G_norm', type=str, default='batch', choices=['none', 'batch', 'instance'], help='type of normalization layer')
        parser.add_argument('--G_activation', type=str, default='relu', choices=['relu', 'leaky_relu'], help='type of activation function')
        parser.add_argument('--G_pose_type', type=str, default='joint_2')
        parser.add_argument('--G_appearance_type', type=str, default='img_1')
        # netG (only dual unet)
        parser.add_argument('--G_feat_warp', type=int, default=1, choices=[0,1], help='set 1 to use feature warping; otherwise the model is a simple unet with 2 encoders for pose and appearance respectively')
        parser.add_argument('--G_n_warp_scale', type=int, default=5, help='at scales higher than this, feature warping will not be performed (because the resolution of feature map is too small)')
        parser.add_argument('--G_vis_mode', type=str, default='residual', choices=['none', 'hard_gate', 'soft_gate', 'residual', 'res_no_vis'], help='different approaches to integrate visibility map in feature warping module')
        parser.add_argument('--G_no_end_norm', type=int, default=0, choices=[0,1], help='if set as 1, convolution at the start and the end of netG will not followed by norm_layer like BN.')
        # netG (pixel warping module)
        parser.add_argument('--G_pix_warp', type=int, default=0, choices=[0,1], help='use pixel warping module')
        # parser.add_argument('--G_pix_warp', type=str, default='none', choices=['none', 'mask', 'mask+flow', 'ext_mask', 'ext_mask+flow', 'exth_mask', 'exth_mask+flow'], help='combine generated image_2 and warped image_1 to synthesize final output. "mask": netG output a soft-mask to combine img_gen and img_warp; "mask+flow": netG output a soft-mask and a flow residual')
        parser.add_argument('--G_pix_warp_input_type', type=str, default='img_out_G+img_warp+vis_out+flow_out')
        parser.add_argument('--G_pix_warp_detach', type=int, default=1, choices=[0,1], help='generated image will be detached when it is used to combine with warped image. Thus the gradient from combined image will only propagate backward to soft-mask')
        # netD
        parser.add_argument('--D_nf', type=int, default=64, help='feature number of first conv layer in netD')
        parser.add_argument('--D_n_layers', type=int, default=3, help='number of conv layers in netD (patch gan)')
        parser.add_argument('--gan_type', type=str, default='dcgan', choices=['lsgan', 'dcgan'], help='gan loss type')
        parser.add_argument('--D_input_type_real', type=str, default='img_1+img_2+joint_2', help='input data items to netD')
        parser.add_argument('--D_input_type_fake', type=str, default='img_1+img_out+joint_2', help='input data items to netD')
        # netF
        parser.add_argument('--flow_on_the_fly', type=int, default=1, choices=[0,1], help='use a flow3d model to generate flow on-the-fly')
        parser.add_argument('--F_input_type', type=str, default='joint_1+joint_2', help='input data items for netF(flow) which flow is generated on-the-fly')
        parser.add_argument('--pretrained_flow_id', type=str, default='FlowReg_0.1', help='model id of flow regression model')
        # parser.add_argument('--pretrained_flow_id', type=str, default='FlowReg_0.2')
        parser.add_argument('--pretrained_flow_epoch', type=str, default='best', help='which epoch to load pretrained flow regression module')
        ##############################
        # Pose Setting
        ##############################
        parser.add_argument('--joint_nc', type=int, default=18, help='2d joint number. 18 for openpose joint')
        parser.add_argument('--joint_mode', type=str, default='binary', choices=['binary', 'gaussian'])
        parser.add_argument('--joint_radius', type=int, default=8, help='radius of joint map')
        parser.add_argument('--seg_nc', type=int, default=7, help='number of segmentation classes')
        ##############################
        # data setting (dataset_mode == general_pair)
        ##############################
        parser.add_argument('--dataset_type', type=str, default='pose_transfer', help='type of dataset. see data/data_loader.py')
        parser.add_argument('--dataset_name', type=str, default='deepfashion', choices=['deepfashion', 'market'])
        parser.add_argument('--image_size', type=int, nargs='+', default=[256,256])
        parser.add_argument('--batch_size', type = int, default = 8, help = 'batch size')
        parser.add_argument('--data_root', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--fn_split', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--img_dir', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--seg_dir', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--corr_dir', type=str, default=None, help='Set in Options.auto_set()')
        parser.add_argument('--fn_pose', type=str, default=None, help='Set in Options.auto_set()')
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
        if not opt.id.startswith('PoseTransfer_'):
            opt.id = 'PoseTransfer_' + opt.id
        ###########################################
        # Set dataset path
        ###########################################
        if opt.dataset_name == 'market':
            opt.image_size=[128,64]
            opt.joint_radius = 4
            opt.G_n_scales = 5
            opt.G_n_scale = 5
            opt.G_n_warp_scale = 4

            opt.use_augmentation = 1
            opt.data_root = 'datasets/market1501'
            opt.fn_split = 'Label/pair_split.json'
            opt.img_dir = 'Images/img/'
            opt.seg_dir = 'Images/silhouette6/'
            opt.fn_poes = 'Label/pose_label.pkl'
            opt.corr_dir = '3d/hmr/corr/'

            opt.pretrained_flow_id = 'Flow3d_m1.0.0'
            
        else:
            opt.data_root = 'datasets/DF_Pose/'
            opt.fn_split = 'Label/pair_split_dfm_new.json'
            opt.img_dir = 'Img/img/'
            opt.seg_dir = 'Img/silhouette6/'
            opt.fn_pose = 'Label/pose_label_dfm.pkl' # joint coordinates detected by openpose
            opt.corr_dir = '3d/hmr_dfm_v2/corr/'


    
class TrainPoseTransferOptions(BasePoseTransferOptions):
    def initialize(self):
        super(TrainPoseTransferOptions, self).initialize()
        self.is_train = True
        parser = self.parser
        # basic
        parser.add_argument('--resume_train', action = 'store_true', default = False, help = 'resume training from saved checkpoint')
        parser.add_argument('--small_val_set', type=int, default=1, choices=[0,1], help='use 1/5 test samples as validation set')
        # optimizer
        parser.add_argument('--lr', type = float, default = 2e-4, help = 'initial learning rate')
        parser.add_argument('--beta1', type = float, default = 0.5, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
        parser.add_argument('--lr_D', type=float, default=2e-5)
        parser.add_argument('--weight_decay_D', type=float, default=4e-4)
        # scheduler
        parser.add_argument('--lr_policy', type=str, default='step', choices = ['step', 'plateau', 'lambda'], help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--niter', type = int, default=30, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=100, help='multiply by a gamma every lr_decay_interval epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.1, help='lr decay rate')
        parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--save_epoch_freq', type = int, default = 1, help='frequency of saving model to disk' )
        parser.add_argument('--vis_epoch_freq', type = int, default = 1, help='frequency of visualizing generated images')
        parser.add_argument('--check_grad_freq', type = int, default = 100, help = 'frequency of checking gradient of each loss')
        parser.add_argument('--nvis', type = int, default = 64, help='number of visualized images')
        # loss setting
        parser.add_argument('--epoch_add_gan', type=int, default=6, help='add gan loss after # epochs of training')
        parser.add_argument('--loss_weight_l1', type=float, default=1.)
        parser.add_argument('--loss_weight_content', type=float, default=1.)
        parser.add_argument('--loss_weight_style', type=float, default=0)
        parser.add_argument('--loss_weight_gan', type=float, default=0.01)
        parser.add_argument('--shifted_style_loss', type=int, default=1, choices=[0,1])
        #parser.add_argument('--vgg_content_weights', type=float, nargs='+', default=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0])
        parser.add_argument('--vgg_content_weights', type=float, nargs='+', default=[0.125, 0.125, 0.125, 0.125, 0.125])
        parser.add_argument('--vgg_content_mode', type=str, default='balance', choices=['balance', 'imbalance', 'special'])

    def auto_set(self):
        super(TrainPoseTransferOptions, self).auto_set()
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
        parser.add_argument('--data_split', type=str, default='test')
        parser.add_argument('--nbatch', type=int, default=-1, help='set number of minibatch used for test')
        # visualize samples
        parser.add_argument('--visualize', action='store_true', help='visualize result')
        parser.add_argument('--test_nvis', type = int, default = 64, help='number of visualized images')
        # save generated images
        parser.add_argument('--save_output', action='store_true', help='save output images in the folder exp_dir/test/')
        parser.add_argument('--output_dir', type=str, default='output', help='path to save generated images')
        # visualize batch
        parser.add_argument('--visualize_batch', action='store_true')
        parser.add_argument('--test_batch_nvis', type=int, default=20)
        
