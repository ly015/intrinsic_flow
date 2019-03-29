from base_options import BaseOptions

class BaseFlow3dOptions(BaseOptions):
    def initialize(self):
        super(BaseFlow3dOptions, self).initialize()
        parser = self.parser
        ##############################
        # General Setting
        ##############################
        parser.add_argument('--pavi', action = 'store_true', help = 'activate pavi log')
        ##############################
        # Model Setting
        ##############################
        parser.add_argument('--which_model', type=str, default='unet', choices=['unet', 'pwc', 'unet4'], help='"unet" is for debug')
        parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'network initialization method [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--norm', type=str, default='batch', choices=['batch', 'instance', 'none'])
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--nf', type=int, default=32, help='feature channel number of first conv layer')
        parser.add_argument('--max_nf', type=int, default=128, help='max feature number')
        parser.add_argument('--input_type1', type=str, default='img+seg+joint', help='model input for image 1')
        parser.add_argument('--input_type2', type=str, default='joint', help='model input for image 2')
        parser.add_argument('--start_scale', type=int, default=2, help='scale of the top-scale flow compared to original image')
        parser.add_argument('--num_scale', type=int, default=5, help='scale number of image/flow pyramid')
        parser.add_argument('--flow_loss_type', type=str, default='l2', choices=['l1', 'l2'], help='type of flow loss')
        # priori
        parser.add_argument('--use_flow_priori', type=int, default=0, choices=[0,1])
        parser.add_argument('--priori_nc', type=int, default=18)
        parser.add_argument('--priori_guide')
        # for PWC net
        parser.add_argument('--md', type=int, default=4, help='correlation will be computed within a max displacement limit of md pixels')
        parser.add_argument('--dense_connect', type=int, default=1, choices=[0,1], help='use dense connection in flow predictor at each level')
        parser.add_argument('--use_context_refine', type=int, default=0, choices=[0,1], help='use context refine network to postprocess flow')
        parser.add_argument('--input_feat_type', type=str, default='joint')
        parser.add_argument('--input_context_type', type=str, default='img+seg')
        parser.add_argument('--start_dim_level', type=int, default=0, help='set the channel number at the start layer(0:16, 1:32, 2:64, ...)')
        parser.add_argument('--residual', type=int, default=0, choices=[0,1], help='predict flow residual at each scale')
        # for Pose Refiner
        parser.add_argument('--use_post_refine', type=int, default=0, choices=[0,1])
        parser.add_argument('--which_model_PR', type=str,default='resblock', choices=['normal', 'dilation', 'autoencoder', 'resblock'])
        parser.add_argument('--pr_input_type1', type=str, default='flow_feat+flow+vis+img+seg+joint')
        parser.add_argument('--pr_input_type2', type=str, default='joint')
        parser.add_argument('--pr_recon_type', type=str, default='seg', choices=['seg', 'img'])
        parser.add_argument('--pr_recon_meas', type=str, default='l2', choices=['l2', 'ssim'])
        # for segmentation sensitive flow loss
        parser.add_argument('--use_ss_flow_loss', type=int, default=0, choices=[0,1])
        parser.add_argument('--ss_guide', type=str, default='silh', choices=['silh', 'seg'])
        # for GAN
        parser.add_argument('--use_gan', type=int, default=0, choices=[0,1])
        parser.add_argument('--gan_type', type=str, default='lsgan', choices=['lsgan', 'dcgan'])
        parser.add_argument('--D_input_type', type=str, default='warp_err', choices=['warp_err', 'flow_img', 'flow_silh'])
        ##############################
        # Pose Setting
        ##############################
        parser.add_argument('--joint_nc', type=int, default=18, help='2d joint number. 18 for openpose joint')
        parser.add_argument('--joint_mode', type=str, default='binary', choices=['binary', 'gaussian'])
        parser.add_argument('--joint_radius', type=int, default=8, help='radius of joint map')
        parser.add_argument('--seg_nc', type=int, default=20, help='number of segmentation classes, 7 for ATR and 8 for LIP, 20 for full LIP')
        parser.add_argument('--seg_bin_size', type=int, default=1, help='bin size of downsampled seg mask')
        parser.add_argument('--silh_nc', type=int, default=7, help='number of silhouette channels')
        ##############################
        # Output & Visualization Setting
        ##############################
        parser.add_argument('--vis_warp', type=int, default=1, choices=[0,1], help='visualize warped images according to flow_out and flow_gt respectively')
        parser.add_argument('--vis_flow_pyr', type=int, default=0, choices=[0,1], help='visualize flow at all levels')
        parser.add_argument('--output_full_losses', type=int, default=0, choices=[0,1], help='output flow loss at all levels')
        ##############################
        # data setting (dataset_mode == flow3d)
        ##############################
        parser.add_argument('--dataset_mode', type=str, default='flow3d', help='type of dataset. see data/data_loader.py')
        parser.add_argument('--dataset_name', type=str, default='dfm', choices=['dfm', 'dfm_aug', 'dfm_easy', 'market'])
        parser.add_argument('--image_size', type=int, nargs='+', default=[256,256], help='image size (H,W)')
        parser.add_argument('--batch_size', type = int, default = 8, help = 'batch size')
        parser.add_argument('--data_root', type=str, default=None, help='Set in BaseFlow3dOptions.auto_set()')
        parser.add_argument('--fn_split', type=str, default=None, help='Set in BaseFlow3dOptions.auto_set()')
        parser.add_argument('--img_dir', type=str, default=None, help='Set in BaseFlow3dOptions.auto_set()')
        parser.add_argument('--img_dir2', type=str, default=None, help='Set in BaseFlow3dOptions.auto_set()')
        parser.add_argument('--seg_dir', type=str, default=None, help='Set in BaseFlow3dOptions.auto_set()')
        parser.add_argument('--silh_dir', type=str, default=None, help='Set in BaseFlow3dOptions.auto_set()')
        parser.add_argument('--corr_dir', type=str, default=None, help='Set in BaseFlow3dOptions.auto_set()')
        parser.add_argument('--fn_pose', type=str, default=None, help='Set in BaseFlow3dOptions.auto_set()')
        parser.add_argument('--fn_pose2', type=str, default=None, help='Set in BaseFlow3dOptions.auto_set()')
        parser.add_argument('--debug', action='store_true', help='debug')

        parser.add_argument('--aug_scale_range', type=float, default=0)
        parser.add_argument('--aug_shiftx_range', type=int, default=20)
        parser.add_argument('--aug_shifty_range', type=int, default=10)
        parser.add_argument('--vis_smooth_rate', type=int, default=5, help='use a median filter of size # to smooth the visibility map')

    def auto_set(self):
        super(BaseFlow3dOptions, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        ###########################################
        if not opt.id.startswith('Flow3d_'):
            opt.id = 'Flow3d_' + opt.id
        ###########################################
        # Dataset Settings
        ###########################################
        if opt.dataset_name == 'market':
            opt.image_size = [128,64]
            opt.num_scale = 4
            opt.aug_shiftx_range = 5
            opt.aug_shifty_range = 5
            opt.joint_radius = 4
            opt.data_root = 'datasets/market1501/'
            opt.img_dir = 'Images/img/'
            opt.silh_dir = 'Images/silhouette6/'
            opt.seg_dir = 'Images/silhouette6/'
            opt.fn_pose = 'Label/pose_label.pkl'
            opt.fn_split = 'Label/pair_split.json'
            opt.corr_dir = '3d/hmr/corr/'
        else:
            if opt.data_root is None:
                opt.data_root = 'datasets/DF_Pose/'
            if opt.img_dir is None:
                opt.img_dir = 'Img/img/'
            if opt.seg_dir is None:
                # opt.seg_dir = 'Img/seg-lip/'
                opt.seg_dir = 'Img/seg-lip-full/'
            if opt.silh_dir is None:
                opt.silh_dir = 'Img/silhouette6/'
                #opt.silh_dir = 'External/silh_3.0.0/'
                #opt.silh_dir = 'External/silh_4.0.0/'
            if opt.fn_pose is None:
                #opt.fn_pose = 'Label/pose_label_dfm.pkl'
                opt.fn_pose = 'Label/pose_label_hmr_adapt.pkl'
            if opt.dataset_name == 'dfm':
                if opt.fn_split is None:
                    #opt.fn_split = 'Label/pair_split_dfm_new_clean.json'
                    opt.fn_split = 'Label/pair_split_dfm_old_clean.json'
                if opt.corr_dir is None:
                    opt.corr_dir = '3d/hmr_dfm_v2/corr/'
            elif opt.dataset_name == 'dfm_easy':
                if opt.fn_split is None:
                    opt.fn_split = 'Label/pair_split_dfm_easy-front-side.json'
                if opt.corr_dir is None:
                    opt.corr_dir = '3d/hmr_dfm_v2/corr/'
            elif opt.dataset_name == 'dfm_aug':
                if opt.fn_split is None:
                    opt.fn_split = 'Label/pair_split_dfm_aug.json'
                if opt.corr_dir is None:
                    opt.corr_dir = '3d/hmr_dfm_aug_v2/corr/'
                if opt.img_dir2 is None:
                    opt.img_dir2 = '3d/hmr_dfm_aug_v2/img2/'
                if opt.fn_pose2 is None:
                    opt.fn_pose2 = 'Label/pose_label_dfm_aug.pkl'

class TrainFlow3dOptions(BaseFlow3dOptions):
    def initialize(self):
        super(TrainFlow3dOptions, self).initialize()
        self.is_train = True
        parser =self.parser
        # basic
        parser.add_argument('--continue_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        # pretrain
        parser.add_argument('--pretrain_id', type=str, default='', help='id of pretrain model')
        parser.add_argument('--pretrain_epoch', type=str, default='latest', help='specify which epoch of the pretrain model to be used')
        # optimizer
        parser.add_argument('--lr', type = float, default = 1e-4, help = 'initial learning rate')
        parser.add_argument('--lr_warmup', type=int, default = 0, help='if lr_warmup > 0, lr will increate from 0.01xlr to 1.0xlr in first lr_warmup steps')
        parser.add_argument('--beta1', type = float, default = 0.9, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        parser.add_argument('--weight_decay', type=float, default=4e-4, help='weight decay')
        parser.add_argument('--fix_netF', action='store_true', help='fix netF parameter and only train other parts')
        parser.add_argument('--lr_D', type=float, default=1e-5, help='initial learning rate for discriminator')
        # scheduler
        parser.add_argument('--lr_policy', type=str, default='step', choices = ['step', 'plateau', 'lambda'], help='learning rate policy: lambda|step|plateau')
        # parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type = int, default=30, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=8, help='multiply by a gamma every lr_decay_interval epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.5, help='lr decay rate')
        parser.add_argument('--display_freq', type = int, default = 10, help='frequency of showing training results on screen')
        parser.add_argument('--test_epoch_freq', type = int, default = 1, help='frequency of testing model')
        parser.add_argument('--save_epoch_freq', type = int, default = 5, help='frequency of saving model to disk' )
        parser.add_argument('--vis_epoch_freq', type = int, default = 1, help='frequency of visualizing generated images')
        parser.add_argument('--check_grad_freq', type = int, default = 100, help = 'frequency of checking gradient of each loss')
        parser.add_argument('--nvis', type = int, default = 64, help='number of visualized images')
        # loss setting
        parser.add_argument('--loss_weight_flow', type=float, default=1., help='weight of multiscale flow loss. see models.networks_flow.MultiScaleFlowLoss')
        parser.add_argument('--loss_weight_vis', type=float, default=1., help='weight of visibility loss')
        parser.add_argument('--loss_weight_flow_ss', type=float, default=5., help='weight of segmentation-sensitive flow loss. see models.networks_flow.SS_FlowLoss')
        parser.add_argument('--loss_weight_gan', type=float, default=0.1, help='weight of adversarial loss')
        parser.add_argument('--loss_weight_pr_recon', type=float, default=1., help='weight of reconstruction loss for netPR(post refine)')
        parser.add_argument('--loss_weight_pr_reg', type=float, default=1e-2, help='weight of regularization loss for netPR(post refine)')
        parser.add_argument('--loss_weight_pr_vis', type=float, default=1e-2, help='weight of visibility loss for netPR(post refine)')
    
    def auto_set(self):
        super(TrainFlow3dOptions, self).auto_set()
        opt = self.opt
        if opt.pretrain_id != '' and (not opt.pretrain_id.startswith('Flow3d_')):
            opt.pretrain_id = 'Flow3d_' + opt.pretrain_id

        
class TestFlow3dOptions(BaseFlow3dOptions):
    def initialize(self):
        super(TestFlow3dOptions, self).initialize()
        self.is_train = False
        parser = self.parser

        parser.add_argument('--visualize', action='store_true')
        parser.add_argument('--test_nvis', type=int, default=64, help='number of visualized images')
        parser.add_argument('--nbatch', type=int, default=-1, help='test minibatch number')
        
        parser.add_argument('--visualize_batch', action='store_true')
        parser.add_argument('--test_batch_nvis', type=int, default=20)

