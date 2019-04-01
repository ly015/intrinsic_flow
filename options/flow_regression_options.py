from base_options import BaseOptions

class BaseFlowRegressionOptions(BaseOptions):
    def initialize(self):
        super(BaseFlowRegressionOptions, self).initialize()
        parser = self.parser
        ##############################
        # General Setting
        ##############################

        ##############################
        # Model Setting
        ##############################
        parser.add_argument('--which_model', type=str, default='unet_v2', choices=['unet', 'unet_v2'], help='model architecture')
        parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'network initialization method [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--norm', type=str, default='batch', choices=['batch', 'instance', 'none'])
        parser.add_argument('--nf', type=int, default=32, help='feature channel number of first conv layer')
        parser.add_argument('--max_nf', type=int, default=128, help='max feature number')
        parser.add_argument('--input_type1', type=str, default='joint', help='model input for image 1')
        parser.add_argument('--input_type2', type=str, default='joint', help='model input for image 2')
        parser.add_argument('--start_scale', type=int, default=2, help='scale of the top-scale flow compared to original image')
        parser.add_argument('--num_scale', type=int, default=5, help='scale number of image/flow pyramid')
        parser.add_argument('--flow_loss_type', type=str, default='l2', choices=['l1', 'l2'], help='type of flow loss')
        # for segmentation sensitive flow loss
        parser.add_argument('--use_ss_flow_loss', type=int, default=1, choices=[0,1])
        ##############################
        # Pose Setting
        ##############################
        parser.add_argument('--joint_nc', type=int, default=18, help='2d joint number. 18 for openpose joint')
        parser.add_argument('--joint_mode', type=str, default='binary', choices=['binary', 'gaussian'])
        parser.add_argument('--joint_radius', type=int, default=8, help='radius of joint map')
        # parser.add_argument('--seg_nc', type=int, default=20, help='number of segmentation classes, 7 for ATR and 8 for LIP, 20 for full LIP')
        parser.add_argument('--seg_nc', type=int, default=7, help='number of segmentation classes, 7 for SMPL projection')
        parser.add_argument('--seg_bin_size', type=int, default=1, help='bin size of downsampled seg mask')
        ##############################
        # data setting
        ##############################
        parser.add_argument('--dataset_type', type=str, default='flow', help='type of dataset. see data/data_loader.py')
        parser.add_argument('--dataset_name', type=str, default='deepfashion', choices=['deepfashion', 'market'])
        parser.add_argument('--image_size', type=int, nargs='+', default=[256,256], help='image size (H,W)')
        parser.add_argument('--batch_size', type = int, default = 8, help = 'batch size')
        parser.add_argument('--data_root', type=str, default=None, help='Set in auto_set()')
        parser.add_argument('--fn_split', type=str, default=None, help='Set in auto_set()')
        parser.add_argument('--img_dir', type=str, default=None, help='Set in auto_set()')
        parser.add_argument('--seg_dir', type=str, default=None, help='Set in auto_set()')
        parser.add_argument('--corr_dir', type=str, default=None, help='Set in auto_set()')
        parser.add_argument('--fn_pose', type=str, default=None, help='Set in auto_set()')
        parser.add_argument('--debug', action='store_true', help='debug')
        # augmentation setting
        parser.add_argument('--aug_shiftx_range', type=int, default=20)
        parser.add_argument('--aug_shifty_range', type=int, default=10)
        parser.add_argument('--vis_smooth_rate', type=int, default=5, help='use a median filter of size # to smooth the visibility map')

    def auto_set(self):
        super(BaseFlowRegressionOptions, self).auto_set()
        opt = self.opt
        ###########################################
        # Add id profix
        ###########################################
        if not opt.id.startswith('FlowReg_'):
            opt.id = 'FlowReg_' + opt.id
        ###########################################
        # Dataset Settings
        ###########################################
        if opt.dataset_name == 'market':
            opt.image_size = [128,64]
            opt.num_scale = 4
            opt.joint_radius = 4
            opt.aug_shiftx_range = 5
            opt.aug_shifty_range = 5
            opt.data_root = 'datasets/market1501/'
            opt.img_dir = 'Images/img/'
            opt.seg_dir = 'Images/silhouette6/'
            opt.fn_pose = 'Label/pose_label.pkl'
            opt.fn_split = 'Label/pair_split.json'
            opt.corr_dir = '3d/hmr/corr/'
        else:
            opt.image_size = [256,256]
            opt.data_root = 'datasets/DF_Pose/'
            opt.img_dir = 'Img/img/'
            opt.seg_dir = 'Img/silhouette6/'
            opt.fn_pose = 'Label/pose_label_hmr_adapt.pkl'
            opt.fn_split = 'Label/pair_split_dfm_new_clean.json'
            opt.corr_dir = '3d/hmr_dfm_v2/corr/'
                    

class TrainFlowRegressionOptions(BaseFlowRegressionOptions):
    def initialize(self):
        super(TrainFlowRegressionOptions, self).initialize()
        self.is_train = True
        parser =self.parser
        # basic
        parser.add_argument('--resume_train', action = 'store_true', default = False, help = 'coninue training from saved model')
        parser.add_argument('--last_epoch', type=int, default=1, help='which epoch to resume training from')
        # pretrain
        parser.add_argument('--pretrain_id', type=str, default='', help='id of pretrain model')
        parser.add_argument('--pretrain_epoch', type=str, default='latest', help='specify which epoch of the pretrain model to be used')
        # optimizer
        parser.add_argument('--lr', type = float, default = 1e-4, help = 'initial learning rate')
        parser.add_argument('--beta1', type = float, default = 0.9, help = 'momentum1 term for Adam')
        parser.add_argument('--beta2', type = float, default = 0.999, help = 'momentum2 term for Adam')
        parser.add_argument('--weight_decay', type=float, default=4e-4, help='weight decay')
        parser.add_argument('--lr_D', type=float, default=1e-5, help='initial learning rate for discriminator')
        # scheduler
        parser.add_argument('--lr_policy', type=str, default='step', choices = ['step', 'plateau', 'lambda'], help='learning rate policy: lambda|step|plateau')
        # parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--niter', type = int, default=30, help = '# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay', type=int, default=8, help='multiply by a gamma every lr_decay_interval epochs')
        parser.add_argument('--lr_gamma', type = float, default = 0.5, help='lr decay rate')
        parser.add_argument('--display_freq', type=int, default=10, help='every # iteration display loss information')
        parser.add_argument('--test_epoch_freq', type = int, default = 1, help='every # epoch test model')
        parser.add_argument('--save_epoch_freq', type = int, default = 5, help='every # epoch save current model weights' )
        parser.add_argument('--vis_epoch_freq', type = int, default = 1, help='every # epoch visualize result in exp_dir/vis/')
        parser.add_argument('--check_grad_freq', type = int, default = 100, help = 'every # iteration check each loss gradients')
        parser.add_argument('--nvis', type = int, default = 64, help='number of visulazition samples')
        # loss setting
        parser.add_argument('--loss_weight_flow', type=float, default=1., help='weight of multiscale flow loss. see models.modules.MultiScaleFlowLoss')
        parser.add_argument('--loss_weight_vis', type=float, default=1., help='weight of visibility loss')
        parser.add_argument('--loss_weight_flow_ss', type=float, default=5., help='weight of segmentation-sensitive flow loss. see models.modules.SS_FlowLoss')

        
class TestFlowRegressionOptions(BaseFlowRegressionOptions):
    def initialize(self):
        super(TestFlowRegressionOptions, self).initialize()
        self.is_train = False
        parser = self.parser
        parser.add_argument('--which_epoch', type=int, default='best', help='which epoch to load stored model weights')

