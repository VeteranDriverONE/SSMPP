import argparse as arg

parser = arg.ArgumentParser()

# data organization parameters
parser.add_argument('--train_dir',default="" ,help = 'training files')
parser.add_argument('--root_path',default='MS_root',help="model output directory")
parser.add_argument('--model_dir',default='MS_checkpoints',help="model output directory")
parser.add_argument('--load_model',default=None,help="optional model file to initialize with")
parser.add_argument('--train_tb',default='MS_tb')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size (default: 8)')
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled batch size (default: 4)')
parser.add_argument('--semi_partition', type=float, default=0.3, help='batch size (default: 1.0)')
parser.add_argument('--epoch',default=6000, type= int,help='number of training epoch')
parser.add_argument('--max_iter',default=60000, type= int,help='number of training epoch')
parser.add_argument('--img_shape',default=(256, 256))
# parser.add_argument('--dim',default=3)
parser.add_argument('--in_channel',default=1)
parser.add_argument('--out_channel',default=2)
parser.add_argument('--save_per_epoch', type=int, default=20, help='frequency of model saves (default: 100)')
parser.add_argument('--tb_save_freq', type=int, default=60, help='frequency of tensorboard saves (default: 100)')
parser.add_argument('--cudnn-nondet', action='store_true', help='disable cudnn determinism - might slow down training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--adam', type=bool, default=True, help='Whether to use adam (default is rmsprop)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

# test
parser.add_argument('--test_dir',default="" ,help = 'training files')
parser.add_argument('--test_load_model', default=None, help="optional model file to initialize with")
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--test_jpg_outpath', default='test_img')

args = parser.parse_args()