import argparse
import logging
import torch
from torchsummary import summary
from ptflops import get_model_complexity_info

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=150,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-E', '--early_stopping', dest='ES', type=int, default=20,
                        help='Stop training when the verification result has n epoches no longer improved，0 represent do not early_stop'
                             )
    parser.add_argument('-w', '--wrirte_config_path', type=str, default=r'D:\ACPA_Net\output\config.txt',
                        help='The path to write the config')
    parser.add_argument('--val_result_path', type=str, default=r'D:\ACPA_Net\output\result_val.txt',
                        help='The path to write the result')
    parser.add_argument('--train_result_path', type=str, default=r'D:\ACPA_Net\output\result_train.txt',
                        help='The path to write the result')

    parser.add_argument('-i', '--set_seed', dest='seed', type=int, default=888,
                        help='Chose wheather to set the seed')


    parser.add_argument('-L', '--Loss', dest='loss', type=str, choices=['BCE', 'W_BCE', 'CE', 'BFL', 'FL', 'DL','BCE+DL' ],
                        default='BCE+DL',
                        help='loos: FL:Focal loss  DL:Dice loss')
    #Deep_Supervision
    parser.add_argument('-D', '--Deep_Supervision', dest='DS', type=str,  default=False, choices=['Decoder','Enconder', 'DAHead', 'All' ,'Enconder+Decoder'],
                        help='Decide Weather to use Deep_Supervisionv')
    parser.add_argument('-n', '--net', type=str, default='ACPA_Net',
                        choices=['ACPA_Net'],
                        help='The name of the model')
    parser.add_argument('-d'  , '--dataset', type=str, default='crack500',choices=['water','crack500'], help='The name of the dataset')
    parser.add_argument('-A', '--dataAug', type=bool, default=False,  help='yes/no data enhancement')
    parser.add_argument('-W', '--wave', type=bool, default=False, help='yes/no wavelet transform')
    return parser.parse_args()

# os.environ['PYTHONHASHSEED'] = str(seed)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()


    device = torch.device('cuda' )
    logging.info(f'Using device {device}')

    net = torch.load(
            r'D:\ACPA_Net\checkpoints\CP_BestDice_epoch_Allnet.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.cuda()
    net.eval()

    summary(net, input_size=(3, 256, 256), batch_size=4)

    macs, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))