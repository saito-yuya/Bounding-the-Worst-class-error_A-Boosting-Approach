import argparse
import models

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MedMNIST Training')
    parser.add_argument('--seed', type = int ,default=0,help='seed')
    parser.add_argument('--dataset', default='medmnist', help='dataset setting')
    parser.add_argument('--data_flag', default='tissuemnist', help='medmnist dataset')
    parser.add_argument('--num_in_channels', default=1, type=int, help='input channels 1 or 3 for medmnist ')
    parser.add_argument('--num_classes', default=8, type=int, help='number of classes ')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--max_epoch', default=10**3, type=int, metavar='N',
                    help='number of maximum epochs to run')
    parser.add_argument('-b', '--batch-size', default=128*4, type=int,
                    metavar='N',
                    help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
    parser.add_argument('--eps', type = float, default=5e-4,help='small margin of gamma')
    parser.add_argument('--theta', type = float, default=0.6,help='theta')
    parser.add_argument('--gpu', default="0",help='device')
    parser.add_argument('--root_log',type=str, default='tissue_log')
    parser.add_argument('--root_model', type=str, default='checkpoint')
    return parser.parse_args()