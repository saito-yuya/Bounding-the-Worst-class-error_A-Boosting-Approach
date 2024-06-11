import argparse
import models

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--seed', default=0, type=int,help='seed for initializing training. ')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32')
# parser.add_argument('--num_in_channels', default=3, type=int, help='input channels 1 or 3 for medmnist ')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes ')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=1.0, type=float, help='imbalance factor')
parser.add_argument('--start_ib_epoch', default=100, type=int, help='start epoch for IB Loss')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128*4, type=int,metavar='N',help='batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,metavar='LR', help='fixed learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,metavar='W', help='weight decay (default: 1e-4)',dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=100, type=int,metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',help='use pre-trained model')
parser.add_argument('--gpu', default=0, type=int,help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='cifar_log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--early_stop', type=str, default=True)
parser.add_argument('--stop_mode', type=str, default='average')