import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix
from opts_medmnist import parser


args = parser.parse_args()
if args.dataset == 'cifar100':
    num_classes = 100
elif args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'tinyimagenet':
    num_classes = 200
elif args.dataset == 'emnist':
    num_classes = 62
elif args.dataset == 'medmnist':
    if args.data_flag == 'pathmnist':
        num_classes = 9
    elif args.data_flag == 'dermamnist':
        num_classes = 7
    elif args.data_flag == 'octmnist':
        num_classes = 4
    elif args.data_flag == 'bloodmnist':
        num_classes = 8
    elif args.data_flag == 'tissuemnist':
        num_classes = 8
    elif args.data_flag == 'organamnist':
        num_classes = 11
    elif args.data_flag == 'organcmnist':
        num_classes = 11
    elif args.data_flag == 'organsmnist':
        num_classes = 11
    else:
        print("choose dataset on Medmnist")
else: 
    warnings.warn('Dataset is not listed')

def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()

class IBLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)),1) # N * 1
        ib = grads*features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)

def ib_focal_loss(input_values, ib, gamma):
    """Computes the ib focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * ib
    return loss.mean()

class IB_FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000., gamma=0.):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)),1) # N * 1
        ib = grads*(features.reshape(-1))
        ib = self.alpha / (ib + self.epsilon)
        return ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    #loss = (1 - p) ** gamma * input_values
    loss = (1- p) ** gamma * input_values * 10
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class WorstLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        loss = self.cross_entropy_loss(input, target)

        input_np = input.argmax(1).cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()

        cm = confusion_matrix(
            y_true=target_np, y_pred=input_np, normalize='true')
        acc_list = np.array([cm[c][c] for c in range(cm.shape[0])])
        worst_acc = np.min(acc_list)
        worst_class = np.where(acc_list == worst_acc)[0]

        if len(worst_class) == 1:
            mask = (target == worst_class[0])
        else:
            mask = (target == worst_class[0])
            for c in worst_class[1:]:
                mask += (target == c)

        loss[mask != 1] *= 0
        loss = loss.mean()
        return loss