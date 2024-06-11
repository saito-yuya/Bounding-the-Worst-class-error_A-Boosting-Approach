from collections import defaultdict
import os
import time
from os import TMP_MAX
import torch
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from itertools import chain 
import sklearn.metrics as metrics
from tqdm import tqdm
import sys
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import warnings
from losses import FocalLoss
warnings.filterwarnings('ignore')
from sklearn.utils.multiclass import unique_labels

    
class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def make_label_clsnum(loader):
    label = []
    cls_num = []
    for i in loader:
        _,labels = i
        label.extend(labels.tolist())
    n = len(set(label))
    for i in range(n):
        cls_num.append(label.count(i))
        
    return label,cls_num

def torch_fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


class Feedbuck():
    def ACC(TP,num):
        return TP/num

    def Binary_ACC(accuracy,theta):
        if (accuracy) >= theta:
            return 1
        else:
            return 0
        
feed = Feedbuck.Binary_ACC
    
def Network_init(model,path,device):
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model

def OP_init(model,train_loader,optimizer,weight_tmp,loss_type,device):
    weight_tmp = weight_tmp.to(device)
    if loss_type == 'CE':
        criterion = nn.CrossEntropyLoss(weight=weight_tmp).to(device)
    elif loss_type == 'Focal':
        criterion = FocalLoss(weight=weight_tmp, gamma=1).to(device)
    for data in train_loader:
        inputs, labels = data
        labels = labels.reshape(-1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs,_ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("OP_init Finish")

def class_wise_acc(model,loader,device):
    class_acc_list,y_preds,true_label = [],[],[]
    model = model.to(device) 
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate((loader)):
                inputs, labels = data
                labels = labels.reshape(-1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                predicted,_ = model(inputs) 
                predicted = torch.max(predicted, 1)[1]
                y_preds.extend(predicted.cpu().numpy())
                true_label.extend(labels.cpu().numpy())
        cf = confusion_matrix(true_label,y_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt !=0)
        class_acc_list.append(cls_acc)
    model.train()
    return class_acc_list[0],y_preds,true_label,cls_cnt

def calc_acc(label,pred):
    class_acc_list = []
    cf = confusion_matrix(label,pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt !=0)
    cls_acc = np.around(cls_acc ,decimals=4)
    class_acc_list.append(cls_acc.tolist())
    return class_acc_list

def class_wise_acc_h(y_pred,labels):
    ans = defaultdict(int)
    for item in zip(labels,y_pred):
        if item[0] == item[1]:
            ans[item[0]] += 1
    return ans 

def train(model,train_loader,classes,weight_tmp,optimizer,loss_type,max_epoch,theta,gamma,log,device):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    train_acc_list = []

    end = time.time()
    
    for epoch in range(max_epoch):
        running_loss = 0.0
        weight_tmp = weight_tmp.to(device)
        if loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=weight_tmp).to(device)
        elif loss_type == 'Focal':
            criterion = FocalLoss(weight=weight_tmp, gamma=1).to(device)
        for images, labels in tqdm(train_loader, leave=False):
            labels = labels.reshape(-1)
            images, labels = images.to(device), labels.to(device)
            outputs,_ = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss/len(train_loader)
        
        train_acc_list,y_preds,_,_ = class_wise_acc(model,train_loader,device)

        print("-----------------total_epoch:{}------------------".format(epoch))
        print("train_loss:{}".format(train_loss))

        weight_l = weight_tmp.tolist()

        rt = [0]*classes
        for k in range(classes):
            rt[k] = feed(train_acc_list[k],theta)

        ft = 0
        for k in range(classes):
            ft +=  weight_l[k]*rt[k]

        if ft >=  (1/2) + gamma:
            print("Satisfied with W.L Definition : {}".format(epoch))
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end)
            output = (
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss:.4f} ({loss:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_time=batch_time,loss=train_loss, top1=top1, top5=top5))
            print(output)
            log.write(output + '\n')
            end = time.time()
            break
        elif epoch == (max_epoch-1):
            print("Couldn't Satisfied with W.L Definition")
            sys.exit()

    return train_acc_list,model,y_preds,rt

def Hedge(weight_tmp,rt,classes,round_num,device):
    eta = ((8*(math.log(classes)))/(round_num))**(1/2)
    down = 0
    for i in range(classes):
        down +=  (weight_tmp[i].item()*math.exp(-(eta*rt[i])))
    for i,item in enumerate(rt):
        weight = weight_tmp[i].item()
        weight_tmp[i] = ((weight*math.exp(-(eta*item)))/down)

    weight_tmp = torch.tensor(weight_tmp)
    weight_tmp = weight_tmp.to(device)
    return weight_tmp

def transposition(matrix):
    matrix = np.array(matrix).T
    matrix = matrix.tolist()
    return matrix

def voting(ht):
    h_var = []
    # ht = treatment(ht)
    for m in range(len(ht)):
        count = Counter((ht[m]))
        majority = count.most_common()
        h_var.append(majority[0][0])
    h_var = transposition(h_var)
    return h_var

def ensemble(ht,keep):
    keep.append(ht)
    keep = transposition(keep)
    h_var = []
    for m in range(len(keep)):
        count = Counter((keep[m]))
        majority = count.most_common()
        h_var.append(majority[0][0])
    h_var = transposition(h_var)
    return h_var

def best_N(out_list,y_true):
    keep = []
    acc_list = []
    for i in range(len(out_list)):
        out = out_list[i]
        if i != 0:
            res = ensemble(out,keep)
        else:
            res = out
        acc = calc_acc(y_true,res)
        keep.append(out)
        acc_list.append(acc[0])
    return acc_list

def worst_val_idx(acc_list):
    acc_list = np.array(acc_list)
    idx = acc_list.argmin(axis=1)
    val = acc_list.min(axis=1)
    # n = val.argmax()
    n = max([i for i,x in enumerate(val) if x == max(val)])
    worst = val[n]
    return worst,n,idx

def calc_acc_ave(label,pred):
    # print(pred)
    ave_list = []
    cf = confusion_matrix(label,pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    ave = sum(cls_hit)/sum(cls_cnt)

    ave_list.append(ave.tolist())
    return ave_list

def ave(out_list,y_true):
    keep = []
    ave_list = []
    for i in range(len(out_list)):
        out = out_list[i]
        if i != 0:
            res = ensemble(out,keep)
            # print(res)
        else:
            res = out
        acc = calc_acc_ave(y_true,res)
        keep.append(out)
        ave_list.append(acc[0])
    return ave_list