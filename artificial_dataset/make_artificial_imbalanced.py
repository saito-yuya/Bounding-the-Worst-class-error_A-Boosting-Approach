import torch
import numpy as np
import math
from torch.utils.data import Dataset


def randomInCircle(centerX,centerY,dim, num_sample):
    datax,datay = [],[]
    for i in range(num_sample):
        u,v = np.random.uniform(0,1,(dim,1))
        theta = 2 * math.pi * u
        rad = math.sqrt(v)
        x = rad * math.cos(theta) + centerX
        y = rad * math.sin(theta) + centerY
        datax.append(x)
        datay.append(y)
    return datax,datay

def std(l):
    mi = abs(min(l))
    l = list(map(lambda item: item+mi,l))
    ma = max(l)
    mi = min(l)
    std_l = list([round((x-mi)/(ma-mi), 2) for x in l])
    return std_l

## Artificial imbalanced dataset
class CustomDataset(Dataset):
    def __init__(self,min_size):
        self.min_size = min_size
        A,B,C = self.min_size*10,self.min_size*10,self.min_size*10
        D = self.min_size
        self.data_list = []
        self.labels = [0]*A + [1]*B +[2]*C + [3]*D
        self.max_x = 0
        self.max_y = 0

        self.x_c0,self.y_c0 = randomInCircle(0,0,2,A)
        self.x_c1,self.y_c1 = randomInCircle(0,1,2,B)
        self.x_c2,self.y_c2 = randomInCircle(1,1,2,C)
        # Imbalanced class
        self.x_c3,self.y_c3 = randomInCircle(1,0,2,D)
        
        x_l = std(self.x_c0 + self.x_c1 + self.x_c2 + self.x_c3)
        y_l = std(self.y_c0 + self.y_c1 + self.y_c2 + self.y_c3)

        self.data_list.append(list(map(list,zip(x_l,y_l))))
        self.data_list = self.data_list[0]

    def __len__(self):
        return len(list(self.data_list))
        # return 
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data_list[idx]), self.labels[idx]
