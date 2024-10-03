import torch
import numpy as np
import math
from torch.utils.data import Dataset


def randomInRotatedSquare(centerX, centerY, dim, num_sample):
    datax, datay = [], []
    theta = -math.pi/4
    for i in range(num_sample):
        x = np.random.uniform(centerX - dim/2, centerX + dim/2)
        y = np.random.uniform(centerY - dim/2, centerY + dim/2)
        x_rot = (x-centerX) * math.cos(theta) - (y-centerY) * math.sin(theta) + centerX
        y_rot = (x-centerX) * math.sin(theta) + (y-centerY) * math.cos(theta) + centerY
        datax.append(x_rot)
        datay.append(y_rot)
    return datax, datay

def std(l):
    mi = abs(min(l))
    l = list(map(lambda item: item+mi,l))
    ma = max(l)
    mi = min(l)
    std_l = list([round((x-mi)/(ma-mi), 2) for x in l])
    return std_l

## Artificial balanced dataset
class CustomDataset(Dataset,):
    def __init__(self,min_size):
        self.min_size = min_size
        A,B,C,D,E = self.min_size,self.min_size,self.min_size,self.min_size,self.min_size
        self.data_list = []
        self.labels = [0]*A + [1]*B +[2]*C + [3]*D + [4]*E
        self.max_x = 0
        self.max_y = 0

        self.x_c0,self.y_c0 = randomInRotatedSquare(0,1.4,2,A)
        self.x_c1,self.y_c1 = randomInRotatedSquare(1.4,1.4,2,B)
        self.x_c2,self.y_c2 = randomInRotatedSquare(2.8,1.4,2,C)
        self.x_c3,self.y_c3 = randomInRotatedSquare(1.4,2.8,2,D)
        self.x_c4,self.y_c4 = randomInRotatedSquare(1.4,0,2,E)
        
        x_l = std(self.x_c0 + self.x_c1 + self.x_c2 + self.x_c3 + self.x_c4)
        y_l = std(self.y_c0 + self.y_c1 + self.y_c2 + self.y_c3 + self.y_c4)

        self.data_list.append(list(map(list,zip(x_l,y_l))))
        self.data_list = self.data_list[0]

    def __len__(self):
        return len(list(self.data_list))
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data_list[idx]), self.labels[idx],idx