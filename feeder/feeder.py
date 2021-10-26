# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,#/media/wow/disk2/kinetics-skeleton/train_data.npy
                 label_path,#/media/wow/disk2/kinetics-skeleton/train_label.pkl
                 random_choose=False,#True
                 random_move=False,#True
                 window_size=-1,#200
                 debug=False,
                 mmap=True):
        self.debug = debug#false
        self.data_path = data_path#/media/wow/disk2/kinetics-skeleton/train_data.npy
        self.label_path = label_path#/media/wow/disk2/kinetics-skeleton/train_label.pkl
        self.random_choose = random_choose#True
        self.random_move = random_move#True
        self.window_size = window_size#200

        self.load_data(mmap)#True

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:#true
            self.data = np.load(self.data_path, mmap_mode='r')
            #print('训练',len(self.data)) 24036
            #print('测试',len(self.data)) 19796
            #print('第1ge',self.data[1].shape)#(3, 300, 18, 2)
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:#false
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        #print('shape',self.N, self.C, self.T, self.V, self.M)
    #N代表视频的数量，通常一个 batch 有 256 个视频 其实随便设置，最好是 2 的指数
    #C代表关节的特征，通常一个关节包含x,y,acc3个特征
    #T代表关键帧的数量，一般一个视频有 150 帧
    #V代表关节的数量，通常一个人标注 18 个关节
    #M代表一帧中的人数，一般选择平均置信度最高的 2 个人
    #所以，OpenPose 的输出，也就是 ST-GCN 的输入，形状为 (256,3,150,18,2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        #print('index_now',index)取index这个视频
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        if self.random_choose:#True
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:#True
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label