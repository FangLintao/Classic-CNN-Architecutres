#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import torch.nn.functional as f

class AlexNet(nn.Module):
    def __init__(self,num_class):
        super(AlexNet,self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,stride=4)
        self.maxpooling_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.LRN_1 = nn.LocalResponseNorm(size=5,alpha=0.0001, beta=0.75, k=2)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,kernel_size=5,stride=1, padding=2)
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.LRN_2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.maxpooling_3 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.dropout_1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=self.num_class)
    def forward(self,data):
        assert data.shape[2:] == (227,227),        "the image shape should match the recommended shape on paper"
        data = f.relu(self.conv1(data))
        data = self.LRN_1(data)
        data = self.maxpooling_1(data)
        #data = self.LRN_1(data)
        data = f.relu(self.conv2(data))
        data = self.LRN_2(data)
        data = self.maxpooling_2(data)
        #data = self.LRN_2(data)
        data = f.relu(self.conv3(data))
        data = f.relu(self.conv4(data))
        data = f.relu(self.conv5(data))
        data = self.maxpooling_3(data)
        data = data.flatten(1)
        data = self.dropout_1(data)
        data = f.relu(self.fc1(data))
        data = self.dropout_2(data)
        data = f.relu(self.fc2(data))
        data = self.fc3(data)
        return data





