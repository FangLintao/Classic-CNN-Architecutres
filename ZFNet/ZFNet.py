#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as f
import torch

class ZFNet(nn.Module):
    
    def __init__(self,num_class):
        super(ZFNet, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=1)
        self.maxpooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,kernel_size=5, stride=2, padding=0)
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.maxpooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True)
        
        self.dropout_1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=self.num_class)
    
        self.feature_outputs = {}
        self.switch_indices = {}
     
    def forward(self, data):
        assert data.shape[2:] == (224,224),"the image shape should match the recommended shape on paper"
        data = self.conv1(data)
        self.feature_outputs["conv1"] = data
        data = f.relu(data)
        self.feature_outputs["relu1"] = data
        data,indices = self.maxpooling_1(data)
        self.feature_outputs["maxpooling_1"] = data
        self.switch_indices["maxpooling_1"] = indices
        
        data = self.conv2(data)
        self.feature_outputs["conv2"] = data
        data = f.relu(data)
        self.feature_outputs["relu2"] = data
        data,indices = self.maxpooling_2(data)
        self.feature_outputs["maxpooling_2"] = data
        self.switch_indices["maxpooling_2"] = indices
        
        data = self.conv3(data)
        self.feature_outputs["conv3"] = data
        data = f.relu(data)
        self.feature_outputs["relu3"] = data
        data = self.conv4(data)
        self.feature_outputs["conv4"] = data
        data = f.relu(data)
        self.feature_outputs["relu4"] = data
        data = self.conv5(data)
        self.feature_outputs["conv5"] = data
        data = f.relu(data)
        self.feature_outputs["relu5"] = data
        data,indices = self.maxpooling_3(data)
        self.feature_outputs["maxpooling_3"] = data
        self.switch_indices["maxpooling_3"] = indices
        
        data = data.flatten(1)
        data = self.dropout_1(data)
        data = f.relu(self.fc1(data))
        data = self.dropout_2(data)
        data = f.relu(self.fc2(data))
        data = self.fc3(data)
        return data