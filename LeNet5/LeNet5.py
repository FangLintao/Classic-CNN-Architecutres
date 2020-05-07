#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch.nn.functional as f

class LeNet5(nn.Module):
    """
    LeNet5 neural network model is mainly used in hand-written recognition, it is capable of recognizing numbers
    from 0-9 with different affine in their shapes.
    This model mainly refers to the paper" Gradient-Based Learning Applied to Document Recognition " published at 1998,
    """
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.batchnorm1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16, kernel_size=5)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    def forward(self, data):
        assert data.shape[2:] == (32,32),        "input images' shape should match the requirement on the input size recommended on papare "" Gradient-Based Learning Applied to Document Recognition"" "
        data = self.max_pool_1(f.relu(self.batchnorm1(self.conv1(data))))
        data = self.max_pool_2(f.relu(self.batchnorm2(self.conv2(data))))
        data = data.view(-1,16*5*5)
        data = f.relu(self.fc1(data))
        data = f.relu(self.fc2(data))
        data = self.fc3(data)
        return data

