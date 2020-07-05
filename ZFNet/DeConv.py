#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

class DeConv(nn.Module):
    def __init__(self):
        super(DeConv,self).__init__()
        self.deconv_pool5 = nn.MaxUnpool2d(kernel_size=3,stride=2,padding=0)
        self.deconv_act5 = nn.ReLU()
        self.deconv_conv5 = nn.ConvTranspose2d(256,384,kernel_size=3,stride=1,padding=1,bias=False)
        
        self.deconv_act4 = nn.ReLU()
        self.deconv_conv4 = nn.ConvTranspose2d(384,384,kernel_size=3,stride=1,padding=1,bias=False)
        
        self.deconv_act3 = nn.ReLU()
        self.deconv_conv3 = nn.ConvTranspose2d(384,256,kernel_size=3,stride=1, padding=1,bias=False)
        
        self.deconv_pool2 = nn.MaxUnpool2d(kernel_size=3,stride=2,padding=1)
        self.deconv_act2 = nn.ReLU()
        self.deconv_conv2 = nn.ConvTranspose2d(256,96,kernel_size=5,stride=2,padding=0,bias=False)
        
        self.deconv_pool1 = nn.MaxUnpool2d(kernel_size=3,stride=2,padding=1)
        self.deconv_act1 = nn.ReLU()
        self.deconv_conv1 = nn.ConvTranspose2d(96,3,kernel_size=7,stride=2,padding=1,bias=False)

    def forward(self,data,indices,layer):
        if layer < 1 or layer > 5:
            raise Exception("ZFnet -> forward_deconv(): layer value should be between [1, 5]")
        
        data = self.deconv_pool5(data,indices["maxpooling_3"],output_size=torch.Size([13, 13]))
        data = self.deconv_act5(data)
        data = self.deconv_conv5(data)
        if layer == 1:
            return data
        
        data = self.deconv_act4(data)
        data = self.deconv_conv4(data)
        
        if layer == 2:
            return data
        
        data = self.deconv_act3(data)
        data = self.deconv_conv3(data)
        
        if layer == 3:
            return data
        
        data = self.deconv_pool2(data,indices["maxpooling_2"],output_size=torch.Size([26, 26]))
        data = self.deconv_act2(data)
        data = self.deconv_conv2(data)
     
        if layer == 4:
            return data
        
        data = self.deconv_pool1(data,indices["maxpooling_1"],output_size=torch.Size([110, 110]))
        data = self.deconv_act1(data)
        data = self.deconv_conv1(data)
        
        if layer == 5:
            return data
        
    def deconv_visualize(self,network,network_layer,layer, data):
        
        self.network = network.cpu()
        self.network.load_state_dict(torch.load("./ZFNet/saved_model/ZFNet_with_accuracy=80.36.pth"))
        self.network_layer = network_layer
        self.layer = layer
        activation = {}
        def feature_hook(name):
            def hook(model, inputs, outputs):
                activation[name] = outputs
            return hook
        #List = random.choices(np.arange(len(data)),k=5)
        List = [0]
        for i in List:
            self.Image, self.labels = data[i]
            #self.Image = self.Image.unsqueeze(0)
            network_layer.register_forward_hook(feature_hook(layer))
            prediction = self.network(self.Image)
            output,_= activation[layer]
            indice = self.network.indices
            count = 0
            fig = plt.figure(figsize=(60,60))
            for j in range(1,6):
                count += 1
                ax = fig.add_subplot(4,5, count)
                ax.set_title("{}, Layer {}".format(self.categories[self.labels],j), fontsize= 30)
                plt.axis('off')
                plt.imshow(self.deforward(output,indice,j).detach().numpy()[0, 2, :])

