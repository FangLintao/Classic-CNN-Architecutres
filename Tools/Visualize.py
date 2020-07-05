#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
cpu = torch.device("cpu")

class Visualize:
    def __init__(self,categories=None):
        self.categories = categories
    def image_visualize(self,image,prediction):
        self.image = image
        self.prediction = prediction
        batch_size = self.image.shape[0]
        row = int(np.ceil(batch_size/5))
        fig = plt.figure(figsize=(32, 32))
        
        for i in range(batch_size):
            ax = fig.add_subplot(row,5,i+1)
            if self.image[i].shape[0] == 3:
                img_rgb = np.transpose(self.image[i].to(cpu).numpy(), (1,2,0))
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb, interpolation='nearest')
            else:
                Image = self.image[i][0].squeeze()
                ax.imshow(Image.to(cpu), interpolation='nearest')
            ax.axis('off')
            if self.categories !=None:
                ax.set_title("Prediction -> {}".format(self.categories[self.prediction[i]]),fontsize=20)

    def feature_visualize(self,network,network_layer,layer,data):
        self.network = network.to(cpu)
        self.Image, self.labels = data
        self.Image = self.Image.to(cpu)
        self.network_layer = network_layer
        self.layer = layer
        activation = {}
        def feature_hook(name):
            def hook(model, inputs, outputs):
                activation[name] = outputs[0].detach()
            return hook
        network_layer.register_forward_hook(feature_hook(self.layer))
        prediction = self.network(self.Image)
        act = activation[self.layer].squeeze()
        i = 0
        fig = plt.figure(figsize=(self.Image.size(2), self.Image.size(3)))
        for idx in range(act.size(0)):
            if act.size(1) <= 3:
                for channel in range(act.size(1)):
                    ax = fig.add_subplot(act.size(0),act.size(1),i+1)
                    ax.axis('off')
                    if self.categories !=None:
                        ax.set_title("name -> {}".format(self.categories[self.labels[idx]]),fontsize=150)
                    ax.imshow(act[idx][channel],interpolation='none')
                    i+=1
            else:
                for channel in range(5):
                    ax = fig.add_subplot(act.size(0),5,i+1)
                    ax.axis('off')
                    if self.categories !=None:
                        ax.set_title("name -> {}".format(self.categories[self.labels[idx]]),fontsize=150)
                    ax.imshow(act[idx][channel], interpolation='none')
                    i+=1
