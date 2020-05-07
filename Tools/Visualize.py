#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
cpu = torch.device("cpu")

class Visualize:
    def __init__(self):
        pass
    def image_visualize(self,image,prediction):
        self.image = image
        self.prediction = prediction
        batch_size = self.image.shape[0]
        row = int(np.ceil(batch_size/5))
        fig = plt.figure(figsize=(32, 32))
        for i in range(batch_size):
            ax = fig.add_subplot(row,5,i+1)
            Image = self.image[i][0].squeeze()
            ax.imshow(Image.to(cpu))
            ax.axis('off')
            ax.set_title("Prediction={}".format(self.prediction[i]))

    def feature_visualize(self,network,network_layer,layer,data):
        self.network = network.to(cpu)
        self.data = data.to(cpu)
        self.network_layer = network_layer
        self.layer = layer
        activation = {}
        def feature_hook(name):
            def hook(model, inputs, outputs):
                activation[name] = outputs.detach()
            return hook
        network_layer.register_forward_hook(feature_hook(self.layer))
        prediction = self.network(self.data)
        act = activation[self.layer].squeeze()

        i = 0
        fig = plt.figure(figsize=(32, 32))
        for idx in range(act.size(0)):
            for channel in range(act.size(1)):
                ax = fig.add_subplot(act.size(0),act.size(1),i+1)
                ax.imshow(act[idx][channel])
                ax.axis('off')
                ax.set_title("shape={}".format(act[idx][channel].shape))
                i+=1





