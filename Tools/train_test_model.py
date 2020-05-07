#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from RUN.RunManager import RunManager
from RUN.RunBuilder import RunBuilder
from Tools.Visualize import Visualize

device = torch.device("cuda")
cpu = torch.device("cpu")
Vis = Visualize()
class train_test_model:
    def __init__(self,network):
        self.network = network

    def train_model(self,model, epoch,train_set,val_set):
        self.model = model
        self.epoch = epoch
        self.train_set = train_set
        self.val_set = val_set
        
        parameters = OrderedDict(
        filename = ["{}".format(self.model)],
        learning_rate = [0.01,0.001],
        batch_size = [100,200],
        num_worker = [4],
        #shuffle = [True,False]
        )
        m = RunManager()
        for run in RunBuilder.get_runs(parameters):
            train_Loader = torch.utils.data.DataLoader(self.train_set, batch_size=run.batch_size, shuffle = True, num_workers = run.num_worker)
            val_Loader = torch.utils.data.DataLoader(self.val_set,batch_size = run.batch_size, shuffle = True, num_workers=run.num_worker)

            self.network = self.network.to(device)


            optimizer = optim.Adam(self.network.parameters(), lr = run.learning_rate)
            m.begin_run(run,self.network,val_Loader)
            for epoch in range(int(self.epoch)):
                m.begin_epoch()
                for data in train_Loader:
                    inputs, labels = data
                    inputs, labels = Variable(inputs).type(torch.FloatTensor), Variable(labels).type(torch.long)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    predict = self.network(inputs)
                    loss_fc = nn.CrossEntropyLoss()
                    train_Loss = loss_fc(predict, labels)
                    optimizer.zero_grad()
                    train_Loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    for data in val_Loader:
                        inputs, labels = data
                        inputs, labels = Variable(inputs).type(torch.FloatTensor), Variable(labels).type(torch.long)
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        predict = self.network(inputs)
                        loss_fc = nn.CrossEntropyLoss()
                        val_Loss = loss_fc(predict, labels)
                        m.track_loss(val_Loss)
                        m.track_num_correct(predict,labels)
                accuracy = m.end_epoch()
                if accuracy > 99:
                    torch.save(network.state_dict(), "./Saved_model/{}_with_accuracy={}".format(self.model,accuracy))
            m.end_run()
            m.save('result')

    def test_model(self,load_model,test_set,accuracy):
        self.load_model = load_model
        self.test_set = test_set
        self.accuracy = accuracy
        Test_Loader = torch.utils.data.DataLoader(self.test_set, batch_size = 20, shuffle = True, num_workers = 2)

        self.network = self.network.to(device)

        #network = network.to(device)
        self.network.load_state_dict(torch.load("Saved_model/{}_with_accuracy={}".format(self.load_model,self.accuracy)))
        self.network = self.network.to(device)
        for data in Test_Loader:
            inputs,labels = data
            inputs = Variable(inputs).type(torch.FloatTensor)
            inputs = inputs.to(device)
            prediction = self.network(inputs)
            predict_results = prediction.data.max(1,keepdim=True)[1]
            Vis.image_visualize(inputs,predict_results)
            break


