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
from tqdm import tqdm
import os

from RUN.RunManager import RunManager
from RUN.RunBuilder import RunBuilder
from Tools.Visualize import Visualize

device = torch.device("cuda")
cpu = torch.device("cpu")
class train_test_model:
    def __init__(self,network,model_dir="saved_models"):
        self.network = network
        self.model_dir = os.path.join("./"+self.network+"/"+model_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def train_model(self,model, epoch,train_set,val_set,lr,batch_size,num_worker,saved_accuracy):
        """
        training models
        Inputs:
            model(string): the training model name
            epoch(scalar): epoch number
            train_set: training data
            val_set: validation data
            lr(list): learning rates, e.g. [0.001,0.002...]
            batch_size(list): batch size, e.g. [100,200,...]
            num_worker(list): number works, e.g. [0,1,...]
            saved_accuracy(scalar): when the accuracy above certain value, we save trained model parameters
        """
        self.model = model.cuda()
        self.epoch = epoch
        self.train_set = train_set
        self.val_set = val_set
        self.lr = lr
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.saved_accuracy = saved_accuracy
        
        parameters = OrderedDict(
        filename = ["{}".format(self.network)],
        learning_rate = self.lr,
        batch_size = self.batch_size,
        num_worker = self.num_worker,
        )
        self.Accuracy = []
        m = RunManager()
        for run in RunBuilder.get_runs(parameters):
            train_Loader = torch.utils.data.DataLoader(self.train_set, batch_size=run.batch_size, shuffle = True, num_workers = run.num_worker)
            val_Loader = torch.utils.data.DataLoader(self.val_set,batch_size = run.batch_size, shuffle = True, num_workers=run.num_worker)
            optimizer = optim.Adam(self.model.parameters(), lr = run.learning_rate)
            m.begin_run(run,self.model,val_Loader)
            for epoch in tqdm(range(int(self.epoch)),ascii=True,desc="epoch"):
                m.begin_epoch()
                for data in train_Loader:
                    inputs, labels = data
                    inputs, labels = Variable(inputs).type(torch.FloatTensor), Variable(labels).type(torch.long)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    predict = self.model(inputs)
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
                        predict = self.model(inputs)
                        loss_fc = nn.CrossEntropyLoss()
                        val_Loss = loss_fc(predict, labels)
                        m.track_loss(val_Loss)
                        m.track_num_correct(predict,labels)
                accuracy = m.end_epoch()
                self.Accuracy.append(accuracy)
                if accuracy > self.saved_accuracy:
                    torch.save(self.model.state_dict(), self.model_dir+"/{}_with_accuracy={}.pth".format(self.network,accuracy))
            m.end_run()
            m.save('result')

    def test_model(self,load_model,test_set,accuracy, batch_size, num_workers):
        self.load_model = load_model
        self.test_set = test_set
        self.accuracy = accuracy
        self.batch_size = batch_size
        self.num_workers = num_workers
        Test_Loader = torch.utils.data.DataLoader(self.test_set, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)
        
        if self.test_set.classes:
            categories = self.test_set.classes
        Vis = Visualize(categories)
        self.network = self.network.to(device)

        #network = network.to(device)
        self.network.load_state_dict(torch.load(self.model_dir+"/{}_with_accuracy={}.pth".format(self.load_model,self.accuracy)))
        self.network = self.network.to(device)
        for data in Test_Loader:
            inputs,labels = data
            inputs = Variable(inputs).type(torch.FloatTensor)
            inputs = inputs.to(device)
            prediction = self.network(inputs)
            predict_results = prediction.data.max(1,keepdim=True)[1]
            Vis.image_visualize(inputs,predict_results)
            break


