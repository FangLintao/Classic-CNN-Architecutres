#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict
import torch

class Load_Model:
    def __init__(self):
        pass
    def load_model(self,pretrained_model_file, network, deconv_network):
        """
        Loads a pretrained model parameters into our network and deconvolution network.
        where target model parameters names are different from the pretrained one.

        Inputs:
        pretrained_model_file: pretrained model parametees saved in .pth file.
        network: model to load parameters.
        deconv_network: decovolution model to load parameters.
        """
        pretrained_model = torch.load(pretrained_model_file)
        new_state_dict_1 = OrderedDict()
        model_key = list(network.state_dict().keys())

        count = 0
        for key, value in pretrained_model.items():
            new_key = model_key[count]
            new_state_dict_1[new_key] = value
            count += 1

        deconv_key = list(deconv_network.state_dict().keys())
        mapping = {}
        for idx,item in enumerate(deconv_key):
            mapping[model_key[2*idx]] = deconv_key[-(idx+1)]
        new_state_dict_2 = OrderedDict()
        # Load Deconv part
        for key, value in new_state_dict_1.items():
            if key in mapping:
                new_state_dict_2[mapping[key]] = value
        deconv_network.load_state_dict({**new_state_dict_2})
        network.load_state_dict({**new_state_dict_1})