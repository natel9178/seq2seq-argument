#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1h

import torch
from torch import nn


class Highway(nn.Module):
    def __init__(self, num_features, dropout_rate=0.0, bias=True):
        """ Initialize Highway Layer
        @param num_features (number of input/output features)
        @param dropout_rate (probability to apply to dropout)
        @param bias (set true to add additive bias)
        """
        super(Highway, self).__init__()
        self.projection = nn.Linear(num_features, num_features, bias=bias)
        self.gate = nn.Linear(num_features, num_features, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """ Forward propagate highway layer
        @param x_convout (input tensor)
        @returns tensor after applying highway layer with shape (num_batches, num_features)
        """
        # print('HW', x.shape)
        x_proj = self.projection(x)
        x_proj = nn.ReLU()(x_proj)
        x_gate = self.gate(x)
        x_gate = nn.Sigmoid()(x_gate)

        x_highway = torch.mul(x_proj, x_gate) + \
            torch.mul((1-x_gate), x)

        x_wordemb = self.dropout(x_highway)
        # print('HW2', x_wordemb.shape)
        return x_wordemb

# END YOUR CODE
