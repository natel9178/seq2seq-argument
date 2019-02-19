#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1i

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, char_embedding_size, output_embedding_size, k=5, bias=True):
        """ Initialize CNN Layer
        @param embedding_size (embedding size to use)
        @param k (kernel size)
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(char_embedding_size,
                              output_embedding_size, k, bias=bias)
        conv_output_size = char_embedding_size - k + 1
        self.maxpool = nn.MaxPool1d(conv_output_size)

    def forward(self, x):
        """ Forward propagate CNN layer
        """
        # print('CNN', x.shape)
        x_conv = self.conv(x)
        x_conv_out = nn.ReLU()(x_conv)
        # print('CNN2', x_conv_out.shape)
        x_conv_out = self.maxpool(x_conv_out)
        return x_conv_out

# END YOUR CODE
