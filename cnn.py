#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1i

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, char_embedding_size, output_embedding_size, max_word_len=21, k=5, bias=True):
        """ Initialize CNN Layer
        @param embedding_size (embedding size to use)
        @param k (kernel size)
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(char_embedding_size,
                              output_embedding_size, k, bias=bias)
        conv_output_size = max_word_len - k + 1
        self.maxpool = nn.MaxPool1d(conv_output_size)

    def forward(self, x):
        """ Forward propagate CNN layer
        @param x (input tensor, shape (batch_size, char_embed_size, max_word_len))
        @returns x_conv_out (output tensor after convolutions, (batch_size, embed_size, 1))
        """
        # print('CNN', x.shape)
        x_conv = self.conv(x)
        x_conv_out = nn.ReLU()(x_conv)
        x_conv_out = self.maxpool(x_conv_out)
        # x_conv_out, _ = torch.max(x_conv_out, dim=-1)
        return x_conv_out  # .unsqueeze(-1)

# END YOUR CODE
