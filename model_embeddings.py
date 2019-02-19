#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

import torch
from cnn import CNN
from highway import Highway

# End "do not change"


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        # A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(
        #     len(vocab.src), embed_size, padding_idx=pad_token_idx)
        # End A4 code

        # YOUR CODE HERE for part 1j
        self.char_embed_size = 50
        self.embed_size = embed_size
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(
            len(vocab.char2id), self.char_embed_size, padding_idx=pad_token_idx)
        # print(embed_size)
        self.convolution = CNN(self.char_embed_size, self.embed_size)
        self.highway_layer = Highway(self.embed_size, dropout_rate=0.3)
        # END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        # A4 code
        # output = self.embeddings(input)
        # return output
        # End A4 code

        # YOUR CODE HERE for part 1j
        # print(input.shape)
        # (sentence_length, batch_size, max_word_length, char_embed_size)
        # print('input', input.shape)
        x_emb = self.embeddings(input)
        sentence_length, batch_size, max_word_length, char_embed_size = list(
            x_emb.size())
        # print(list(x_emb.size()))
        # (N,C,L), N should be all the words individually. C should be char_embed_size, L should be max_word_length
        x_reshape = x_emb.view(sentence_length*batch_size,
                               char_embed_size, max_word_length)
        # print('x_reshape', x_reshape.shape)
        x_conv_out = self.convolution(x_reshape)
        # print('x_conv_out', x_conv_out.size())
        x_word_emb = self.highway_layer(x_conv_out.squeeze(dim=2))
        # print('highway_layer', x_word_emb.size())
        x_word_emb = x_word_emb.view(
            sentence_length, batch_size, self.embed_size)
        # print('x_word_emb', x_word_emb.size())
        # print('-'*100)
        return x_word_emb
        # END YOUR CODE
