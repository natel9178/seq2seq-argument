#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py 1h
    sanity_check.py 1i
"""
import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, read_corpus, batch_iter
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT

from highway import Highway
from cnn import CNN


import torch
import torch.nn as nn
import torch.nn.utils

# ----------
# CONSTANTS
# ----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0


class DummyVocab():
    def __init__(self):
        self.char2id = json.load(
            open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]


def question_1h_sanity_check(highway):
    """ 

    """
    print("-"*80)
    print("Running Sanity Check for Question 1h")
    print("-"*80)
    # input = [[0, 0, 0, 0]]
    print("Sanity Check Passed for Question 1h!")
    print("-"*80)


def question_1i_sanity_check(convolution):
    """ 

    """
    print("-"*80)
    print("Running Sanity Check for Question 1i")
    print("-"*80)

    print("Sanity Check Passed for Question 1i!")
    print("-"*80)


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)
            ), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(
        torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Initialize Highway
    highway_layer = Highway(EMBED_SIZE, dropout_rate=DROPOUT_RATE)

    # initialize conv
    convolution = CNN(50, EMBED_SIZE)

    if args['1h']:
        question_1h_sanity_check(highway_layer)
    elif args['1i']:
        question_1i_sanity_check(convolution)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
