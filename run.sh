#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./data/train.src --train-tgt=./data/train.tgt \
        --dev-src=./data/valid.src --dev-tgt=./data/valid.tgt --vocab=vocab.json --cuda
elif [ "$1" = "test" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./data/test.src ./data/test.tgt outputs/test_outputs.txt --cuda
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./data/train.src --train-tgt=./data/train.tgt --freq-cutoff=10 vocab.json
elif [ "$1" = "dump" ]; then
    mkdir -p data
    python3 dump_pickle.py -train pickles/train.pkl -valid pickles/val.pkl -train_save data/train -valid_save data/valid -test pickles/test.pkl -test_save data/test
else
	echo "Invalid Option Selected"
fi
