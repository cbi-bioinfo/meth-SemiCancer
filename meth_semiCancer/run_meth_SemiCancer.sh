#!/bin/bash
train_X="../preprocessing/train_X.csv"
test_X="../preprocessing/test_X.csv"
train_Y="../preprocessing/train_Y.csv"
test_Y="../preprocessing/test_Y.csv"
unlabel_X="../preprocessing/unlabel_X.csv"
unlabel_Y="../preprocessing/unlabel_Y.csv"
confidence_threshold=0.0

python meth_SemiCancer.py $train_X $test_X $train_Y $test_Y $unlabel_X $unlabel_Y $confidence_threshold
