#!/bin/bash
train_X="../preprocessing/train_X"
test_X="../preprocessing/test_X"
train_Y="../preprocessing/train_Y"
test_Y="../preprocessing/test_Y"
unlabel_X="../preprocessing/unlabel_X"
unlabel_Y="../preprocessing/unlabel_Y"
confidence_threshold=0.0

python meth_SemiCancer.py $train_X $test_X $train_Y $test_Y $unlabel_X $unlabel_Y $confidence_threshold