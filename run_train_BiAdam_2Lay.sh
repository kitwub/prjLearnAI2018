#!/bin/bash

CHAINER_TYPE_CHECK=0
export CHAINER_TYPE_CHECK

cd /mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/

{ { python3 -B scripts/train_B.py \
    --train_file data/train_w_NoReplace.csv \
    --test_file data/test_w_NoReplace.csv \
    --gpu 3 \
    --case bi_adam_2layer \
    --opt adam \
    --layer 2 \
    | tee result/bi2_adam_2layer/stdout.log; } 3>&2 2>&1 1>&3 \
    | tee result/bi2_adam_2layer/stderr.log; } 3>&2 2>&1 1>&3
