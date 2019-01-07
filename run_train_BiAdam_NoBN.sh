#!/bin/bash

CHAINER_TYPE_CHECK=0
export CHAINER_TYPE_CHECK

cd /mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/

{ { python3 -B scripts/train_C.py \
    --train_file data/train_w_NoReplace.csv \
    --test_file data/test_w_NoReplace.csv \
    --gpu 2 \
    --case bi2_adam_nobn \
    --opt  bi2_adam_nobn \
    | tee result/bi2_adam_nobn/stdout.log; } 3>&2 2>&1 1>&3 \
    | tee result/bi2_adam_nonb/stderr.log; } 3>&2 2>&1 1>&3
