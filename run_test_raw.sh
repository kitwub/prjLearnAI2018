#!/bin/bash

python3 -B /mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/scripts/test.py \
	--gpu 6 \
	--train_file /mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/train_w_NoReplace.csv \
	--test_file  /mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/test_w_NoReplace.csv \
	--model  /mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/sample_result_raw/snapshot_epoch-30