#!/bin/bash

# -B option: Don't create __pycache__ directory (for original import package)

python3 -B /mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/train.py \
	--gpu 5 \
	--train_file /mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/train_w_NoReplace.csv \
	--test_file  /mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/test_w_NoReplace.csv
