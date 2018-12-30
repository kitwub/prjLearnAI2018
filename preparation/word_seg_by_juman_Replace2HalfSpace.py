#!/usr/bin/env python3

import pandas as pd
from pyknp import Juman
import re


def seg2word(seg):
    len_split = 1000
    seg = seg.replace('\u3000', ' ')
    seg = re.sub('[ ]{2,}', ' ', seg)
    len_seg = len(seg)
    seg_splits = [seg[i:i+len_split] for i in range(0, len_seg, len_split)]

    juman_def = Juman(command="/mnt/gold/users/s18153/bin/jumanpp")
    return ' '.join([" ".join([mrph.midasi for mrph in juman_def.analysis(seg_part).mrph_list()]) for seg_part in seg_splits])


# print('cp:start - df of train_out')
# df_train = pd.read_csv('/mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/train.csv', names=('ctg', 'str'), header=None)
# seg_lst_train = df_train['str'].values.tolist()
# word_lst_train = [seg2word(s) for s in seg_lst_train]
# df_train_words = pd.Series(word_lst_train, name='words')
# df_train_out = pd.concat([df_train['ctg'], df_train_words], axis=1)
# df_train_out.to_csv('/mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/train_w_HalfSpace.csv', header=False, index=False)
# print('cp:end - df of train_out')

print('cp:start - df of test_out')
df_test = pd.read_csv('/mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/test.csv',
                      names=('ctg', 'str'),
                      header=None,
                      na_filter=False)
seg_lst_test = df_test['str'].values.tolist()
word_lst_test = [seg2word(s) for s in seg_lst_test]
df_test_words = pd.Series(word_lst_test, name='words')
df_test_out = pd.concat([df_test['ctg'], df_test_words], axis=1)
df_test_out.to_csv('/mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/test_w_HalfSpace.csv', header=False, index=False)
print('cp:end - df of test_out')

# exit()
