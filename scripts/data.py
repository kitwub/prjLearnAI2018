import sys
import copy
import chainer
import numpy as np
from collections import defaultdict
UNK = 0

class DocDataset(chainer.dataset.DatasetMixin):
    def __init__(self, fname, vocab=None, vocab_size=50000):
        self._fname = fname
        self._vocab = vocab
        self._vocab_size = vocab_size
        self._data = []
        self.read_data()
                    
    def __len__(self):
        return len(self._data)
                    
    def get_example(self, i):
        return self._data[i][0], self._data[i][1]

    def get_vocab(self):
        return self._vocab

    def read_data(self):
        # 語彙リストが与えられていない場合は作る
        if self._vocab is None:
            self._vocab = defaultdict(int)
            # まずは全ての単語を読み込んで頻度を計数
            with open(self._fname, 'r', encoding='UTF-8') as f:
                for line in f:
                    (label, text) = line.strip().split(',', 1)
                    for w in text.split(' '):
                        self._vocab[w] += 1
            # 頻度の高い順にvocab_sizeまで語彙として登録
            index = 0
            for k, v in sorted(self._vocab.items(), key=lambda x: -x[1]):
                index += 1
                if index <= self._vocab_size:
                    self._vocab[k] = index
                else:
                    del(self._vocab[k])

        # 単語をIDに変換
        with open(self._fname, 'r', encoding='UTF-8') as f:
            for line in f:
                (label, text) = line.strip().split(',', 1)
                text_index = [[]]
                for w in text.split(' '):
                    text_index[-1].append(self._vocab.get(w, UNK))
                    if w == "。":
                        text_index.append([])
                if len(text_index[-1]) == 0:
                    text_index.pop()
                self._data.append([copy.deepcopy(text_index), np.int32(label)])
        for i in range(len(self._data)):
            self._data[i][0] = [np.array(x, dtype=np.int32) for x in self._data[i][0]]

