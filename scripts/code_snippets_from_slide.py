import numpy as np
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


# 文表現を作る層
class SentRepRNN(chainer.Chain):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, dropout=0.5):
        super(SentRepRNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)

    def __call__(self, x):
        # 単語をembedding
        emb = self.sequence_embed(x)

        # 単語列を文ベクトルに変換
        last_h, last_c, ys = self.encoder(None, None, emb)

        # 最終層のhidden stateを返す
        return last_h[-1]

    # 文を効率的に(一気に)embeddingするための関数
    def sequence_embed(self, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = self.embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return exs


# 文書表現を作る層
class DocRepRNN(chainer.Chain):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, dropout=0.5):
        super(DocRepRNN, self).__init__()
        with self.init_scope():
            self.sen_enc = SentRepRNN(n_vocab, n_units, n_layers, dropout)
            self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, dropout)

    def __call__(self, x):
        # バッチ内の文書ごとにembedding (並列化するには？？？)
        sent_rep = [self.sen_enc(doc) for doc in x]

        # 1文ずつBiLSTMに読み込む
        last_h, last_c, ys = self.encoder(None, None, sent_rep)

        # 最終層の各分の状態を平均化したものを返す
        return [F.average(x, axis=0) for x in ys]


# Output Layer
class DocClassify(chainer.Chain):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, n_out=4, dropout=0.5):
        super(DocClassify, self).__init__()
        with self.init_scope():
            self.doc_enc = DocRepRNN(n_vocab, n_units, n_layers, dropout)
            self.out = L.Linear(None, n_out)

    def __call__(self, x):
        # バッチ内の文書ごとに、各分をembedding
        sent_rep = self.doc_enc(x)
        # 出力層をかませる
        return self.out(F.concat([F.expand_dims(x, 0) for x in sent_rep], axis=0))
