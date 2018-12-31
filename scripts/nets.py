import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


# 文を文ベクトルに変換
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
        x_len = [len(x) for x in xs]  #
        x_section = np.cumsum(x_len[:-1])
        ex = self.embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return exs


# BiLSTMで文ベクトルを文書ベクトルに変換
class DocRepRNN(chainer.Chain):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, dropout=0.5):
        super(DocRepRNN, self).__init__()
        with self.init_scope():
            self.sen_enc = SentRepRNN(n_vocab, n_units, n_layers, dropout)
            self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, dropout)

    def __call__(self, x):
        # バッチ内の文書ごとに、各文をembedding (並列化するには???)
        sent_rep = [self.sen_enc(doc) for doc in x]
        # 1文ずつBiLSTMに読み込む
        last_h, last_c, ys = self.encoder(None, None, sent_rep)
        # 最終層の各文の状態を平均したものを返す
        return [F.average(x, axis=0) for x in ys]


# 与えられた文書の分類を行う 
class DocClassify(chainer.Chain):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, n_out=4, dropout=0.5):
        super(DocClassify, self).__init__()
        with self.init_scope():
            self.doc_enc = DocRepRNN(n_vocab, n_units, n_layers, dropout)
            self.out = L.Linear(None, n_out)
            self.bn = L.BatchNormalization(n_units*2)

    def __call__(self, x):
        # バッチ内の文書ごとに、各文をembedding
        sent_rep = self.doc_enc(x)
        sent_rep = F.concat([F.expand_dims(x, 0) for x in sent_rep], axis=0)
        sent_rep = self.bn(sent_rep)
        # 出力層を噛ませる
        return self.out(sent_rep)
