import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


# 文を文ベクトルに変換 # note: ミニバッチから1ラベル分の複数文書に対して処理
class SentRepRNN(chainer.Chain):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, dropout=0.5):
        super(SentRepRNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)  # word embedding（入力は単一のリスト)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout) # (embed時と異なり、)ラベル内の各文ごとに文ベクトルを作成

    def __call__(self, x):
        # 単語をembedding
        emb = self.sequence_embed(x)  # emb: 1ラベル内の各文を、分散表現ベクトルを単語数だけ並べたVariableにembeddingし、タプルとしてまとめたもの。
        # emb = self.embed(x)  # emb: 1ラベル内の各文を、分散表現ベクトルを単語数だけ並べたVariableにembeddingし、タプルとしてまとめたもの。
        # 単語列を文ベクトルに変換
        last_h, last_c, ys = self.encoder(None, None, emb)  # 1ラベルの各文を、それぞれ1つの文ベクトルに変換

        # 最終層のhidden stateを返す
        # return last_h[-1]

        # ys_list = [y[-1] for y in ys]
        # ys_val = F.concat([F.expand_dims(x, 0) for x in ys_list], axis=0)
        # return ys_val

        # return F.concat([F.expand_dims(x, 0) for x in [y[-1] for y in ys]], axis=0)
        # 順・逆方向の最終層のhidden stateを平均化して返す
        return F.concat([last_h[0], last_h[1]], axis=1)

    # 文を効率的に(一気に)embeddingするための関数
    def sequence_embed(self, xs):
        x_len = [len(x) for x in xs]   # １ラベルに含まれる各文の単語数ベクトル [3, 5, 4, 7, ...]
        x_section = np.cumsum(x_len[:-1])  # 分割位置を単語数の累計で記録 [3, 8, 12, 19, ...]
        ex = self.embed(F.concat(xs, axis=0))  # ラベル内の各文をまっすぐ結合し分散表現に変換(Word Embeddingを実行)
        exs = F.split_axis(ex, x_section, 0)  # x_selection の値をもとに再度、文ごとに分割
        return exs
        # return F.concat([self.embed(x) for x in xs], axis=0)


# 文を文ベクトルに変換 # note: ミニバッチから1ラベル分の複数文書に対して処理
class SentRepBiRNN(SentRepRNN):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, dropout=0.5):
        super(SentRepRNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)  # word embedding（入力は単一のリスト)
            self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, dropout) # (embed時と異なり、)ラベル内の各文ごとに文ベクトルを作成


# 文を文ベクトルに変換 # note: ミニバッチから1ラベル分の複数文書に対して処理
class SentRepBi2RNN(SentRepRNN):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, dropout=0.5):
        super(SentRepRNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)  # word embedding（入力は単一のリスト)
            self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, dropout) # (embed時と異なり、)ラベル内の各文ごとに文ベクトルを作成

    def __call__(self, x):
        # 単語をembedding
        emb = self.sequence_embed(x)  # emb: 1ラベル内の各文を、分散表現ベクトルを単語数だけ並べたVariableにembeddingし、タプルとしてまとめたもの。
        # 単語列を文ベクトルに変換
        last_h, last_c, ys = self.encoder(None, None, emb)  # 1ラベルの各文を、それぞれ1つの文ベクトルに変換
        # 順・逆方向の最終層のhidden stateを平均化して返す
        return F.average(last_h, axis=0)


# BiLSTMで文ベクトルを文書ベクトルに変換 # ミニバッチのまま処理
class DocRepRNN(chainer.Chain):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, dropout=0.5):
        super(DocRepRNN, self).__init__()
        with self.init_scope():
            self.sen_enc = SentRepRNN(n_vocab, n_units, n_layers, dropout)
            self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, dropout)

    def __call__(self, x):
        # バッチ内の文書ごとに、各文をembedding (並列化するには???)
        sent_rep = [self.sen_enc(doc) for doc in x]  # x: ミニバッチ, doc: １ラベルの複数文
        # sent_rep = self.sen_enc(x)
        # 1文ずつBiLSTMに読み込む
        last_h, last_c, ys = self.encoder(None, None, sent_rep)
        # 最終層の各文の状態を平均したものを返す
        # return [F.average(x, axis=0) for x in ys]
        return F.concat([last_h[0], last_h[1]], axis=1)


# BiLSTMで文ベクトルを文書ベクトルに変換 # ミニバッチのまま処理
class DocRepBiRNN(DocRepRNN):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, dropout=0.5):
        super(DocRepRNN, self).__init__()
        with self.init_scope():
            self.sen_enc = SentRepBiRNN(n_vocab, n_units, n_layers, dropout)
            self.encoder = L.NStepBiLSTM(n_layers, n_units * 2, n_units * 2, dropout)


# BiLSTMで文ベクトルを文書ベクトルに変換 # ミニバッチのまま処理
class DocRepBi2RNN(DocRepRNN):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, dropout=0.5):
        super(DocRepRNN, self).__init__()
        with self.init_scope():
            self.sen_enc = SentRepBiRNN(n_vocab, n_units, n_layers, dropout)
            self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, dropout)

    def __call__(self, x):
        # バッチ内の文書ごとに、各文をembedding (並列化するには???)
        sent_rep = [self.sen_enc(doc) for doc in x]  # x: ミニバッチ, doc: １ラベルの複数文
        # sent_rep = self.sen_enc(x)
        # 1文ずつBiLSTMに読み込む
        last_h, last_c, ys = self.encoder(None, None, sent_rep)
        # 最終層の各文の状態を平均したものを返す
        # return [F.average(x, axis=0) for x in ys]
        return F.average(last_h, axis=0)


# 与えられた文書の分類を行う
class DocClassify(chainer.Chain):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, n_out=4, dropout=0.5):
        super(DocClassify, self).__init__()
        with self.init_scope():
            self.doc_enc = DocRepRNN(n_vocab, n_units, n_layers, dropout)
            # self.bn = L.BatchNormalization(n_units*2)
            self.out = L.Linear(None, n_out)

    def __call__(self, x):
        # バッチ内の文書ごとに、各文をembedding
        sent_rep = self.doc_enc(x)  # x: ラベル文書 × ミニバッチサイズ
        sent_rep = F.concat([F.expand_dims(x, 0) for x in sent_rep], axis=0)
        # sent_rep = self.bn(sent_rep)
        # 出力層を噛ませる
        return self.out(sent_rep)


# 与えられた文書の分類を行う
class DocClassifyBi(DocClassify):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, n_out=4, dropout=0.5):
        super(DocClassify, self).__init__()
        with self.init_scope():
            self.doc_enc = DocRepBiRNN(n_vocab, n_units, n_layers, dropout)
            # self.bn = L.BatchNormalization(n_units * 4)
            # self.bn = L.BatchNormalization(1)
            self.out = L.Linear(None, n_out)


class DocClassifyBi2(DocClassify):
    def __init__(self, n_vocab=30000, n_units=200, n_layers=2, n_out=4, dropout=0.5):
        super(DocClassify, self).__init__()
        with self.init_scope():
            self.doc_enc = DocRepBiRNN(n_vocab, n_units, n_layers, dropout)
            # self.bn = L.BatchNormalization(n_units * 4)
            self.out = L.Linear(None, n_out)

    def __call__(self, x):
        # バッチ内の文書ごとに、各文をembedding
        sent_rep = self.doc_enc(x)  # x: ラベル文書 × ミニバッチサイズ
        sent_rep = F.concat([F.expand_dims(x, 0) for x in sent_rep], axis=0)
        # sent_rep = self.bn(sent_rep)
        # 出力層を噛ませる
        return self.out(sent_rep)
