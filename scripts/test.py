from datetime import datetime
import os
import sys
import argparse

import chainer
import chainer.cuda
from chainer import training, Variable, iterators, optimizers, serializers
from chainer.training import extensions
# from chainer.datasets import split_dataset_random
import chainer.links as L

import numpy as np

import nets
import data
from nlp_utils import convert_seq
from my_utils import get_str_of_val_name_on_code

import pickle

import matplotlib
from pandas.tests.test_compat import test_re_type

matplotlib.use('Agg')


def main():
    parser = argparse.ArgumentParser(
        description='Document Classification Example')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of documents in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=200,
                        help='Number of units')
    parser.add_argument('--vocab', '-v', type=int, default=50000,
                        help='Vocabulary size')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers of LSMT')
    parser.add_argument('--dropout', '-d', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--gradclip', type=float, default=5,
                        help='Gradient clipping threshold')
    parser.add_argument('--train_file', '-train', default='data/train.seg.csv',
                        help='Trainig data file.')
    parser.add_argument('--test_file', '-test', default='data/test.seg.csv',
                        help='Test data file.')
    parser.add_argument('--model', '-m', help='read model parameters from npz file')
    parser.add_argument('--vcb_file',
                        default='/mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/vocab_train_w_NoReplace.vocab_file',
                        help='Vocabulary data file.')
    args = parser.parse_args()

    if os.path.exists(args.vcb_file):  # args.vocab_fileの存在確認(作成済みの場合ロード)
        with open(args.vcb_file, 'rb') as f_vocab_data:
            train_val = pickle.load(f_vocab_data)
    else:
        train_val = data.DocDataset(args.train_file, vocab_size=args.vocab)  # make vocab from training data
        with open(args.vcb_file, 'wb') as f_vocab_save:
            pickle.dump(train_val, f_vocab_save)


    # train_val = data.DocDataset(args.train_file, vocab_size=args.vocab)  # make vocab from training data
    # test = [x[0] for x in data.DocDataset(args.test_file, train_val.get_vocab())]  # [ データ１[文１[], 文２[], ...], データ２[文１[], 文２[], ...], ... ]
    # test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # 文章,ラベルを同時取得
    # test_doc_label = [x for x in data.DocDataset(args.test_file, train_val.get_vocab())]  # [ データ１[文１[], 文２[], ...], データ２[文１[], 文２[], ...], ... ]
    test_doc_label = data.DocDataset(args.test_file, train_val.get_vocab())
    test_doc = [x[0] for x in test_doc_label]
    test_label = [x[1] for x in test_doc_label]
    test_iter = iterators.SerialIterator(test_doc, args.batchsize, repeat=False, shuffle=False)
    test_label_iter = iterators.SerialIterator(test_label, args.batchsize, repeat=False, shuffle=False)
    # test_doc_label_iter = iterators.SerialIterator(test_doc_label, args.batchsize, repeat=False, shuffle=False)

    model = nets.DocClassify(n_vocab=args.vocab+1, n_units=args.unit, n_layers=args.layer, n_out=4, dropout=args.dropout)
    # load npzができなくなる→解消？
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.model:
        serializers.load_npz(args.model, model, 'updater/model:main/predictor/')

    confusion_mat = np.zeros([4, 4])  # [label, prediction]
    # model2 = L.Classifier(nets.DocClassify(n_vocab=args.vocab+1, n_units=args.unit, n_layers=args.layer, n_out=4, dropout=args.dropout))
    with chainer.using_config('train', False):
        # test_eval = extensions.Evaluator(test_doc_label_iter, model, converter=convert_seq, device=args.gpu)
        # test_result = test_eval()

        # while True:
        #     result = model(convert_seq(test_iter.next(), device=args.gpu, with_label=False))
        #     test_label_batch = test_label_iter.next()

        for (label_batch, each_testinput_batch) in zip(test_label_iter, test_iter):
            result = model(convert_seq(each_testinput_batch, device=args.gpu, with_label=False))
            predict = np.argmax(result.array, axis=1)

            for (each_label, each_predict) in zip(label_batch, predict):
                confusion_mat[each_label][chainer.cuda.to_cpu(each_predict)] += 1

    print(confusion_mat)

    # dummy_val = 'dummy data'

    time_now = datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = '/mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/vocab_train_w_NoReplace.saved_'
    save_val_str = get_str_of_val_name_on_code(confusion_mat)[0]

    # for (each_val, each_val_str) in zip(save_val, save_val_str):
    #     with open(save_path + each_val_str + time_now, 'wb') as f_save:
    #         pickle.dump(each_val, f_save)
    with open(save_path + save_val_str + '_' + time_now, 'wb') as f_save:
        pickle.dump(confusion_mat, f_save)

    pass  # for breakpoint

if __name__ == '__main__':
    main()
