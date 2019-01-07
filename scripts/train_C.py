import os
import sys
import argparse
import warnings

import chainer
from chainer import training, Variable, iterators, optimizers, serializers
from chainer.backends.cuda import get_device_from_id, to_cpu
from chainer.training import extensions
from chainer.datasets import split_dataset_random
import chainer.links as L
import chainer.cuda

import numpy as np

# import nets
# import nets_A
# import nets_B
import nets_B_NoBN
import data
from nlp_utils import convert_seq
from my_utils import *

import pickle

import matplotlib
matplotlib.use('Agg')


def main():
    set_random_seed(0)

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
    parser.add_argument('--vcb_file', '-vf',
                        default='/mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/vocab_train_w_NoReplace.vocab_file',
                        help='Vocabulary data file.')
    parser.add_argument('--case', '-c', default='original',
                        help='Select NN Architecture.')
    parser.add_argument('--opt', default='sgd',
                        help='Select Optimizer.')
    parser.add_argument('--dbg_on', action='store_true',
                        help='No save, MiniTrain')
    args = parser.parse_args()
    print(args)
    # train_val = data.DocDataset(args.train_file, vocab_size=args.vocab)

    if os.path.exists(args.vcb_file):  # args.vocab_fileの存在確認(作成済みの場合ロード)
        with open(args.vcb_file, 'rb') as f_vocab_data:
            train_val = pickle.load(f_vocab_data)
            if len(train_val.get_vocab()) != args.vocab:
                warnings.warn('vocab size incorrect (not implemented...)')
    else:
        train_val = data.DocDataset(args.train_file, vocab_size=args.vocab)  # make vocab from training data
        with open(args.vcb_file, 'wb') as f_vocab_save:
            pickle.dump(train_val, f_vocab_save)

    if args.dbg_on:
        len_train_data = len(train_val)
        N = 100
        print('N', N)
        rnd_ind = np.random.permutation(range(len_train_data))[:N]
        train_val = train_val[rnd_ind]
        (train, valid) = split_dataset_random(train_val, 80, seed=0)
    else:
        (train, valid) = split_dataset_random(train_val, 4000, seed=0)
        print('train', len(train))
        print('valid', len(valid))

    train_iter = iterators.SerialIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)

    # test = data.DocDataset(args.test_file, train_val.get_vocab())
    # test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    print('case', args.case)
    if args.case == 'original':
        print('originalで実行されます')
        result_path = 'result/original'
        model = L.Classifier(nets_B_NoBN.DocClassify(
            n_vocab=args.vocab+1, n_units=args.unit, n_layers=args.layer, n_out=4, dropout=args.dropout))
    elif args.case == 'bi':
        print('biで実行されます')
        result_path = 'result/bi'
        model = L.Classifier(nets_B_NoBN.DocClassifyBi(
            n_vocab=args.vocab+1, n_units=args.unit, n_layers=args.layer, n_out=4, dropout=args.dropout))
    elif args.case == 'bi2' or args.case == 'bi_adam_2layer' or args.case == 'bi2_adam_nobn':
        print('bi改良版')
        result_path = 'result/bi2'
        model = L.Classifier(
            nets_B_NoBN.DocClassifyBi2(n_vocab=args.vocab + 1, n_units=args.unit, n_layers=args.layer, n_out=4,
                                       dropout=args.dropout))
    else:
        warnings.warn('指定したケースは存在しません。デフォルトで実行します')
        result_path = 'result/sample_result'
        model = L.Classifier(nets_B_NoBN.DocClassify(n_vocab=args.vocab+1, n_units=args.unit, n_layers=args.layer,
                                                     n_out=4, dropout=args.dropout))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        # get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.opt == 'sgd':
        result_path += '_sgd'
        print('SGD')
        optimizer = optimizers.SGD(lr=0.01)
    elif args.opt == 'adam':
        result_path += '_adam'
        print('Adam')
        optimizer = optimizers.Adam()
    elif args.opt == 'bi_adam_2layer':
        result_path += '_adam_2layer'
        print('Adam')
        optimizer = optimizers.Adam()
    elif args.opt == 'bi2_adam_nobn':
        result_path += '_adam_nobn'
        print('Adam')
        optimizer = optimizers.Adam()
    else:
        print('指定なしのためSGDで実行')
        optimizer = optimizers.SGD(lr=0.01)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    # optimizer.add_hook(chainer.optimizer.Lasso(0.01))

    updater = training.StandardUpdater(train_iter, optimizer, converter=convert_seq, device=args.gpu)

    print('save here:', result_path)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_path)
    trainer.extend(extensions.LogReport())
    if not args.dbg_on:
        trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(valid_iter, model, converter=convert_seq, device=args.gpu), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ParameterStatistics(model.predictor.doc_enc, {'std': np.std}))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    if args.model:
        serializers.load_npz(args.model, trainer)

    trainer.run()

    pass


if __name__ == '__main__':
    main()
