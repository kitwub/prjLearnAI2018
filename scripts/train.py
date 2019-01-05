import os
import sys
import argparse

import chainer
from chainer import training, Variable, iterators, optimizers, serializers
from chainer.training import extensions
from chainer.datasets import split_dataset_random
import chainer.links as L

import numpy as np

import nets
import data
from nlp_utils import convert_seq

import pickle

import matplotlib
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
    parser.add_argument('--vocab_file', '-vf', default='/mnt/gold/users/s18153/prjPyCharm/prjNLP_GPU/data/vocab_train_w_NoReplace.vocab_file',
                        help='Vocabulary data file.')
    args = parser.parse_args()

    train_val = data.DocDataset(args.train_file, vocab_size=args.vocab)

    if os.path.exist(args.vocab_file):  # args.vocab_fileの存在確認(未作成の場合、新規作成)
        with open(args.vocab_file, 'wb') as vocab_data_file:
            pickle.dump(train_val, vocab_data_file)


    # test = data.DocDataset(args.test_file, train_val.get_vocab())
    (train, valid) = split_dataset_random(train_val, 4000, seed=0)

    train_iter = iterators.SerialIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)
    # test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    model = L.Classifier(nets.DocClassify(n_vocab=args.vocab+1, n_units=args.unit, n_layers=args.layer, n_out=4, dropout=args.dropout))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.SGD(lr=0.01)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    updater = training.StandardUpdater(train_iter, optimizer, converter=convert_seq, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='sample_result')
    trainer.extend(extensions.LogReport())
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


if __name__ == '__main__':
    main()
