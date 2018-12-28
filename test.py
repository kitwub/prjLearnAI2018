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
    args = parser.parse_args()

    train_val = data.DocDataset(args.train_file, vocab_size=args.vocab)
    test = [x[0] for x in data.DocDataset(args.test_file, train_val.get_vocab())]
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    model = nets.DocClassify(n_vocab=args.vocab+1, n_units=args.unit, n_layers=args.layer, n_out=4, dropout=args.dropout)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.model:
        serializers.load_npz(args.model, model, 'updater/model:main/predictor/')

    with chainer.using_config('train', False):
        while True:
            result = model(convert_seq(test_iter.next(), device=args.gpu, with_label=False))
            print(result)

if __name__ == '__main__':
    main()
