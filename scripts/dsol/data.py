# Code based on https://github.com/OpenNMT/OpenNMT-py/blob/master/preprocess.py

import pickle
import numpy as np
import argparse
import sys
import dsol

__author__ = 'Sameer Khurana'
__email__ = 'sameerkhurana10@gmail.com'
__version__ = '0.2'


parser = argparse.ArgumentParser(description='preprocess.py')


# Preprocess options

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")
parser.add_argument('-test_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-test_tgt', required=True,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-seq_length', type=int, default=80,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")


opt = parser.parse_args()

np.random.seed(opt.seed)


def make_vocabulary(filename):
    vocab = dsol.Dict([dsol.Constants.PAD_WORD, dsol.Constants.UNK_WORD,
                       dsol.Constants.BOS_WORD, dsol.Constants.EOS_WORD],
                      lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    print('Created dictionary of size %d' %
          (vocab.size()))

    return vocab


def init_vocabulary(name, dataFile):

    vocab = None

    print('Building ' + name + ' vocabulary...')
    gen_word_vocab = make_vocabulary(dataFile)

    vocab = gen_word_vocab

    print()
    return vocab


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def make_data(src_file, tgt_file, src_dicts):
    """
    """
    src, tgt = [], []
    sizes = []
    count = 0

    print('Processing %s & %s ...' % (src_file, tgt_file))
    srcF = open(src_file, 'r')
    tgtF = open(tgt_file, 'r')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('Error: source and target do not have the same number of sentences')
            sys.exit(-1)
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        src_chars = sline.split()
        tgt_label = tline

        src += [src_dicts.convertToIdx(src_chars,
                                       dsol.Constants.UNK_WORD)]
        tgt += [tgt_label]

        sizes += [len(src_chars)]

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = np.random.permutation(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = sizes.sort()
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences' %
          (len(src)))

    return src, tgt


def main():

    dicts = {}
    # generate vocabulary
    dicts['src'] = init_vocabulary('source', opt.train_src)

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = make_data(opt.train_src, opt.train_tgt,
                                           dicts['src'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = make_data(opt.valid_src, opt.valid_tgt,
                                           dicts['src'])

    print('Preparing Test ...')
    test = {}
    test['src'], test['tgt'] = make_data(opt.test_src, opt.test_tgt,
                                         dicts['src'])

    print('Saving data to \'' + opt.save_data + '.data.pickle\'...')
    save_data = {'train': train,
                 'valid': valid,
                 'test': test}
    with open(opt.save_data + '.data.pickle') as handle:
        pickle.dump(save_data, handle)


if __name__ == "__main__":
    main()
