"""
Copyright 2017, Sameer Khurana, All rights reserved.
"""

# Code based on https://github.com/OpenNMT/OpenNMT-py/blob/master/preprocess.py

import pickle
import numpy as np
import argparse
import sys
import dsol
from keras.preprocessing import text


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
parser.add_argument('-valid_tgt', required=False,
                    help="Path to the validation target data")
parser.add_argument('-test_src', required=False,
                    help="Path to the validation source data")
parser.add_argument('-test_tgt', required=False,
                    help="Path to the validation target data")
parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")
parser.add_argument('-lower', default=False,
                    help="Lower case dataset")
parser.add_argument('-start_char', default=1,
                    help="label for the start of the sequence")
parser.add_argument('-oov_char', default=2,
                    help="label for the out of vocab words")
parser.add_argument('-index_from', default=3,
                    help="start the words indices from")
parser.add_argument('-skip_top', default=0,
                    help="")
parser.add_argument('-char_level', default=True,
                    help="whether to have character level features")
parser.add_argument('-lowercase', default=False,
                    help="whether to lowercase the data")
parser.add_argument('-num_words', default=None,
                    help="restirct the vocab to the number of words")


opt = parser.parse_args()

np.random.seed(opt.seed)


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def make_vocabulary(filename):
    vocab = dsol.Dict([dsol.Constants.PAD_WORD, dsol.Constants.UNK_WORD,
                       dsol.Constants.BOS_WORD, dsol.Constants.EOS_WORD],
                      lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            for char in sent.rstrip():
                vocab.add(char)

    print('Created dictionary of size %d' %
          (vocab.size()))

    save_vocabulary('voc', vocab, 'data/vocabulary')
    
    return vocab


def init_vocabulary(name, dataFile):

    vocab = None

    print('Building ' + name + ' vocabulary...')
    gen_word_vocab = make_vocabulary(dataFile)

    vocab = gen_word_vocab

    print()
    return vocab


def make_data(src_file, tgt_file, train=False):
    """
    """
    src, tgt = [], []
    sizes = []
    count = 0

    print('Processing %s & %s ...' % (src_file, tgt_file))
    srcF = open(src_file, 'r')
    tgtF = open(tgt_file, 'r')

    all_lines = []
    all_tgts = []

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

        all_lines.append(sline)
        all_tgts.append(tline)

                
    srcF.close()
    tgtF.close()

    if train:
        vectorizer = text.Tokenizer(lower=opt.lower, split=" ", num_words=opt.num_words, char_level=opt.char_level)
        vectorizer.fit_on_texts(all_lines)
        opt.vectorizer = vectorizer

    # a list of lists of indices
    X = opt.vectorizer.texts_to_sequences(all_lines)

    # adding start of sequence character
    X = [[opt.start_char] + [w + opt.index_from for w in x] for x in X]

    nb_words = opt.num_words
    if nb_words is None:
        nb_words = max([max(x) for x in X])

    # replace indices with oov index if the word_idx in the list exceed num_words
    src = [[opt.oov_char if (w >= nb_words or w < opt.skip_top) else w for w in x] for x in X]
    tgt = all_tgts
    
    if opt.shuffle == 1:
        print('... shuffling sequences')
        perm = np.random.permutation(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences' %
          (len(src)))

    return src, tgt


def main():

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = make_data(opt.train_src, opt.train_tgt,
                                           train=True)

    valid = {}
    if (opt.valid_src!=None and opt.valid_tgt!=None):
	print('Preparing validation ...')
    	valid['src'], valid['tgt'] = make_data(opt.valid_src, opt.valid_tgt)

    test = {}
    if (opt.test_src!=None and opt.test_tgt!=None):
    	print('Preparing Test ...')
    	test['src'], test['tgt'] = make_data(opt.test_src, opt.test_tgt)

    print('Saving data to \'' + opt.save_data + '\'...')
    save_data = {'train': train,
                 'valid': valid,
                 'test': test}

    with open(opt.save_data, 'wb') as handle:
        pickle.dump(save_data, handle)


if __name__ == "__main__":
    main()
