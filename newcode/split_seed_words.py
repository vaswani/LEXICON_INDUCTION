
__author__ = 'Tomer'

import IO
from common import *
import sys
import string
from collections import OrderedDict


def filter(X, N):
    (freq, I) = perm.sort(X.freq, reverse=True)
    I = I[:N]
    X.features = X.features[I, :]
    X.words = X.words[I]
    X.freq = X.freq[I]
    # filter in the top N words
    return X


def split_seed_words(X, seed_list):
    # split X into seed and words
    seeds = Words()
    words = Words()
    word2index = OrderedDict()
    # word ==> its row index
    for i, w in enumerate(X.words):
        word2index[w] = [i, 1]  # index in words and 1 if its a non-seed word (see below)

    for j, seed in enumerate(seed_list):
        [i, _] = word2index[seed]
        seeds.words.append(seed)
        seeds.freq.append(X.freq[i])
        seeds.features.append(X.features[i, :])
        word2index[seed] = [i, 0]  # it is removed - mark as seed

    for word in word2index:
        [i, v] = word2index[word]
        if v == 1:
            words.words.append(word)
            words.freq.append(X.freq[i])
            words.features.append(X.features[i, :])

    words.toNP()
    seeds.toNP()

    return seeds, words


def writeWords(filename, seed, words):
    seed_outname = filename.replace(".", "_seed.")
    words_outname = filename.replace(".", "_words.")
    IO.writeWords(seed_outname, seed)
    IO.writeWords(words_outname, words)


if __name__ == '__main__':
    # filters in the top N frequent words/freq/features and writes them to a file
    fileX = (sys.argv[1])  # csv of wordX,freq,features
    fileY = (sys.argv[2])  # csv of wordY,freq,features
    fileSeed = (sys.argv[3])   # csv of wordX,wordY

    seed_list = Struct()
    X = IO.readWords(fileX)
    Y = IO.readWords(fileY)
    seed_list.X, seed_list.Y = IO.readSeed(fileSeed)  # read the seed list (X,Y)

    seedX, wordsX = split_seed_words(X, seed_list.X)
    seedY, wordsY = split_seed_words(Y, seed_list.Y)

    writeWords(fileX, seedX, wordsX)
    writeWords(fileY, seedY, wordsY)