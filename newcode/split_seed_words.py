from collections import OrderedDict
from common import *
import sys
import IO
import words


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
    seedsX = words.Words()
    wordsX = words.Words()
    word2index = OrderedDict()
    # word ==> its row index
    for i, w in enumerate(X.words):
        word2index[w] = [i, 1]  # index in words and 1 if its a non-seed word (see below)

    for j, seed in enumerate(seed_list):
        [i, _] = word2index[seed]
        seedsX.words.append(seed)
        seedsX.freq.append(X.freq[i])
        seedsX.features.append(X.features[i, :])
        word2index[seed] = [i, 0]  # it is removed - mark as seed

    for word in word2index:
        [i, v] = word2index[word]
        if v == 1:
            wordsX.words.append(word)
            wordsX.freq.append(X.freq[i])
            wordsX.features.append(X.features[i, :])

    wordsX.toNP()
    seedsX.toNP()

    return seedsX, wordsX


def writeWords(filename, seed, words):
    filename_seed = filename.replace(".", "_seed.")
    filename_words = filename.replace(".", "_words.")
    IO.writeWords(filename_seed, seed)
    IO.writeWords(filename_words, words)


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