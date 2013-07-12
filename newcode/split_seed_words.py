from collections import OrderedDict
from optparse import OptionParser
from common import *
import sys
import IO
import words
import pprint


def filter(X, N):
    (freq, I) = perm.sort(X.freq, reverse=True)
    I = I[:N]
    X.features = X.features[I, :]
    X.words = X.words[I]
    X.freq = X.freq[I]
    # filter in the top N words
    return X


def split_seed_words(X, seed_list, pickled):
    # split X into seed and words
    seedsX = words.Words()
    wordsX = words.Words()
    word2index = OrderedDict()
    # word ==> its row index
    for i, w in enumerate(X.words):
        word2index[w] = [i, 1]  # index in words and 1 if its a non-seed word (see below)
    # arrange seeds in seedsX
    for j, seed in enumerate(seed_list):
        [i, _] = word2index[seed]
        seedsX.words.append(seed)
        seedsX.freq.append(X.freq[i])

        if pickled:
            seedsX.repr[seed] = X.repr[seed]
        else:
            seedsX.features.append(X.features[i, :])
        word2index[seed] = [i, 0]  # it is removed - mark as seed
    # arrange the rest of the words in wordsX
    for word in word2index:
        [i, v] = word2index[word]
        if v == 1:
            wordsX.words.append(word)
            wordsX.freq.append(X.freq[i])

            if pickled:
                wordsX.repr[word] = X.repr[word]
            else:
                wordsX.features.append(X.features[i, :])

    wordsX.toNP()
    seedsX.toNP()

    return seedsX, wordsX


def writeWords(filename, seeds, words, pickled):
    filename_seed = filename.replace(".", "_seed.")
    filename_words = filename.replace(".", "_words.")
    if pickled:
        # in this case, it is important to pickle the words in order of insertion.
        freq = OrderedDict()
        for i, w in enumerate(seeds.words):
            freq[w] = seeds.freq[i]
        IO.writePickledWords(filename_seed, freq, seeds.repr)
        freq = OrderedDict()
        for i, w in enumerate(seeds.words):
            freq[w] = seeds.freq[i]
        freq = {word: words.freq[i] for i, word in enumerate(words.words)}
        IO.writePickledWords(filename_words, freq, words.repr)
    else:
        IO.writeWords(filename_seed, seeds)
        IO.writeWords(filename_words, words)


if __name__ == '__main__':
    # filters in the top N frequent words/freq/features and writes them to a file
    filename_X = (sys.argv[1])  # csv of wordX,freq,features
    filename_Y = (sys.argv[2])  # csv of wordY,freq,features
    filename_seed = (sys.argv[3])   # csv of wordX,wordY

    parser = OptionParser()
    # general setting
    parser.add_option('-p', '--pickled', dest='pickled', type="int", action='store', default=1)
    (options, args) = parser.parse_args()
    pickled = options.pickled

    seed_list = Struct()

    if pickled:
        X = IO.readPickledWords(filename_X)
        Y = IO.readPickledWords(filename_Y)
    else:
        X = IO.readWords(filename_X)
        Y = IO.readWords(filename_Y)

    seed_list.X, seed_list.Y = IO.readSeed(filename_seed)  # read the seed list (X,Y)

    seedsX, wordsX = split_seed_words(X, seed_list.X, pickled)
    seedsY, wordsY = split_seed_words(Y, seed_list.Y, pickled)

    print "seed:"
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(zip(seedsX.words, seedsY.words))

    writeWords(filename_X, seedsX, wordsX, pickled)
    writeWords(filename_Y, seedsY, wordsY, pickled)