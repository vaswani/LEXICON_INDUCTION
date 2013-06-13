from common import *
import MatchingUtil as MU
import IO
import os.path


def setupFeatures(options, X):
    (N,D) = X.features.shape
    (freq, I) = perm.sort(X.freq, reverse=True)
    I = I[:options.max_words]

    X.features = X.features[I, :]
    X.words = X.words[I]

    X.features = normalize_rows(X.features)
    # TODO: should be add logFr and L ?
    return X


def med(X, Y): # match edit distance
    filename = IO.getEditDistFilename(X, Y)
    if os.path.isfile(filename):
        D = IO.readNumpyArray(filename)
    else:
        D = strings.pweditdist(X, Y)
        IO.writeNumpyArray(filename, D)
    (cost, pi, edge_cost) = MU.ApproxMatch(D)
    # TODO:
    # 3. set up an initial matching based on edit distance.
    return cost, pi, edge_cost

if __name__ == '__main__':
    # load data
    fileX = (sys.argv[1])
    fileY = (sys.argv[2])
    fileX = '../SCRIPTS/matlab/Jun10_en.txt'
    fileY = '../SCRIPTS/matlab/Jun10_es.txt'
    X = IO.readWords(fileX)
    Y = IO.readWords(fileY)

    options = Options()
    options.max_words = 2000;
    X = setupFeatures(options, X)
    Y = setupFeatures(options, Y)

    (cost, pi, edge_cost) = med(X.words, Y.words)
    matching = MU.getMatching(X.words, Y.words, pi, edge_cost)

    options = Options()
    options.exp_id = -1

    IO.writeMatching(options, X.words, Y.words, pi, edge_cost)
