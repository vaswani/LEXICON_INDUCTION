from common import *
import MatchingUtil as MU
import IO
import os.path


def med(X, Y):  # computes a matching by calculating the pairwise edit distance
    filename = IO.getEditDistFilename(X, Y)
    if os.path.isfile(filename):
        print 'found file', filename
        D = IO.readNumpyArray(filename)
    else:
        print 'file', filename, 'not found'
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
    #fileX = '../SCRIPTS/matlab/Jun10_en.txt'
    #fileY = '../SCRIPTS/matlab/Jun10_es.txt'
    X = IO.readWords(fileX)
    Y = IO.readWords(fileY)

    X.features = normalize_rows(X.features)
    Y.features = normalize_rows(Y.features)

    (cost, pi, edge_cost) = med(X.words, Y.words)
    matching = MU.getMatching(X.words, Y.words, pi, edge_cost)
    Y = MU.permuteFirstWords(Y, pi)
    MU.printMatching(X, Y, edge_cost)

    options = Options()
    options.exp_id = -1

    IO.writeMatching(options, X.words, Y.words, pi, edge_cost)

