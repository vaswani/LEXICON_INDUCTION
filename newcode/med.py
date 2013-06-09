from common import *
import MatchingUtil as MU
import IO


def med(X, Y): # match edit distance
    D = strings.pweditdist(X, Y)
    (cost, pi, edge_cost) = MU.ApproxMatch(D)
    # TODO:
    # 3. set up an initial matching based on edit distance.
    return pi

if __name__ == '__main__':
    # load data
    fileX = (sys.argv[1])
    fileY = (sys.argv[2])
    X = IO.readWords(fileX)
    Y = IO.readWords(fileY)

    pi = med(X.words, Y.words)
    matching = MU.getMatching(X.words, Y.words, pi)

    options = Options()
    options.exp_id = -1

    IO.writeMatching(options, X.words, Y.words, pi)
