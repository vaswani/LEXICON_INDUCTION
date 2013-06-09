from common import *
import MatchingUtil as MU
import IO

def med(X, Y): # match edit distance
    D = strings.pweditdist(X, Y)
    cost, pi = MU.ApproxMatch(D)
    # TODO:
    # 1. evaluate the matching result (need a dictionary)
    # 2. save the edit distance permutation - filename should be dependent on the order of lists.
    # 3. set up an initial matching based on edit distance.
    print pi, cost
    return cost, pi

if __name__ == '__main__':
    # load data
    fileX = (sys.argv[1])
    fileY = (sys.argv[2])
    X = IO.readFeatures(fileX)
    Y = IO.readFeatures(fileY)

    med(X.words, Y.words)
