import sys
import IO
from MatrixStringKeys import MSK
import numpy as np
import common


# python extract_frequent_neighbors.py data/en-es/en_pickled_N\=3100.txt 10 > en_co_N=3100.edges
# python extract_frequent_neighbors.py data/en-es/es_pickled_N\=3100.txt 10 > es_co_N=3100.edges
if __name__ == '__main__':
    filename = sys.argv[1]
    K = int(sys.argv[2])

    wordsX = IO.readPickledWords(filename)
    M = MSK(wordsX.repr, wordsX.words, wordsX.featureNames)
    Nw, Nf = M.M.shape

    for iw in xrange(Nw):
        word = M.reverseStrings[iw]
        vw = M.M[iw, :]
        J = common.asVector(np.argsort(vw.todense()))
        count = 0
        if iw % 100 == 0:
            print >> sys.stderr, "word i", iw, '=', word
        for jf in reversed(J):
            feature = M.reverseFeatures[jf]
            if feature in M.strings:
                print "%s,%s,%d" % (word, feature, M.M[iw, jf])
            count += 1
            if count == K:
                break