from optparse import OptionParser
import sys
import IO
from MatrixStringKeys import MSK
import numpy as np
import common
from sklearn.preprocessing import normalize

def parseOptions():
    # parse cmdline arguments
    parser = OptionParser()
    # general setting
    parser.add_option('--K', dest='K', type="int", action='store', default=10) # how many neighbors to extract?
    parser.add_option('--minCoFreq', dest='minCoFreq', type="int", action='store', default=0) # but only if they occurred at least minCoFreq times.
    parser.add_option('--M', dest='M', type="int", action='store', default=150) # remove words that co-occur with more than 50 distance words (they carry no information)
    (options, args) = parser.parse_args()
    return options


def output_edges(M, L, reverseLookup):
    L = np.array(L)
    print >> sys.stderr, 'class:', L.__class__
    # output top K values.
    Nw, Nf = M.M.shape
    for iw in xrange(Nw):
        word = M.reverseStrings[iw]  # get words
        vw = L[iw, :]
        J = common.asVector(np.argsort(vw))
        count = 0
        if iw % 100 == 0:
            print >> sys.stderr, "word i", iw, '=', word

        for jf in reversed(J):
            feature = reverseLookup[jf]
            if feature in M.strings and L[iw, jf] > 0:
                print "%s,%s,%f" % (word, feature, L[iw, jf])
                #count += 1
            if count == options.K:
                break

# python extract_frequent_neighbors.py data/en-es/en_pickled_N\=3100.txt 10 > en_co_N=3100.edges
# python extract_frequent_neighbors.py data/en-es/es_pickled_N\=3100.txt 10 > es_co_N=3100.edges

if __name__ == '__main__':
    filename = sys.argv[1]
    graph_mode = int(sys.argv[2])
    options = parseOptions()

    wordsX = IO.readPickledWords(filename)
    M = MSK(wordsX.repr, wordsX.words, wordsX.featureNames)

    if graph_mode == 1:  # remove words (columns) that are too frequently occurring
        L = M.M.todense()
        I = (L > 0).sum(axis=0) >= options.M  # find words that co-occur with at least M distinct words
        J = np.nonzero(np.array(I)[0])[0]
        # pi_f = [M.features[i] for i in M.strings]
        # pi_s = [M.strings[i] for i in M.strings]
        # P = L[pi_s, pi_f]
        FCW = set([M.reverseFeatures[j] for j in J])
        print >> sys.stderr, 'FCW length:', len(FCW)
        #too_frequent = FCW.intersection(wordsX.words)

        L = np.array(L)
        for w in FCW:
            i = M.features[w]
            L[:, i] = 0
        #L *= (L > options.minCoFreq)
        output_edges(M, L, M.reverseFeatures)
    elif graph_mode == 2:  # PMI
        M.M = M.M.todense()
        L = M.materialize(wordsX.words, wordsX.words)  # L's rows/columns are ordered by wordsX.words
        L[:, :1500] = 0  # remove common words words
        L = np.array(L) * np.array(L>options.minCoFreq) # remove low occuring bigrams
        #L = normalize(L, norm, axis=1)  # normalize rows
        P = np.triu(L)
        unigram = np.mat(wordsX.freq*1.0 / np.sum(wordsX.freq))
        P -= np.diag(np.diag(P))  # remove diagonal
        P += P.T
        P /= P.sum() # P now contains the joint probability P[i,j]

        Q = np.array(unigram.T * unigram)  # Q_ij = Ui*Uj
        PMI = P / Q  # pointwise mutual information
        output_edges(M, PMI, M.reverseStrings)