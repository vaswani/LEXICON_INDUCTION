from collections import defaultdict
from optparse import OptionParser

import IO
from graphs import toSymmetricStochastic
from words import *
import graphs
from MatrixStringKeys import MSK


def parseOptions():
    # parse cmdline arguments
    parser = OptionParser()
    # general setting
    parser.add_option('--sym', dest='sym', type="int", action='store', default=1)
    parser.add_option('--stoc', dest='stochastic', type="int", action='store', default=1)
    parser.add_option('--normalize', dest='normalize', type="int", action='store', default=1)
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    # parse arguments
    filename_wordsX = (sys.argv[1])
    filename_edges = (sys.argv[2])

    # read input
    wordsX = IO.readPickledWords(filename_wordsX)
    options = parseOptions()

    # load edges
    lines = IO.readCSV(filename_edges)
    DD_lower = defaultdict(dict)
    for row in lines:
        a = row[0].lower()
        b = row[1].lower()
        DD_lower[a][b] = 1
        DD_lower[b][a] = 1

    # connect (w,u) if an edge (w_lower, u_lower) exists
    words = wordsX.words
    N = len(words)
    print >> sys.stderr, "Building graph NxN", N
    count = 0;
    DD = {}
    lower_words = [w.lower() for w in words]
    for (i, w) in enumerate(words):
        w_lower = lower_words[i]
        DD[w] = {}
        DD[w][w] = 1
        if w_lower in DD:
            for (j, u) in enumerate(words):
                if u != w:
                    u_lower = lower_words[j]
                    if u_lower in DD_lower[w_lower]:
                        DD[w][u] = DD_lower[w_lower][u_lower]
                        #print "%s, %s" % (w, u)
                        count += 1

    msk = MSK(DD, words, words)

    G = msk.M
    G = G.todense()

    if options.normalize == 1:
        G = toSymmetricStochastic(G, sym=(options.sym == 1), stochastic=(options.stochastic == 1), norm='l1')
    elif options.normalize == 2:
        G = toSymmetricStochastic(G, sym=(options.sym == 1), stochastic=(options.stochastic == 1), norm='l2')

    msk.M = G
    graphFilename = filename_edges.replace('.', '_graph.')
    print >> sys.stderr, "Writing graph for N =", N, "words, edge count =", count
    print >> sys.stderr, "Filename:", graphFilename
    IO.pickle(graphFilename, msk)