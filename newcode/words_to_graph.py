from optparse import OptionParser

import IO
from graphs import toSymmetricStochastic
from words import *
import graphs
from MatrixStringKeys import MSK


def makeGraph(wordsX, options):
    wordsX.setupFeatures()
    #wordsX.computeKernel()
    #kernel = wordsX.getKernel()
    kernel = wordsX.computeContextKernel()
    if options.KNN > 0:
        G, ngbr_idx = graphs.kernel_knn_graph(kernel.K, options.KNN)
        return G
    return 0


def parseOptions():
    # parse cmdline arguments
    parser = OptionParser()
    # general setting
    parser.add_option('--sym', dest='sym', type="int", action='store', default=1)
    parser.add_option('--stoc', dest='stochastic', type="int", action='store', default=1)
    parser.add_option('--KNN', dest='KNN', type="int", action='store', default=10)
    parser.add_option('--normalize', dest='normalize', type="int", action='store', default=1)
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    # parse arguments
    filename_wordsX = (sys.argv[1])

    # read input
    wordsX = IO.readPickledWords(filename_wordsX)
    options = parseOptions()

    # make graph
    G = makeGraph(wordsX, options)
    G = G.todense()

    if options.normalize == 1:
        G = toSymmetricStochastic(G, sym=(options.sym == 1), stochastic=(options.stochastic == 1), norm='l1')
    elif options.normalize == 2:
        G = toSymmetricStochastic(G, sym=(options.sym == 1), stochastic=(options.stochastic == 1), norm='l2')

    msk = MSK(None, wordsX.words, wordsX.words)
    # save the matrix.
    # This is hacky, since we're trusting that G is generated with rows/columns that match the order of wordsX.words
    msk.M = G
    graphFilename = filename_wordsX.replace(".", "_WG.")
    if options.KNN > 0:
        graphFilename = graphFilename.replace(".", "_KNN"+str(options.KNN)+".")

    IO.pickle(graphFilename, msk)