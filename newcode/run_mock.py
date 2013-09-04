import copy
from common import *
import makeGraph
import graphs
import mcca


if __name__ == '__main__':
    # parse input
    # first four arguments are filenames for
    #   wordsX, wordsY, seedX, seedY
    # next 2 arguments are
    #   noise, T=number of repetitions
    filename_wordsX = (sys.argv[1])
    filename_wordsY = (sys.argv[2])
    filename_seedX = (sys.argv[3])
    filename_seedY = (sys.argv[4])
    R = int(sys.argv[5])  # experiment repetitions

    options = mcca.parseOptions()

    noise_levels = [options.noise_level]
    hammings = np.zeros(R)

    for (i, noise) in enumerate(noise_levels):
        for r in xrange(R):
            np.random.seed(r)
            # read files
            wordsX, wordsY, seedsX, seedsY = mcca.readInput(options, filename_wordsX, filename_wordsY, filename_seedX, filename_seedY)
            seed_length = len(seedsX.words)
            pi = perm.randperm(xrange(len(wordsY.words)))
            wordsY.permuteFirstWords(pi)
            if options.K > 0:
                GX = makeGraph.makeGraph(wordsX, seedsX, options.graph_type, options.K)
                GY = makeGraph.makeGraph(wordsY, seedsY, options.graph_type, options.K)
                GY = graphs.permute(GY, pi)
                
                GX = graphs.toSymmetricStochastic(GX)
                GY = graphs.toSymmetricStochastic(GY)
                print 'Graph norms', norm(GX), norm(GY)
            else:
                GX = None
                GY = None
                print 'no graphs provided'
            # add random noise to the features
            if noise > 0:
                if not options.pickled:
                    wordsX.features += noise * randn(wordsX.features.shape)
                    wordsY.features += noise * randn(wordsY.features.shape)
                    seedsX.features += noise * randn(seedsX.features.shape)
                    seedsY.features += noise * randn(seedsY.features.shape)
                else:
                    wordsX.addReprNoise(noise)
                    wordsY.addReprNoise(noise)
                    seedsX.addReprNoise(noise)
                    seedsY.addReprNoise(noise)
            #print np.linalg.norm(GX-GY)
            # run MCCA
            (wordsX, wordsY, sigma, edge_cost, cost) = mcca.mcca(options, wordsX, wordsY, seedsX, seedsY, GX, GY)
            hammings[r] = perm.hamming(wordsX.words, wordsY.words)

    N = len(pi)
    print options
    print 'Hamming distances:', hammings
    print 'Hamming distance over', R, 'trials', '[mean, std] = ', np.mean(hammings), '|', np.std(hammings)
