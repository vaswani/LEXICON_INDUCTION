from common import *
import MatchingUtil as MU
import BilexiconUtil as BU
import CCAUtil as CU
from words import Words
import graphs
import IO
from termcolor import colored


from optparse import OptionParser


def find_matching(options, concatX, concatY):
    # finds a permutation pi that best matches Y to X
    # The optimization procedure works as follows:
    # suppose there are 2000 words to be matched, 100 seed words and step size is 100
    # The seed is stored at the end (so, X[i] matches Y[i] for i > 2000] in all iterations
    # at each iteration t (starting at t=0):
    # 1. compute the CCA on the last 100 + 100*t entries
    # 2. compute the CCA representation of all words
    # 3. perform a matching on the first N=2000 words to get pi_t
    # 4. sort the first 2000 matches in descending order.

    # initially, assume that pi is ID
    N = len(concatX.words)
    M = N - options.seed_length  # The first M entries can be permuted. The rest are fixed

    sigma = Struct()  # holds the cumulative permutations applied on X and Y
    sigma.X = perm.ID(M)
    sigma.Y = perm.ID(M)
    fixed_point = False
    for t in range(0, options.T):
        options.t = t
        Nt = M - options.step_size*t
        # STEP 0: when the feature dimension is high, ICD the seed and project the rest
        if concatX.isPickled():
            concatX.ICD_representation(Nt, options.eta)
            concatY.ICD_representation(Nt, options.eta)

        # STEP 1: compute CCA model on the well matched portion of the matching (which includes the fixed seed)
        fixedX = concatX.features[Nt:, :]
        fixedY = concatY.features[Nt:, :]
        if options.noise_level > 0:
            fixedX += options.noise_level*common.randn(fixedX.shape)
            fixedY += options.noise_level*common.randn(fixedY.shape)

        print colored('CCA dimensions =', 'green'), len(fixedX)
        cca_model = CU.learn(fixedX, fixedY , options)
        print len(cca_model.p), 'Top 10 correlation coefficients:', cca_model.p[:10]
        # STEP 2: compute CCA representation of all samples
        print 'norms', norm(concatX.features), norm(concatY.features)
        Z = CU.project(options, cca_model, concatX.features, concatY.features)

        print 'Z', norm(Z.X), norm(Z.Y)

        # STEP 3: compute weight matrix and run matching (approximate) algorithm
        W = MU.makeWeights(options, Z.X, Z.Y, concatX.G, concatY.G)
        (cost, pi_t, edge_cost) = MU.fast_ApproxMatch(W[:M, :M])
        # STEP 4: sort the words, such that the best matches are at the end.
        # note that pi_t is of length M < N and that
        (sorted_edge_cost, I) = perm.sort(edge_cost, reverse=True)
        sorted_edge_cost = np.concatenate((sorted_edge_cost, np.zeros(N-M)))

        if perm.isID(pi_t):  # the best permutation is the identity
            fixed_point = True
        else:
            concatX.permuteFirstWords(I)
            concatY.permuteFirstWords(pi_t[I])

            sigma.X = sigma.X[I]  # accumulate the changes from the ID
            sigma.Y = sigma.Y[pi_t[I]]  # accumulate the changes from the ID TODO: is this correct? (and do we use it?)
            # END OF ITERATION: output Matching
        print 'cost =', cost, 'latent inner product = ', np.sum(Z.X.A * Z.Y.A)

        MU.printMatching(concatX.words[:M], concatY.words[:M], sorted_edge_cost[:M], options.gold_lex)
        if options.gold_lex is not None:
            scores = BU.getScores(options.gold_lex, concatX.words[:M], concatY.words[:M], sorted_edge_cost[:M])
            BU.outputScores(scores, options.title)

        log(100, '---------- ', 'iteration = ', (t+1), '/', options.T, '----------\n')
        if fixed_point:
            break

    # either we reached the maximum number of iterations, or a fixed point

    log(100, 'Stopped after, ', t, 'iterations. Fixed point =', fixed_point)
    if options.is_mock:
        log('Hamming distance:', perm.hamming(concatX.words, concatY.words))
    return concatX, concatY, sigma, sorted_edge_cost, cost


def mcca(options, wordsX, wordsY, seedsX, seedsY, GX=None, GY=None):
    # (N, D) = wordsX.features.shape
    concatX = Words.concat(wordsX, seedsX)
    concatX.setupFeatures()
    concatX.G = GX

    concatY = Words.concat(wordsY, seedsY)
    concatY.setupFeatures()
    concatY.G = GY

    print colored("Initial matching hamming distance:", 'yellow'), perm.hamming(wordsX.words, wordsY.words), '/', len(wordsX.words)
    options.seed_length = len(seedsX.words)

    (newX, newY, sigma, edge_cost, cost) = find_matching(options, concatX, concatY)
    print "mcca is Done."
    return newX, newY, sigma, edge_cost, cost


def readInput(options, filename_wordsX, filename_wordsY, filename_seedX, filename_seedY, filename_graphX=None, filename_graphY=None):
    # load data files
    if options.pickled:
        wordsX = IO.readPickledWords(filename_wordsX)
        wordsY = IO.readPickledWords(filename_wordsY)
        seedsX = IO.readPickledWords(filename_seedX)
        seedsY = IO.readPickledWords(filename_seedY)   	    
    else:
        wordsX = IO.readWords(filename_wordsX)
        wordsY = IO.readWords(filename_wordsY)
        seedsX = IO.readWords(filename_seedX)
        seedsY = IO.readWords(filename_seedY)
    
    # load graphs and make stochastic
    if filename_graphX is not None:
        wordsX.G = IO.readNumpyArray(filename_graphX)
        wordsX.G = graphs.toSymmetricStochastic(wordsX.G)
    if filename_graphY is not None:
        wordsY.G = IO.readNumpyArray(filename_graphY)
        wordsY.G = graphs.toSymmetricStochastic(wordsY.G)

    # assert sizes are correct
    Nx = len(wordsX.words)
    Ny = len(wordsY.words)
    if Nx != Ny:
        log(0, 'Number of words must be the same', Nx, Ny)
    else:
        log(0, Nx, 'words loaded.')

    NSx = len(seedsX.words)
    NSy = len(seedsY.words)

    if NSx != NSy:
        log(0, 'Number of seed words must be the same', NSx, NSy)
    else:
        log(0, NSx, 'seed words loaded.')
    assert NSx == NSy

    if filename_graphX is not None:
        (NGx0, NGx1) = wordsX.G.shape
        (NGy0, NGy1) = wordsY.G.shape
        assert NGx0 == NGx1, 'GX is not an adjacency matrix'
        assert NGy0 == NGy1, 'GY is not an adjacency matrix'
        assert NGx0 == Nx + NSx, 'GX dimensions %d do not match those of X %d' % (NGx0, Nx+NSx)
        assert NGy0 == Ny + NSy, 'GY dimensions %d do not match those of Y %d' % (NGy0, Ny+NSy)

    # permute Y if rand_seed > 1, (this should only be used when testing on mock data)
    #wordsY.permuteFirstWords(perm.randperm(perm.ID(Ny)))
    #MU.printMatching(wordsX.words, wordsY.words, perm.ID(Ny))
    return wordsX, wordsY, seedsX, seedsY


def parseOptions():
    # parse cmdline arguments
    parser = OptionParser()
    # general setting
    parser.add_option('-e', '--exp_id', dest='exp_id', type="int", action='store', default=1001)
    parser.add_option('-l', '--lexicon', dest='filename_lexicon', action='store', default=None)    
    parser.add_option('-p', '--pickled', dest='pickled', type="int", action='store', default=1)
    # mcca setting
    parser.add_option('-z', '--step_size', dest='step_size', type="int", action='store', default=100)
    parser.add_option('-T', '--iterations', dest='T', type="int", action='store', default=10)
    parser.add_option('-w', '--weight_type', dest='weight_type', action='store', default='inner')    
    parser.add_option('-t', '--tau', dest='tau', type="float", action='store', default=0.001)  # CCA regularizer
    parser.add_option('--eta', dest='eta', type="float", action='store', default=0.001)
    parser.add_option('--norm_proj', dest='normalize_projections', type="int", action='store', default=1)
    parser.add_option('--covar_type', dest='covar_type', type="string", action='store', default=None)
    parser.add_option('--projection_type', dest='projection_type', type="string", action='store', default=None)
    # graph settings
    parser.add_option('-K', '--K', dest='K', type="int", action='store', default=0)
    parser.add_option('-a', '--alpha', dest='alpha', type="float", action='store', default=0)
    parser.add_option('-g', '--graph_type', dest='graph_type', action='store', default=None)
    # mock related options
    parser.add_option('--noise_level', dest='noise_level', type="float", action='store', default=0.0)
    parser.add_option('--record', dest='record', type="int", action='store', default=None)
    (options, args) = parser.parse_args()
    
    options.is_mock = 'mock' in (sys.argv[1])
    options.gold_lex = None
    
    # post processing of arguments
    
    # graph related options
    if options.graph_type is None:
        if options.K == 0:
            options.alpha = 0
            options.graph_type = None  # some graphs are dynamic (KNN), some are static.
        else:
            options.graph_type = 'KNN'

    options.title = "wt={}, K={}, alpha={}".format(options.weight_type, options.K, options.alpha)

    return options
    

if __name__ == '__main__':
    # cmd line args
    filename_wordsX = (sys.argv[1])
    filename_wordsY = (sys.argv[2])
    filename_seedsX = (sys.argv[3])
    filename_seedsY = (sys.argv[4])
    options = parseOptions()
    # read input files
    wordsX, wordsY, seedsX, seedsY = readInput(options, filename_wordsX, filename_wordsY, filename_seedsX, filename_seedsY)

    NSx = len(seedsY.words)

    if options.filename_lexicon is not None:
        lex = BU.readLexicon(options.filename_lexicon)
        (gold_lex, times) = BU.filterLexicon(lex, wordsX.words, wordsY.words)
        options.gold_lex = gold_lex
        print "Gold lexicon contains", len(gold_lex), 'pairs'
    else:
        options.gold_lex = None
        print colored("No gold lexicon", 'red')

    print len(seedsX.words), "seed pairs:", zip(seedsX.words, seedsY.words)
    (wordsX, wordsY, sigma, edge_cost, cost) = mcca(options, wordsX, wordsY, seedsX, seedsY)
    log(0, 'hamming distance:', perm.hamming(wordsX.words, wordsY.words))

