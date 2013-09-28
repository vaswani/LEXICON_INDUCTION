from common import *
import MatchingUtil as MU
import BilexiconUtil as BU
import CCAUtil as CU
import perm
import IO
from termcolor import colored


from optparse import OptionParser


def find_matching(options, wordsX, wordsY):
    # finds a permutation pi that best matches Y to X
    # The optimization procedure works as follows:
    # suppose there are 2000 words to be matched, 100 seed words and step size is 100
    # The seed is stored at the end (so, X[i, :] matches Y[i, :] for i > 2000] in all iterations
    # at each iteration t (starting at t=0):
    # 1. compute the CCA on the last 100 + 100*t entries
    # 2. compute the CCA representation of all words
    # 3. perform a matching on the first N=2000 words to get pi_t
    # 4. sort the first 2000 matches in descending order.

    # initially, assume that pi is ID
    N = len(wordsX.words)
    M = N - options.seed_length  # The first M entries can be permuted. The rest are fixed
    GX = None
    GY = None

    fixed_point = False
    for t in range(0, options.T):
        options.t = t
        Nt = M - options.step_size*t
        # STEP 0: when the feature dimension is high, ICD the seed and project the rest
        if wordsX.isPickled():
            wordsX.ICD_representation(Nt, options.eta)
            wordsY.ICD_representation(Nt, options.eta)

        # STEP 1: compute CCA model on the well matched portion of the matching (which includes the fixed seed)
        fixedX = wordsX.features[Nt:, :]
        fixedY = wordsY.features[Nt:, :]
        # if options.noise_level > 0:
        #     fixedX += options.noise_level*common.randn(fixedX.shape)
        #     fixedY += options.noise_level*common.randn(fixedY.shape)

        print >> sys.stderr, colored('CCA dimensions =', 'green'), len(fixedX)
        cca_model = CU.learn(fixedX, fixedY , options)
        print >> sys.stderr, len(cca_model.p), 'Top 10 correlation coefficients:', cca_model.p[:10]
        # STEP 2: compute CCA representation of all samples
        print >> sys.stderr, 'norms', norm(wordsX.features), norm(wordsY.features)
        Z = CU.project(options, cca_model, wordsX.features, wordsY.features)

        print >> sys.stderr, 'Z', norm(Z.X), norm(Z.Y)

        # STEP 3: compute weight matrix and run matching (approximate) algorithm
        if options.alpha > 0:
            GX = wordsX.materializeGraph()
            GY = wordsY.materializeGraph()
        print >> sys.stderr, colored('Computing matching weight matrix.', 'green')
        W, U0, Z0 = MU.makeWeights(options, Z.X, Z.Y, GX, GY)
        print >> sys.stderr, 'Matching.'
        (cost, pi_t, edge_cost) = MU.exactMatch(W[:M, :M])
        # STEP 4: sort the words, such that the best matches are at the end.
        # note that pi_t is of length M < N and that
        (sorted_edge_cost, I) = perm.sort(edge_cost, reverse=True)
        sorted_edge_cost = np.concatenate((sorted_edge_cost, np.zeros(N-M)))

        if perm.isID(pi_t):  # the best permutation is the identity
            fixed_point = True
        else:
            wordsX.permuteFirstWords(I)
            wordsY.permuteFirstWords(pi_t[I])
            # END OF ITERATION: output Matching
        print >> sys.stderr, 'cost =', cost, 'latent inner product = ', np.sum(Z.X.A * Z.Y.A)

        #MU.printMatching(wordsX.words[:M], wordsY.words[:M], sorted_edge_cost[:M], options.gold_lex)
        if options.gold_lex is not None:
            scores = BU.getScores(options.gold_lex, wordsX.words[:M], wordsY.words[:M], sorted_edge_cost[:M])
            BU.outputScores(scores, options.title)

        print '---------- ', 'iteration = ', (t+1), '/', options.T, '----------'
        sys.stdout.flush()
        if fixed_point:
            break

    # either we reached the maximum number of iterations, or a fixed point
    log(100, 'Stopped after, ', (t+1), 'iterations. Fixed point =', fixed_point)
    IO.writeString(options.matchingFilename, MU.toString(wordsX.words[:M], wordsY.words[:M], sorted_edge_cost[:M], options.gold_lex))
    if options.is_mock:
        log('Hamming distance:', perm.hamming(wordsX.words, wordsY.words))
    return wordsX, wordsY, sorted_edge_cost, cost


def mcca(options, wordsX, wordsY, seed_list):
    N_seed = len(seed_list.X)
    # (N, D) = wordsX.features.shape
    wordsX.setupFeatures()
    wordsX.computeKernel(options)

    wordsY.setupFeatures()
    wordsY.computeKernel(options)

    print >> sys.stderr, colored("Initial matching hamming distance:", 'yellow'), perm.hamming(wordsX.words[:N_seed], wordsY.words[:N_seed]), '/', N_seed
    options.seed_length = N_seed

    (newX, newY, edge_cost, cost) = find_matching(options, wordsX, wordsY)
    print >> sys.stderr, colored("mcca is Done.", 'green')
    return newX, newY, edge_cost, cost


def readInput(options, filename_wordsX, filename_wordsY, filename_seed):
    # load data files
    if options.pickled:
        wordsX = IO.readPickledWords(filename_wordsX)
        wordsY = IO.readPickledWords(filename_wordsY)
    else:
        wordsX = IO.readWords(filename_wordsX)
        wordsY = IO.readWords(filename_wordsY)

    if options.filename_graphX is not None:
        print 'loading graph -', options.filename_graphX
        wordsX.G = IO.unpickle(options.filename_graphX)
        print 'loading graph -', options.filename_graphY
        wordsY.G = IO.unpickle(options.filename_graphY)

    seed_list = Struct()
    seed_list.X, seed_list.Y = IO.readSeed(filename_seed)  # read the seed list (X,Y)
    wordsX.pushSeedToEnd(seed_list.X)
    wordsY.pushSeedToEnd(seed_list.Y)

    # assert sizes are correct
    Nx = len(wordsX.words)
    Ny = len(wordsY.words)
    if Nx != Ny:
        log(0, 'Number of words must be the same', Nx, Ny)
    else:
        log(0, Nx, 'words loaded.')

    NSx = len(seed_list.X)
    NSy = len(seed_list.Y)

    if NSx != NSy:
        log(0, 'Number of seed words must be the same', NSx, NSy)
    else:
        log(0, NSx, 'seed words loaded.')
    assert NSx == NSy

    if options.filename_graphX is not None:
        (NGx0, NGx1) = wordsX.G.shape()
        (NGy0, NGy1) = wordsY.G.shape()
        assert NGx0 == NGx1, 'GX is not a square adjacency matrix'
        assert NGy0 == NGy1, 'GY is not a square adjacency matrix'

    # permute Y if rand_seed > 1, (this should only be used when testing on mock data)
    #wordsY.permuteFirstWords(perm.randperm(perm.ID(Ny)))
    #MU.printMatching(wordsX.words, wordsY.words, perm.ID(Ny))
    return wordsX, wordsY, seed_list


def parseOptions():
    # parse cmdline arguments
    parser = OptionParser()
    # general setting
    parser.add_option('-e', '--exp_id', dest='exp_id', type="int", action='store', default=1001)
    parser.add_option('-l', '--lexicon', dest='filename_lexicon', action='store', default=None)    
    parser.add_option('-p', '--pickled', dest='pickled', type="int", action='store', default=1)
    parser.add_option('--useContextFeatures', dest='useContextFeatures', type="int", action='store', default=1)
    parser.add_option('--useOrthoFeatures', dest='useOrthoFeatures', type="int", action='store', default=1)
    # mcca setting
    parser.add_option('-z', '--step_size', dest='step_size', type="int", action='store', default=150)
    parser.add_option('-T', '--iterations', dest='T', type="int", action='store', default=10)
    parser.add_option('-w', '--weight_type', dest='weight_type', action='store', default='inner')    
    parser.add_option('-t', '--tau', dest='tau', type="float", action='store', default=0.001)  # CCA regularizer
    parser.add_option('--eta', dest='eta', type="float", action='store', default=0.001)
    parser.add_option('--norm_proj', dest='normalize_projections', type="int", action='store', default=0)
    parser.add_option('--covar_type', dest='covar_type', type="string", action='store', default=None)
    parser.add_option('--projection_type', dest='projection_type', type="string", action='store', default=None)
    # graph settings
    parser.add_option('-K', '--K', dest='K', type="int", action='store', default=0)
    parser.add_option('--alpha', dest='alpha', type="float", action='store', default=0)
    parser.add_option('--GX', dest='filename_graphX', action='store', default=None)
    parser.add_option('--GY', dest='filename_graphY', action='store', default=None)
    # mock related options
    parser.add_option('--noise_level', dest='noise_level', type="float", action='store', default=0.0)
    parser.add_option('--record', dest='record', type="int", action='store', default=None)
    (options, args) = parser.parse_args()
    
    options.is_mock = 'mock' in (sys.argv[1])
    options.gold_lex = None
    
    # post processing of arguments
    
    # graph related options
    if options.filename_graphX is None:
        options.alpha = 0

    options.title = "wt={}, K={}, alpha={}".format(options.weight_type, options.K, options.alpha)

    return options
    

if __name__ == '__main__':
    # cmd line args
    filename_wordsX = sys.argv[1]
    filename_wordsY = sys.argv[2]
    filename_seed = sys.argv[3]
    options = parseOptions()
    # read input files
    wordsX, wordsY, seed_list = readInput(options, filename_wordsX, filename_wordsY, filename_seed)
    N = len(wordsX.words)
    options.matchingFilename = 'results/matching_N=%d_expid=%d_alpha=%2.2f_T=%d.txt' % (N, options.exp_id, options.alpha, options.T)
    NSeed = len(seed_list.X)
    if options.filename_lexicon is not None:
        lex = BU.readLexicon(options.filename_lexicon)
        (gold_lex, times) = BU.filterLexicon(lex, wordsX.words[:-NSeed], wordsY.words[:-NSeed])
        options.gold_lex = gold_lex
        print "Gold lexicon contains", len(gold_lex), 'pairs.'
    else:
        options.gold_lex = None
        print colored("WARNING: No gold lexicon", 'red')

    print >> sys.stderr, "==============#########=========="
    print >> sys.stderr, "Starting mCCA:"
    print >> sys.stderr, NSeed, "seed pairs:", zip(seed_list.X, seed_list.Y)
    (wordsX, wordsY, edge_cost, cost) = mcca(options, wordsX, wordsY, seed_list)
    log(0, 'hamming distance:', perm.hamming(wordsX.words, wordsY.words))
    bell()