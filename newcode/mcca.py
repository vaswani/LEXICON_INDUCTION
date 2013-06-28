from common import *
from words import *
import MatchingUtil as MU
import CCAUtil as CU
import graphs
import IO


def find_matching(options, concatX, concatY):
    # finds a permutation pi that best matches Y to X
    # in optimization procedure works as follows:
    # suppose there are 2000 words to be matched, 100 seed words and step size is 100
    # The seed is stored at the end (so, X[pi[i]] matches Y[i] for i > 2000]
    # at each iteration t we (starting at t=0)
    # 1. compute the CCA on the last 100 + 100*t entries
    # 2. compute the CCA representation of all words
    # 3. perform a matching on the first N=2000 words to get pi_t
    # 4. sort the first 2000 matches in descending order.

    # initially, assume that pi is ID
    (N, D) = concatX.features.shape
    M = N - options.seed_start  # the first M entries can be permuted. The rest are fixed

    sigma = Struct()  # holds the cumulative permutations applied on X and Y
    sigma.X = perm.ID(M)
    sigma.Y = perm.ID(M)
    fixed_point = False
    for t in range(0, options.T):
        options.t = t
        Nt = M - options.step_size*t

        # STEP 1: compute CCA model on the well matched portion of the matching (which includes the fixed seed)
        fixedX = concatX.features[Nt:, :]
        fixedY = concatY.features[Nt:, :]
        cca_model = CU.learn(fixedX, fixedY, options.tau)
        # STEP 2: compute CCA representation of all samples
        Z = CU.project(cca_model, concatX.features, concatY.features)
        # STEP 3: compute weight matrix and run matching (approximate) algorithm
        W = MU.makeWeights(options, Z.X, Z.Y, concatX.G, concatY.G)
        (cost, pi_t, edge_cost) = MU.ApproxMatch(W[:M, :M])
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
        print cost, np.sum(Z.X.A * Z.Y.A)
        WW = MU.makeWeights(options, concatX.features, concatY.features, concatX.G, concatY.G)
        print 'normWW', np.linalg.norm(WW, 2)
        print 'norm GX-GY', np.linalg.norm(concatX.G - concatY.G)
        #MU.printMatching(concatX, concatY, sorted_edge_cost)
        log(100, '----------\n')
        if fixed_point:
            break

    # either we reached the maximum number of iterations, or a fixed point

    log(100,'Stopped after, ', t, 'iterations. Fixed point =', fixed_point)
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

    print np.linalg.norm(concatX.G-concatY.G)


    (newX, newY, sigma, edge_cost, cost) = find_matching(options, concatX, concatY)
    print "done"
    return newX, newY, sigma, edge_cost, cost


def makeOptions(exp_id, seed_length, graph_type=None, M=0, alpha=0, is_mock=False):
    # set params
    options = Options()
    options.exp_id = exp_id
    options.is_mock = is_mock  # mock experiments should have mock in the filename

    options.seed_start = seed_length
    options.step_size = seed_length
    options.tau = 0.001
    options.T = 10

    options.weight_type = 'inner'
    # graph related options
    options.M = M  # 0 = no graphs
    options.alpha = alpha
    options.graph_type = graph_type  # some graphs are dynamic (KNN), some are static.

    return options


def readInput(filename_wordsX, filename_wordsY, filename_seedX, filename_seedY, filename_graphX=None, filename_graphY=None):
    # load data files
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

    if filename_graphX is not None:
        (NGx0, NGx1) = wordsX.G.shape
        (NGy0, NGy1) = wordsY.G.shape
        assert NGx0 == NGx1, 'GX is not an adjacency matrix'
        assert NGy0 == NGy1, 'GY is not an adjacency matrix'
        assert NGx0 == Nx + NSx, 'GX dimensions %d do not match those of X %d' % (NGx0, Nx+NSx)
        assert NGy0 == Ny + NSy, 'GY dimensions %d do not match those of Y %d' % (NGy0, Ny+NSy)

    # permute Y if rand_seed > 1, (this should only be used when testing on mock data)
    MU.printMatching(wordsX, wordsY, perm.ID(Ny))
    return wordsX, wordsY, seedsX, seedsY


if __name__ == '__main__':
    # parse arguments
    filename_wordsX = (sys.argv[1])
    filename_wordsY = (sys.argv[2])
    filename_seedsX = (sys.argv[3])
    filename_seedsY = (sys.argv[4])

    wordsX, wordsY, seedsX, seedsY = readInput(filename_wordsX, filename_wordsY, filename_seedsX, filename_seedsY)
    NSx = len(seedsY.words)
    is_mock = 'mock' in filename_seedsX
    options = makeOptions(is_mock, NSx)

    (wordsX, wordsY, sigma, edge_cost, cost) = mcca(options, wordsX, wordsY, seedsX, seedsY)
    log(0, 'hamming distance:', perm.hamming(wordsX.words, wordsY.words))

