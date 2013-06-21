from common import *
import IO
import MatchingUtil as MU
import CCAUtil as CU


def find_matching(options, X, Y):
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
    (N, D) = X.features.shape
    M = N - options.seed_start  # the first M entries can be permuted. The rest are fixed

    sigma = Struct()  # holds the cumulative permutations applied on X and Y
    sigma.X = perm.ID(M)
    sigma.Y = perm.ID(M)
    fixed_point = False
    for t in range(0, options.T):
        options.t = t
        Nt = M - options.step_size*t

        # STEP 1: compute CCA model on the well matched portion of the matching (which includes the fixed seed)
        fixedX = X.features[Nt:, :]
        fixedY = Y.features[Nt:, :]
        cca_model = CU.learn(fixedX, fixedY, options.tau)
        # STEP 2: compute CCA representation of all samples
        Z = CU.project(cca_model, X.features, Y.features)
        # STEP 3: compute weight matrix and run matching (approximate) algorithm
        W = MU.makeWeights(options, Z.X, Z.Y)
        (cost, pi_t, edge_cost) = MU.ApproxMatch(W[:M, :M])
        # STEP 4: sort the words, such that the best matches are at the end.
        # note that pi_t is of length M < N and that
        (sorted_edge_cost, I) = perm.sort(edge_cost, reverse=True)
        sorted_edge_cost = np.concatenate((sorted_edge_cost, np.zeros(N-M)))

        if perm.isID(pi_t):  # the best permutation is the identity
            fixed_point = True
        else:
            X = MU.permuteFirstWords(X, I)
            Y = MU.permuteFirstWords(Y, pi_t[I])
            sigma.X = sigma.X[I]  # accumulate the changes from the ID
            sigma.Y = sigma.Y[I]  # accumulate the changes from the ID
            # END OF ITERATION: output Matching
        MU.printMatching(X, Y, sorted_edge_cost)
        print '----------\n'
        if fixed_point:
            break

    # either we reached the maximum number of iterations, or a fixed point

    print 'Stopped after, ', t, 'iterations. Fixed point=', fixed_point

    return X, Y, sigma, cost


def setupFeatures(options, X):
    #logFr = np.log(X.freq)
    L = strings.strlen(X.words)
    X.features = normalize_rows(X.features)
    # TODO: should be add logFr and L ?
    return X


def mcca(wordsX, wordsY, seedsX, seedsY, options):
    # (N, D) = wordsX.features.shape
    seedsX = setupFeatures(options, seedsX)
    seedsY = setupFeatures(options, seedsY)
    wordsX = setupFeatures(options, wordsX)
    wordsY = setupFeatures(options, wordsY)

    # NOTE: this code was used when initializing the seed by the edit-distance permutation.
    # edit_dist_options = Options()
    # edit_dist_options.exp_id = -1  # Edit distance exp_id is -1.
    # (ed_pi, ed_edge_cost) = IO.readMatching(edit_dist_options, wordsX.words, wordsY.words)
    # # setup the initial matching according to the edge_cost of edit_distance
    # #I = sort_and_map(ed_edge_cost, options.seed_start)
    # (dummy, I) = perm.sort(ed_edge_cost, reverse=True)
    # wordsX = MU.permuteFirstWords(wordsX, I)
    # wordsY = MU.permuteFirstWords(wordsY, ed_pi[I])
    # # At this point X and Y should be aligned according to the seed permutation.
    # # however, it is not really fair to use I as is, if initialized by the edit_distance,
    # # since it will match much more than the seed. So we randomly permute the non-seed entries of Y
    # Nfirst = N - options.seed_start
    # if Nfirst < 0:
    #     print 'ERROR: Nfirst is negative', Nfirst
    #     exit()
    # J = perm.randperm(xrange(Nfirst))
    # wordsY = MU.permuteFirstWords(wordsY, J)

    concatX = Words.concat(wordsX, seedsX)
    concatY = Words.concat(wordsY, seedsY)

    (newX, newY, sigma, cost) = find_matching(options, concatX, concatY)
    return newX, newY, sigma, cost


def readInput():
    # parse arguments
    wordsFilenameX = (sys.argv[1])
    wordsFilenameY = (sys.argv[2])
    seedsFilenameX = (sys.argv[3])
    seedsFilenameY = (sys.argv[4])
    if len(sys.argv) < 6:  # use a different random seed for numpy
        rand_seed = 1
    else:
        rand_seed = int(sys.argv[5])
    np.random.seed(rand_seed)

    # load data files
    #fileX = '../SCRIPTS/matlab/Jun10_en.txt'
    #fileY = '../SCRIPTS/matlab/Jun10_es.txt'
    wordsX = IO.readWords(wordsFilenameX)
    wordsY = IO.readWords(wordsFilenameY)
    seedsX = IO.readWords(seedsFilenameX)
    seedsY = IO.readWords(seedsFilenameY)

    # assert sizes are correct
    Nx = len(wordsX.words)
    Ny = len(wordsY.words)
    if Nx != Ny:
        print 'Number of words must be the same', Nx, Ny
    else:
        print Nx, 'words loaded.'

    NSx = len(seedsX.words)
    NSy = len(seedsY.words)
    if NSx != NSy:
        print 'Number of seed words must be the same', NSx, NSy
    else:
        print NSx, 'seed words loaded.'

    # permute Y if rand_seed > 1, (this is used when testing on synthetic data)
    if rand_seed > 1:
        print 'here'
        pi = perm.randperm(xrange(Ny))
        wordsY = MU.permuteFirstWords(wordsY, pi)
    MU.printMatching(wordsX, wordsY, perm.ID(Ny))

    # set params
    options = Options()
    options.exp_id = 1000
    options.seed_start = NSx
    options.step_size = 10
    options.tau = 0.001
    options.T = 1
    options.M = 0  # 0 = no graphs
    options.weight_type = 'inner'
    return wordsX, wordsY, seedsX, seedsY, options

if __name__ == '__main__':
    wordsX, wordsY, seedsX, seedsY, options = readInput()
    (wordsX, wordsY, sigma, cost) = mcca(wordsX, wordsY, seedsX, seedsY, options)

