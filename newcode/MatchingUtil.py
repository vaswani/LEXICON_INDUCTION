__author__ = 'Tomer'

import common
import numpy as np
import BilexiconUtil as BU


def getMatching(X, Y, pi, edge_cost):
    M = np.vstack((X, Y[pi]))
    sigma = np.argsort(edge_cost)
    M = M[:, sigma]
    return M


def makeWeights(options, X, Y, GX, GY):
    # hack to compute scores using normalized projections
    # if options.normalize_projections == 1:
    #     X = common.normalize_rows(X)  # note that normalize_rows works on arrays, not matrices.
    #     Y = common.normalize_rows(Y)

    X = np.mat(X)
    Y = np.mat(Y)

    if GX is not None or GY is not None:
        GX = np.mat(GX)
        GY = np.mat(GY)
    if options.weight_type == 'inner':
        U = X*Y.T  # linear kernel
        if options.K > 0:  # TODO: add higher order graphs
            Z = GX * U * GY.T
            W = options.alpha*Z + (1-options.alpha)*U
        else:
            W = U
        # for m in range(1, options.M+1):
        #     Z += (options.alpha ** m) * ((GX ** m) * U * (GY.T ** m))
        # TODO: context: why is the norm of X so small in our case?
        W = np.max(W) - W
    elif options.weight_type == 'dist':
        U = common.dist(X, Y)
        if options.K > 0:  # TODO: add higher order graph
            Z = common.dist(GX * X, GY * Y)
            W = options.alpha*Z + (1-options.alpha)*U
        else:
            W = U
    else:
        W = []
    return W  # , U, Z


# Approximate minimum weighted matching on the input C
# returns the cost = \sum C[ i, pi[i] ]
def ApproxMatch(C):
    N = C.shape[1]
    left = [0] * N
    right = [0] * N
    I = np.tile(np.arange(N), [N, 1])
    J = I.T
    I = I.flatten()
    J = J.flatten()
    sigma = np.argsort(C.flatten())
    #sigma = sigma.flat
    pi = [0] * N
    edge_cost = [0] * N

    for element in sigma.flat:
        #element = sigma[m]
        i = I[element]
        j = J[element]
        if left[i] == 0 and right[j] == 0:
            pi[j] = i
            edge_cost[j] = C[j, i]
            left[i] = 1
            right[j] = 1
    cost = np.sum(edge_cost)
    pi = np.array(pi)
    edge_cost = np.array(edge_cost)
    #print cost
    return cost, pi, edge_cost


def printMatching(wordsX, wordsY, sorted_edge_cost, lex=None):
    N = len(sorted_edge_cost)
    for n in xrange(N):
        weight = sorted_edge_cost[n]
        source_word = wordsX[n]
        target_word = wordsY[n]
        if lex is not None:
            matched = BU.is_valid_match(lex, source_word, target_word)
            matched = "correct" if matched else "wrong"
        else:
            matched = source_word == target_word
        common.log(200, '{} - {:>12}) {:>12} {:>12} {:>6}'.format(matched, n, source_word, target_word, weight))


if __name__ == '__main__':  # test
    # test
    C = np.matrix('1 -1 1 1; 1 1 1 0; 0 1 1 1 ;1 -3 -2 1')
    (cost, pi, edge_cost) = ApproxMatch(C)
    print C
    print "cost:", cost
    print pi
    print edge_cost

    D = 1000
    np.random.seed(1)
    C = common.randn((D, D))
    import cProfile
    cProfile.runctx('ApproxMatch(C)', globals(), locals())

    X = np.array([[ 0.17034787, -1.11852005,  2.3723583 ],
       [ 0.40587496, -0.71610151,  0.24853477],
       [ 0.28721089, -1.62157422,  0.33806607],
       [ 0.88027416,  0.30368357, -1.15908568],
       [-0.50893908, -0.61650787, -1.10849268]])

    Y = np.array([[ 0.49316125, -1.6459511 ,  0.03514289],
       [ 0.16211477,  0.41796482, -0.19066103],
       [-0.68095143,  0.48752118,  1.08161025],
       [-0.44147856,  0.43943179,  1.05625251]])

    GX = np.array([[0, 1, 1, 0, 1],
       [0, 1, 1, 1, 0],
       [0, 1, 0, 1, 1],
       [0, 1, 1, 0, 0],
       [1, 0, 0, 0, 1]])

    GY = np.array([[0, 1, 1, 0],
       [0, 1, 0, 0],
       [1, 0, 1, 1],
       [0, 0, 0, 1]])

    options = common.Struct()
    options.weight_type = 'dist'
    options.K = 1  # K is not really 1 here, but a non-zero value is required.
    options.alpha = 0.3
    W = makeWeights(options, X, Y, GX, GY)
    print 'W shape: ', W.shape
    print W
# output should be
#  [[ 1 -1  1  1]
#  [ 1  1  1  0]
#  [ 0  1  1  1]
#  [ 1 -3 -2  1]]
# cost: -2
# [2 3 0 1]
# [ 1  0  0 -3]
# W shape:  (5, 4)
# [[ 2.9433  3.1083  2.6389  2.6288]
#  [ 1.8387  1.7247  2.4035  2.226 ]
#  [ 1.375   2.2009  3.0356  2.6965]
#  [ 2.6659  1.7213  2.7032  2.7192]
#  [ 2.0872  1.8698  2.1393  2.3427]]
