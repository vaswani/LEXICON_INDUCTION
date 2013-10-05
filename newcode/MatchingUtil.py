__author__ = 'Tomer'

import common
import numpy as np
import BilexiconUtil as BU
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from cyMatching import cy_ApproxMatch, cy_min_submatrix, cy_min_submatrix2, cy_getGraphMinDist
import munkres  # https://github.com/jfrelinger/cython-munkres-wrapper
import LAPJV
import IO


def getMatching(X, Y, pi, edge_cost):
    M = np.vstack((X, Y[pi]))
    sigma = np.argsort(edge_cost)
    M = M[:, sigma]
    return M


def saveWUZ(U, W, Z, options):
    S = (W, U, Z)
    IO.pickle('tmp/S_t=' + str(options.t) + '.txt', S)


def makeWeights(options, X, Y, GX, GY):
    # hack to compute scores using normalized projections
    if options.normalize_projections == 1:
        X = common.normalize_rows(X, 2)  # note that normalize_rows works on arrays, not matrices.
        Y = common.normalize_rows(Y, 2)

    X = np.mat(X)
    Y = np.mat(Y)

    if GX is not None or GY is not None:
        GX = np.mat(GX)
        GY = np.mat(GY)
    Z = np.mat(0)
    if options.weight_type == 'inner':
        U = X*Y.T  # linear kernel
        if options.alpha > 0:  # TODO: add higher order graphs
            Z = GX * U * GY.T
            W = (1-options.alpha)*U + options.alpha*Z
        else:
            W = U
        W = np.max(W) - W
    elif options.weight_type == 'dist':
        U = common.dist(X, Y)
        if options.alpha > 0:  # TODO: add higher order graph
            Z = common.dist(GX * X, GY * Y)
            W = (1-options.alpha)*U + options.alpha*Z
        else:
            W = U
    elif options.weight_type == 'sqrdist':
        U = common.dist(X, Y, metric='sqeuclidean')
        if options.alpha > 0:  # TODO: add higher order graph
            Z = common.dist(GX * X, GY * Y, metric='sqeuclidean')
            W = (1-options.alpha)*U + options.alpha*Z
        else:
            W = U
    elif options.weight_type == 'graph_min_dist':
        U = common.dist(X, Y)
        GX = np.array(GX)
        GY = np.array(GY)
        if options.alpha > 0:  # TODO: add higher order graph
            Z, IX, IY = cy_getGraphMinDist(GX, GY, U)
        W = (1-options.alpha)*U + options.alpha*Z
    else:
        W = []

    saveWUZ(U, W, Z, options)
    #print 'norm(U) = ', np.linalg.norm(U, 2), '| norm(Z) = ', np.linalg.norm(Z, 2)
    return W, U, Z


def fast_approxMatch(C):
    if isinstance(C, np.matrix):
        C = np.array(C, dtype=np.double)
    return cy_ApproxMatch(C)


# Approximate minimum weighted matching on the input C
# returns the cost = \sum C[ i, pi[i] ]
def approxMatch(C):
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


def exactMatch(A, resolution=1e-8, cython=True):
    # via tha LAPJV algorithm
    A = np.array(A)
    if cython:
        [pi, cost, v, u, costMat] = LAPJV.fast_lapjv(A, resolution)
    else:
        [pi, cost, v, u, costMat] = LAPJV.lapjv(A, resolution)
    N = len(pi)
    edge_cost = A[xrange(N), pi]
    return cost, np.array(pi), np.array(edge_cost)


def printMatching(wordsX, wordsY, sorted_edge_cost, lex=None):
    print toString(wordsX, wordsY, sorted_edge_cost, lex)


def toString(wordsX, wordsY, sorted_edge_cost, lex=None):
    N = len(sorted_edge_cost)
    s = ''
    for n in xrange(N):
        weight = sorted_edge_cost[n]
        source_word = wordsX[n]
        target_word = wordsY[n]
        if lex is not None:
            matched = BU.is_valid_match(lex, source_word, target_word)
            matched = "correct" if matched else " wrong "
        else:
            matched = source_word == target_word
        #common.log(200, '{},{},{},{:>6},{:>12}'.format(source_word, target_word, matched, weight, n))
        #common.log(200, '{} - {:>12}) {:>12} {:>12} {:>6}'.format(matched, n, source_word, target_word, weight))
        # s += '{} - {:>4}),{:>10},{:>10},{:>4}'.format(matched, n, source_word, target_word, weight)
        s += '{} - {:>10},{:>10}'.format(matched, source_word, target_word)
        s += '\n'
    return s



def test1():
    # test
    C = np.matrix('1 -1 1 1; 1 1 1 0; 0 1 1 1 ;1 -3 -2 1')
    (cost, pi, edge_cost) = approxMatch(C)
    (cy_cost, cy_pi, cy_edge_cost) = fast_approxMatch(C)
    print C
    print "cost:", cost, cy_cost-cost
    print pi, pi-cy_pi
    print edge_cost, cy_edge_cost - edge_cost

    D = 2000
    np.random.seed(1)
    C = common.randn((D, D))
    import cProfile
    cProfile.runctx('(cost, pi, edge_cost) = approxMatch(C)', globals(), locals())
    cProfile.runctx('(cy_cost0, cy_pi0, cy_edge_cost0) = cy_ApproxMatch(C)', globals(), locals())
    (cost, pi, edge_cost) = approxMatch(C)
    (cy_cost0, cy_pi0, cy_edge_cost0) = cy_ApproxMatch(C)

    assert np.linalg.norm(cy_cost0-cost) == 0
    assert np.linalg.norm(pi-cy_pi0) == 0
    assert np.linalg.norm(cy_edge_cost0 - edge_cost) == 0

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
    options.normalize_projections = 1
    W, U, Z = makeWeights(options, X, Y, GX, GY)
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


def test2():
    N = 3000
    D = 50
    X = common.randn((N, D))
    Y = common.randn((N, D))
    U = common.dist(X, Y)

    import cProfile
    #cProfile.runctx('(cost, pi, edge_cost) = approxMatch(U)', globals(), locals())
    cProfile.runctx('(cy_cost0, cy_pi0, cy_edge_cost0) = cy_ApproxMatch(U)', globals(), locals())
    print cy_cost0


if __name__ == '__main__':  # test
    test1()
