__author__ = 'Tomer'

from common import *
import kernels


def getMatching(X, Y, pi, edge_cost):
    M = np.vstack((X, Y[pi]))
    sigma = np.argsort(edge_cost)
    M = M[:, sigma]
    return M


def makeWeights(options, X, Y, GX=None, GY=None):
    X = np.mat(X)
    Y = np.mat(Y)
    if GX is None or GY is None:
        GX = [0]
        GY = [0]
    GX = np.mat(GX)
    GY = np.mat(GY)
    if options.weight_type == 'inner':
        U = X*Y.T  # linear kernel
        W = U
        for m in range(1, options.M+1):
            W += (options.alpha ** m) * ((GX ** m) * U * (GY.T ** m))
        W = np.max(W) - W
    elif options.weight_type == 'dist':
        U = kernels.dist(X, Y)
        W = U
        for m in range(1, options.M+1):
            W += (options.alpha ** m) * kernels.dist((GX ** m) * X, (GY ** m) * Y)
    else:
        W = []
    return W


# Approximate minimum weighted matching on the input C
# returns the cost = \sum C[ i, pi[i] ]
def ApproxMatch(C):
    N = C.shape[1]
    left = np.zeros((N, 1))
    right = np.zeros((N, 1))
    I = np.tile(np.arange(N), [N, 1])
    J = I.T
    I = I.flatten()
    J = J.flatten()
    sigma = np.argsort(C.flatten())
    pi = np.zeros(N, dtype=np.int)
    edge_cost = np.zeros(N)

    for element in sigma.flat:
        i = I[element]
        j = J[element]
        if (left[i] == 0) and (right[j] == 0):
            pi[j] = i
            edge_cost[j] = C[j, i]
            left[i] = 1
            right[j] = 1
    cost = np.sum(edge_cost)

    return cost, pi, edge_cost


# this method permutes all the fields of X according to pi.
# if pi is shorter than X.words, than only the first entries are permuted,
# and the last remain in their position.
def permuteFirstWords(X, pi):
    id = perm.ID(len(pi))
    X.words[id] = X.words[pi]
    X.features[id, :] = X.features[pi, :]
    X.freq[id] = X.freq[pi]
    #X.G[id,id] = X.G[pi, pi]
    return X


def printMatching(X, Y, sorted_edge_cost):
    N = len(sorted_edge_cost)
    for n in xrange(N):
        weight = sorted_edge_cost[n]
        print '{:>12} {:>12} {:>12} {:>6}'.format(n, X.words[n], Y.words[n], weight)
        #print n, X.words[n], Y.words[n], weight

if __name__ == '__main__':  # test
    # test
    C = np.matrix('1 -1 1 1; 1 1 1 0; 0 1 1 1 ;1 -3 -2 1')
    (cost, pi, edge_cost) = ApproxMatch(C)
    print C
    print "cost:", cost
    print pi
    print edge_cost

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

    options = Options
    options.weight_type = 'dist'
    options.M = 1
    options.alpha = 0.3
    W = makeWeights(options, X, Y, GX, GY)
    print 'W shape: ', W.shape
    print W
