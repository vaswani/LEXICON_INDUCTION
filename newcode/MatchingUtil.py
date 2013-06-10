__author__ = 'Tomer'

from common import *


def getMatching(X, Y, pi):
    return np.vstack((X, Y[pi]))


def permuteWords(X, pi)
    X.words = X.words[]


# Approximate minimum weighted matching on the input C
# returns the cost = \sum C[ i, pi[i] ]
def ApproxMatch(C):
    N = C.shape[1]
    left = np.zeros((N,1))
    right = np.zeros((N,1))
    I = np.tile(np.arange(N), [N, 1])
    J = I.T
    I = I.flatten()
    J = J.flatten()
    sigma = np.argsort(C.flatten())
    pi = [0] * N
    edge_cost = [0.0] * N
    M = sigma.size;
    for element in sigma.flat:
        i = I[element]
        j = J[element]
        if (left[i] == 0) and (right[j] == 0):
            pi[j] = i
            edge_cost[j] = C[j, i]
            left[i] = 1
            right[j] = 1
    cost = np.sum(edge_cost)
    return (cost, pi, edge_cost)

if __name__ == '__main__':
    # test
    C = np.matrix('1 -1 1 1; 1 1 1 0; 0 1 1 1 ;1 -3 -2 1')
    (cost, pi, edge_cost) = ApproxMatch(C)
    print C
    print "cost:", cost
    print pi
    print edge_cost