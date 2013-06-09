__author__ = 'Tomer'

from common import *

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
    cost = 0
    pi = np.zeros([N,1])
    M = sigma.size;
    for element in sigma.flat:
        i = I[element]
        j = J[element]
        if left[i]==0 and right[j]==0:
            pi[j] = i
            cost = cost + C[j,i]
            left[i] = 1
            right[j] = 1
    return cost, pi

if __name__ == '__main__':
    # test
    C = np.matrix('1 0 1 1; 1 1 1 0; 0 1 1 1 ;1 1 0 1')
    cost, pi = ApproxMatch(C)
    print C
    print "cost:", cost
    print pi

