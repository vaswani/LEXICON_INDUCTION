import scipy.sparse as sps
import numpy as np
import time
from sklearn.preprocessing import normalize


# given DD - a dictionary of dictionaries with format DD[string][feature] = numeric value
# enables computing a linear kernel efficiently, by wrapping scipy's sparse matrix.
class MSK:
    def __init__(self, DD, strings, features):
        # init fields
        self.strings = {string: i for (i, string) in enumerate(strings)}
        self.features = {k: j for (j, k) in enumerate(features)}
        self.K = []

        # init sparse array
        n = len(strings)
        d = len(self.features)
        A = sps.lil_matrix((n, d), dtype=np.float64)

        # convert DD to sparse matrix
        for s in self.strings:
            i = self.strings[s]
            v = DD[s]
            for k in v:
                j = self.features[k]
                A[i, j] = v[k]  # i,j are the string,feature indices, k is the feature index in v

        self.M = A

    def makeLinearKernel(self):
        return LinearKernel(self.strings, self.M)

    def normalize(self, norm='l2'):
        self.M = normalize(self.M, norm, axis=1)
        return self


class LinearKernel:
    def __init__(self, strings, M):
        self.strings = strings
        self.K = (M * M.T).todense()

    def materializeKernel(self, strings1=None, strings2=None):
        if strings1 is None:
            strings1 = self.strings
        pi_i = [self.strings[s] for s in strings1]

        if strings2 is None:
            pi_j = pi_i
        else:
            pi_j = [self.strings[s] for s in strings2]
        return self.K[np.ix_(pi_i, pi_j)].tolist()  # list has a more efficient access


if __name__ == '__main__':  # test
    np.random.seed(1)
    N = 400
    D = 800

    DD = dict()
    rangeD = range(D)
    rangeN = range(N)

    for i in xrange(N):
        DD[i] = dict()
        S = np.random.permutation(rangeD)
        S = S[:D/4]
        for j in S:
            DD[i][j] = np.random.randn(1)[0]
    print "finished constructing."
    t = time.time()
    msk = MSK(DD, rangeN, rangeD)
    msk.computeKernel()

    G = msk.materializeKernel(rangeN, rangeN)
    print 'elapsed', time.time() - t
    print G
    print G.__class__
