import scipy.sparse as sps
import numpy as np
import time
from sklearn.preprocessing import normalize
import common


# given DD - a dictionary of dictionaries with format DD[string][feature] = numeric value
# enables computing a linear kernel efficiently, by wrapping scipy's sparse matrix.
class MSK:
    def __init__(self, DD, strings, features):
        # init fields
        self.strings = {string: i for (i, string) in enumerate(strings)}
        self.features = {f: j for (j, f) in enumerate(features)}
        self.reverseStrings = common.invertDict(self.strings)
        self.reverseFeatures = common.invertDict(self.features)

        # init sparse array
        n = len(strings)
        d = len(self.features)
        A = sps.lil_matrix((n, d), dtype=np.float64)

        # convert DD to sparse matrix
        if DD is not None:
            for s in self.strings:
                i = self.strings[s]
                v = DD[s]
                for k in v:
                    j = self.features[k]
                    A[i, j] = v[k]  # i,j are the string,feature indices, k is the feature index in v

        self.M = A

    def shape(self):
        return self.M.shape

    def makeLinearKernel(self):
        return LinearKernel(self.strings, self.M)

    def normalize(self, norm='l2'):
        self.M = normalize(self.M, norm, axis=1)  # normalize rows
        return self

    def log(self, offset=0):
        self.M = sps.lil_matrix(np.log(self.M.todense() + offset))
        return self

    def materialize(self, strings=None, features=None):
        if strings is None:
            strings = self.strings
        if features is None:
            features = self.features

        pi_i = [self.strings[s] for s in strings]
        pi_j = [self.features[f] for f in features]
        A = common.submatrix(self.M, pi_i, pi_j)
        return A  # list has a more efficient cell access


    def getNonZeroFeatures(self, s):
        i = self.strings[s]
        dict = {}
        for f in self.features:
            j = self.features[f]
            if self.M[i, j] != 0:
                dict[f] = self.M[i, j]

        return dict


class LinearKernel:
    def __init__(self, strings, M):
        self.strings = strings
        self.K = (M * M.T).todense()

    def materialize(self, strings1=None, strings2=None):
        if strings1 is None:
            strings1 = self.strings
        pi_i = [self.strings[s] for s in strings1]

        if strings2 is None:
            pi_j = pi_i
        else:
            pi_j = [self.strings[s] for s in strings2]

        A = self.K[np.ix_(pi_i, pi_j)]
        return A  # list has a more efficient cell access


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
