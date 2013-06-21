__author__ = 'Tomer'
import numpy as np
import csv
import sys
import perm
import strings
import scipy.spatial.distance


# Utility classes
class Words:
    def __init__(self):
        self.words = []
        self.freq = []
        self.features = []

    def toNP(self):
        self.words = np.array(self.words)
        self.freq = np.array(self.freq)
        self.features = np.array(self.features)

    @staticmethod
    def concat(A, B):
        C = Words()
        C.words = np.append(A.words, B.words)
        C.freq =  np.append(A.freq, B.freq)
        C.features = np.vstack((A.features, B.features))
        print C.features.shape
        return C



class Options:
    pass


class Struct:  # general structs
    pass


def randn(S):
    A = np.random.randn(*S)
    return A


def randi(N, S):
    return np.random.randint(N, size=S)


def diag(v, d1=None, d2=None):
    N = len(v)
    if d1 is None or d2 is None:
        d1 = d2 = N
    S = np.zeros((d1, d2))
    S[:N, :N] = np.diag(v)
    return S


def normalize_rows(V): # assumes V is a numpy array!
    Z = norm_rows(V)
    V = (V.T / Z).T
    return V


def normsqr_rows(V):
    return (V*V).sum(1)


def norm_rows(V):
    Z = np.sqrt(normsqr_rows(V))
    return Z


def uniq(seq):  # stable version of unique (stable in the sense that it preserves order).
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


def knn_search(x, D, K):
    """ find K nearest neighbours of data among D """
    ndata = D.shape[1]
    K = K if K < ndata else ndata
    # euclidean distances from the other points
    sqd = dist(x, D)
    idx = np.argsort(sqd)  # sorting
    # return the indexes of K nearest neighbours
    return sqd[:,:K], idx[:,:K]


def dist(X, Y):
    return scipy.spatial.distance.cdist(X, Y)


def knngraph(X, k):
    (N, D) = X.shape
    sqd, idx = knn_search(X, X, k)
    G = np.zeros((N, N))
    for j in xrange(0, k):
        Ij = idx[:, j]
        G[xrange(N), Ij] = 1  #sqd[:, j]

    return G, idx


np.set_printoptions(precision=2)
if __name__ == '__main__':
    A = np.array([[0, 0, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 0.5, 0], [1, 0.5, 0.5, 0, 0]])
    B = [[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0]]
    idx = knn_search(B, A, 3)

    (G, idx) = knngraph(A, 2)

    print G


