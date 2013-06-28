__author__ = 'Tomer'
import matplotlib.pylab as pl
import scipy.spatial.distance
import numpy as np
import strings
import perm
import csv
import sys


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


def normalize_rows(V, p=2):  # assumes V is a numpy array!
    Z = norm_rows(V, p)
    V = (V.T / Z).T
    return V


def normsqr_rows(V):
    return (V*V).sum(1)


def norm_rows(V, p=2):
    if p == 2:
        Z = np.sqrt(normsqr_rows(V))
    if p == 1:
        Z = np.sum(V, 1)  # sum across the columns
    return Z


def uniq(seq):  # stable version of unique (stable in the sense that it preserves order).
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


def isPSD(A, tol=1e-8):
    E, V = scipy.linalg.eigh(A)
    return np.all(E > -tol)


def knn_search(x, D, K):
    """ find K nearest neighbours of data among D """
    N = D.shape[1]
    K = K if K < N else N
    # euclidean distances from the other points
    sqd = dist(x, D)
    idx = np.argsort(sqd)  # sorting
    # return the indexes of K nearest neighbours
    return sqd[:,:K], idx[:,:K]


def norm(v, p='fro'):
    return np.linalg.norm(v, p)


def dist(X, Y):
    return scipy.spatial.distance.cdist(X, Y)


def log(level, *args):
    global verbosity
    if verbosity > level:
        for i, s in enumerate(args):
            sys.stdout.write(str(s))
            sys.stdout.write(' ')
        sys.stdout.write('\n')


if __name__ == '__main__':
    print '1.233333'
# general common options
np.set_printoptions(precision=2)
print 'here'
verbosity = 1000


