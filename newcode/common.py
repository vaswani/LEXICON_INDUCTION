__author__ = 'Tomer'
import sys
from collections import defaultdict
import scipy.spatial.distance
import numpy as np


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


def invertDict(D):
    return {D[key]: key for key in D}


def submatrix(A, rows, cols):  # this is quite slow for some reason, maybe since a new array is allocated.
    rows = asVector(rows)
    cols = asVector(cols)
    I = np.ix_(rows, cols)
    return A[I]


def normalize_rows(V, p=2):  # assumes V is a numpy array!
    Z = norm_rows(V, p)
    V = (V.T / Z).T
    return V


def normsqr_rows(V):
    if isinstance(V, np.matrix):
        V = V.A
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


def max(v):
    v = asVector(v)
    i = np.argmax(v)
    return v[i], i


def min(v):
    v = asVector(v)
    i = np.argmin(v)
    return v[i], i


def asVector(v):
    v = np.squeeze(np.asarray(v))
    if len(v.shape) == 0:
        v = np.array([v]) # scalars have shape 0 (e.g. array(0))
    assert len(v.shape) == 1
    return v


def sub2ind(dims, X, Y=None, Z=None):
    X = asVector(X)
    Y = asVector(Y)

    if Z is None:
        V = [X, Y]
    else:
        Z = asVector(Z)
        V = [X, Y, Z]
    return np.ravel_multi_index(V, dims)


def norm(v, p='fro'):
    return np.linalg.norm(v, p)


def dist(X, Y=None, metric='euclidean'):
    if Y is None:
        Y = X
    return scipy.spatial.distance.cdist(X, Y, metric)


def isEmpty(X):
    if isinstance(X, np.ndarray) or isinstance(X, np.matrix):
        return X.shape[0] == 0
    if isinstance(X, list):
        return len(X) == 0


def log(level, *args):
    global verbosity
    if verbosity > level:
        for i, s in enumerate(args):
            sys.stderr.write(str(s))
            sys.stderr.write(' ')
        sys.stderr.write('\n')


#%%%%%%%%%%%% CONTEXT = make two runs of mock the same. make pock work.
# def record_or_compare(name, V, is_record):
#     filename = '/tmp/matrices/roc_' + name
#     if is_record == 1:
#         IO.writeNumpyArray(filename, V)
#     elif is_record == 2:
#         filename += '.npy'
#         U = IO.readNumpyArray(filename)
#         tol = 1e-5
#         if np.allclose(U, V, tol):
#             print "'%s' passed"
#         else:
#             print "'%s' does not match. (tol=%2.2f)" % (name, tol)
#             print U, V, norm(U), norm(V), U.shape, V.shape
#             assert 1==0


# a module level function, instead of lambda.
# lambda functions cannot be pickled
def dd():
    return defaultdict(int)


def bell():
    print('\a')


if __name__ == '__main__':
    print 'common: 1.233333'
# general common options
np.set_printoptions(precision=4)
verbosity = 1000


