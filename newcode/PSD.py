import numpy as np
import scipy as sp


def isPSD(A, tol=1e-8):
    E, V = sp.linalg.eigh(A)
    return np.all(E > -tol)


def to_square_dist(K):
    # you can always pass the entire kernel and select an off-diagonal block
    s = K.shape
    assert s[0] == s[1]
    N = s[0]
    d = np.mat(np.diag(K)).T
    e = np.mat(np.ones((N, 1)))
    B = d*e.T
    D = -2*K + B + B.T
    return D