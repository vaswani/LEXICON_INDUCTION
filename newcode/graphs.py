from sklearn.preprocessing import normalize
import KNN
import common
import numpy as np
import scipy.sparse as sps
import PSD
from MatrixStringKeys import MSK


def kernel_knn_graph(K, k):
    (N, D) = K.shape
    D = PSD.to_square_dist(K)
    idx = KNN.find_knn(D, k, want_self=False)
    G = create_knn_graph(N, idx, k)
    return G, idx


# makes an NxN graph, from the sequence of K indices given by find_knn
def create_knn_graph(N, idx, k=None):
    if k is None:
        k = idx.shape[1]
    G = sps.lil_matrix((N * N, 1), dtype=np.float64)
    for j in xrange(k):  # assign entries using linear indexing
        Ij = idx[:, j]
        lin_I = common.sub2ind((N, N), xrange(N), Ij.T)
        G[lin_I, 0] = 1
    G = G.reshape((N, N))
    return G


def fromList(shape, coords, weights=None):
    if weights is not None:
        assert len(coords) == len(weights)

    if len(shape) == 1:
        NX = NY = shape[0]
    else:
        NX = shape[0]
        NY = shape[1]

    if weights is None:
        X = [a for (a, b) in coords]
        Y = [b for (a, b) in coords]
        G = sps.lil_matrix((NX * NY, 1), dtype=np.float64)
        lin_I = common.sub2ind((NX, NY), X, Y)
        G[lin_I, 0] = 1
        G = G.reshape((NX, NY))
    else:
        G = sps.lil_matrix((NX, NY), dtype=np.float64)

        for (i, w) in enumerate(weights):
            c = coords[i]
            x = c[0]
            y = c[1]
            G[x, y] = w
    return G


def fromDictOfDict(DD, words_a, words_b):
    return MSK(DD, words_a, words_b)


def knn_graph(X, k):
    (N, D) = X.shape
    idx = KNN.search(X, X, k)
    G = create_knn_graph(N, idx, k)
    return G, idx


def toSymmetricStochastic(G, sym=True, stochastic=True, norm='l1'):
    if sym:
        G = (G + G.T) / 2
    if stochastic:
        G = normalize(G, norm, axis=1)  # make stochastic matrix.
    return G


# def permute(G, pi):
#     (N, _) = G.shape
#     M = len(pi)
#     pi = np.append(pi, np.arange(M, N))
#     G = G[:, pi]
#     G = G[pi, :]
#     return G

if __name__ == '__main__':
    A = np.mat(common.randn((4, 5)))
    K = A * A.T
    D = np.mat(common.dist(A))
    k = 2
    (G, I) = knn_graph(A, k)
    (KG, KI) = kernel_knn_graph(K, k)
    print "G-KG", np.linalg.norm((G-KG).todense())
    print 'G:\n', G.todense()
    print 'D:\n', D