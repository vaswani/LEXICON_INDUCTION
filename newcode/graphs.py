from common import *


def knngraph(X, k):
    (N, D) = X.shape
    sqd, idx = knn_search(X, X, k)
    G = np.zeros((N, N))
    for j in xrange(0, k):
        Ij = idx[:, j]
        G[xrange(N), Ij] = 1  #sqd[:, j]

    return G, idx


if __name__ == '__main__':
    A = np.array([[0, 0, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 0.5, 0], [1, 0.5, 0.5, 0, 0]])
    B = [[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0]]
    idx = knn_search(B, A, 3)

    (G, idx) = knngraph(A, 2)

    print G
