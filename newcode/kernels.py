from common import *
import scipy.spatial.distance


def inner(X, Y):
    K = X.dot(Y.T)
    return K


def dist2(X, Y):
    A = dist(X, Y)
    return A * A


def dist(X, Y):
    return scipy.spatial.distance.cdist(X, Y)


if __name__ == '__main__':
    X = randn((5, 3))
    Y = randn((6, 3))

    K = inner(X, Y)
    D = dist(X, Y)
    print K
    print D


