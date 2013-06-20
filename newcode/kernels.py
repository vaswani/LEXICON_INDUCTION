from common import *


def inner(X, Y):
    K = X.dot(Y.T)
    return K


def dist2(X, Y):
    A = dist(X, Y)
    return A * A


if __name__ == '__main__':
    X = randn((5, 3))
    Y = randn((6, 3))

    K = inner(X, Y)
    D = dist(X, Y)
    print K
    print D


