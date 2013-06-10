from common import *
import scipy.spatial.distance


def inner(X, Y):
    K = X.dot(Y.T)
    return K


def dist2(X, Y):
    return scipy.spatial.distance.cdist(X, Y)


def dist(X, Y):
    return np.sqrt(dist2(X, Y))


def innerG(X, Y, GX, GY):
    return inner(concatFG(X, Y, GX, GY))


def distG(X, Y, GX, GY):
    return dist(concatFG(X, Y, GX, GY))


def concatFG(X, Y, GX, GY):  # concats the (F)eatures and the (G)raphs
    ZX = np.hstack((X, GX))
    ZY = np.hstack((Y, GY))
    return (ZX, ZY)



if __name__ == '__main__':
    X = randn(5, 3)
    Y = randn(6, 3)
    K = inner(X, Y)
    D = dist(X, Y)
    print K
    print D


