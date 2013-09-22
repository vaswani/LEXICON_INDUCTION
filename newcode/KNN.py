import numpy as np
from common import dist


def search(unseenX, trainX, k, want_self=False):
    """ find K nearest neighbours of unseenX within trainX """
    N = trainX.shape[1]  # samples are rows
    k = np.min([k, N])
    # compute euclidean distances from the other points
    sqd = dist(unseenX, trainX)
    return find_knn(sqd, k, want_self)


def find_knn(DIST, k, want_self=False):
    idx = np.argsort(DIST)  # sorting
    # return the distances and indexes of K nearest neighbours
    if want_self:
        return idx[:, :k]
    else:
        return idx[:, 1:(k+1)]  # be careful, the first argument is not sorted!