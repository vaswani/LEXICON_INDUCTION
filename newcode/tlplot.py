__author__ = 'Tomer'

import matplotlib.pyplot as plt
import common
import numpy as np


def imshow(M):
    plt.imshow(M)
    show()


def hist(v, bins=10):
    v = common.asVector(v)
    H = plt.hist(v, bins=bins)
    show()


def show():
    plt.show() # TODO: how to make it non-blocking?


def plot(V, L=None, X=None, title=None):
    K = V.shape[0]
    if L is None:
        L = ["Line " + str(i) for i in xrange(K)]

    P = [0] * K
    for k in xrange(K):
        if X is None:
            P[k], = plt.plot(V[k, :])
        else:
            P[k], = plt.plot(X, V[k, :])

    plt.legend(P, L)
    if title is not None:
        plt.title(title)
    show()


def save(filename):
    plt.savefig(filename)