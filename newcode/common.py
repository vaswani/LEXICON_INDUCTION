__author__ = 'Tomer'
import numpy as np
import csv
import sys
import perm
import strings


class Words:
    pass


class Options:
    pass


def randn(S):
    A = np.random.randn(*S)
    return A


def randi(N, S):
    return np.random.randint(N, size=S)


def normalize_rows(V): # assumes V is a numpy array!
    Z = norm_rows(V)
    V = (V.T / Z).T
    return V


def normsqr_rows(V):
    return (V*V).sum(1)


def norm_rows(V):
    Z = np.sqrt(normsqr_rows(V))
    return Z
