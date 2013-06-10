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


def randn(N, D):
    A = np.random.randn(N, D)
    return A


def normalize_rows(V): # assumes V is a numpy array!
    Z = norm_rows(V)
    V = (V.T / Z).T
    return V


def normsqr_rows(V):
    return (V*V).sum(1)


def norm_rows(V):
    Z = np.sqrt(normsqr_rows(V))
    return Z
