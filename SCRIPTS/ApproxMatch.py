#!/usr/bin/env python

from numpy import *
import sys

# Approximate minimum weighted matching on the input C
# returns the cost = \sum C[ i, pi[i] ]
def ApproxMatch(C):
    N = C.shape[1]
    left = zeros((N,1))
    right = zeros((N,1))
    I = tile(arange(N), [N, 1])
    J = I.T
    I = I.flatten()
    J = J.flatten()
    sigma = argsort(C.flatten())
    cost = 0
    pi = zeros([N,1])
    M = sigma.size;
    for element in sigma.flat:
        i = I[element]
        j = J[element]
        if left[i]==0 and right[j]==0:
            pi[i] = j
            cost = cost + C[i,j]
            left[i] = 1
            right[j] = 1
    return cost, pi

# test
C = matrix('1 2 3; 6 4 5; 8 7 9')
cost, pi = ApproxMatch(C)
print C
print "cost:", cost
print pi

