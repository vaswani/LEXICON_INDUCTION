from common import *
# import collections
import time
# import Lists
import perm


# def inner(X, Y):
#     K = X.dot(Y.T)
#     return K
#
#
# def dist2(X, Y):
#     A = dist(X, Y)
#     return A * A


# ## a linear kernel for a dict of dict
# class DictDictKernel:
#     def __init__(self, DD):
#         self.DD = DD
#         self.K = collections.defaultdict(dict)
#
#     def compute(self, words, other_words):
#         if other_words is None:
#             other_words = words
#         N = len(words)
#         M = len(other_words)
#
#         words == list(words)
#         other_words = list(other_words)
#
#         # naive matrix multiplication
#         if len(words) == len(other_words) and (words == other_words).all():
#             for i in xrange(N):  # exploit symmetry
#                 word_i = words[i]
#                 self.mult(word_i, word_i)
#                 for j in xrange(i):
#                     word_j = other_words[j]  # NOTE: other_words!
#                     self.mult(word_i, word_j)
#         else:
#             for i in xrange(N):
#                 word_i = words[i]
#                 for j in xrange(M):
#                     word_j = other_words[j]  # NOTE: other_words!
#                     self.mult(word_i, word_j)
#
#     def mult(self, word_i, word_j):
#         if word_i in self.K and word_j in self.K[word_i]:
#             pass
#         else:
#             d_i = self.DD[word_i]
#             d_j = self.DD[word_j]
#             if len(d_i) < len(d_j):
#                 A = d_i
#                 B = d_j
#             else:
#                 A = d_j
#                 B = d_i
#
#             sum = 0
#             for k in A:
#                 if k in B:
#                     sum += A[k] * B[k]
#             self.K[word_i][word_j] = sum  # the kernel is symmetric
#             self.K[word_j][word_i] = sum  # always "English-A" * "English-B" words.
#
#     def materialize(self, words, other_words):  # make into a matrix.
#         N = len(words)
#         M = len(other_words)
#         G = Lists.zeros2(N, M)  # np.mat(np.zeros((N, M)))
#         for i in xrange(N):
#             word_i = words[i]
#             for j in xrange(M):
#                 word_j = other_words[j]  # NOTE: other_words!
#                 G[i][j] = self.K[word_i][word_j]
#
#         return np.mat(G)


if __name__ == '__main__':  # test
    np.random.seed(1)
    N = 400
    D = 800

    DD = dict()
    rangeD = range(D)
    rangeN = range(N)
    for i in xrange(N):
        DD[i] = dict()
        S = perm.randperm(rangeD)
        S = S[:D/4]
        for j in S:
            DD[i][j] = randn((1, 1))[0, 0]
    print "finished constructing."
    t = time.time()
    # K = DictDictKernel(DD)
    # #import cProfile
    # #cProfile.runct("K.compute(R, R)", globals(), locals())
    # K.compute(rangeN, rangeN)
    # G = K.materialize(rangeN, rangeN)
    # print 'elapsed', time.time() - t
    # print G








