from common import *
import editdist  # please import the edit distance code available at https://code.google.com/p/py-editdist/
import numpy as np


def strlen(a): # given an array of strings, returns an array of string lengths
    v = np.array([len(a[i]) for i in xrange(len(a))])
    return v


def pweditdist(X, Y):  # computes the pairwise edit-distance between lists of words X and Y
    NX = X.size
    NY = Y.size
    D = np.zeros((NX, NY))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            D[i, j] = levenshtein(x, y)
        if i%10==0:
            print i
    return D


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return editdist.distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = xrange(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


if __name__ == '__main__':
    print levenshtein("abcd", "abcef")==2

# def editdist(seq1, seq2):
#     """Calculate the Damerau-Levenshtein distance between sequences.
#
#     This distance is the number of additions, deletions, substitutions,
#     and transpositions needed to transform the first sequence into the
#     second. Although generally used with strings, any sequences of
#     comparable objects will work.
#
#     Transpositions are exchanges of *consecutive* characters; all other
#     operations are self-explanatory.
#
#     This implementation is O(N*M) time and O(M) space, for N and M the
#     lengths of the two sequences.
#
#     >>> dameraulevenshtein('ba', 'abc')
#     2
#     >>> dameraulevenshtein('fee', 'deed')
#     2
#
#     It works with arbitrary sequences too:
#     >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
#     2
#     """
#     # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
#     # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
#     # However, only the current and two previous rows are needed at once,
#     # so we only store those.
#     oneago = None
#     thisrow = range(1, len(seq2) + 1) + [0]
#     for x in xrange(len(seq1)):
#         # Python lists wrap around for negative indices, so put the
#         # leftmost column at the *end* of the list. This matches with
#         # the zero-indexed strings and saves extra calculation.
#         twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
#         for y in xrange(len(seq2)):
#             delcost = oneago[y] + 1
#             addcost = thisrow[y - 1] + 1
#             subcost = oneago[y - 1] + (seq1[x] != seq2[y])
#             thisrow[y] = min(delcost, addcost, subcost)
#             # This block deals with transpositions
#             if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
#                 and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]):
#                 thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
#     return thisrow[len(seq2) - 1]