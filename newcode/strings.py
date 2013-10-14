import editdist  # please import the edit distance code available at https://code.google.com/p/py-editdist/
import numpy as np
from collections import Counter
from collections import OrderedDict


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


def to_ngram_dictionary(strings, MAX_N=3, MIN_N=1, affix=False):
    DD = OrderedDict()
    features = Counter()
    if affix:
        f_affix = lambda s: "_" + s + "_"
    else:
        f_affix = lambda s: s

    for s in strings:
        grams = [g for g in ngrams(f_affix(s), MAX_N, MIN_N)]
        DD[s] = Counter(grams)
        features = features + DD[s]  # used to keep all distinct features

    features = features.keys()
    return DD, features


## note that this function works for both strings and lists.
def ngrams(tokens, MAX_N, MIN_N=1):
    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i+MIN_N, min(n_tokens, i+MAX_N)+1):
            yield tokens[i:j]


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