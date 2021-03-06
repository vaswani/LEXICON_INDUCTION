# computes precision and recall scores
from collections import defaultdict
import math
import common
import sys
import csv
import perm
import numpy as np


def readLexicon(filename, delimiter='\t'):
    print >> sys.stderr, 'reading Bilexicon:', filename
    dict = defaultdict(set)
    j = 0
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        for row in reader:
            source_word = row[0]  # .lower()
            for target_word in row[1:]:
                #dict[source_word].add(target_word.lower())
                dict[source_word].add(target_word)
            j += 1
    print >> sys.stderr, 'Done reading Bilexicon'
    return dict


# dictionary contains the list of all matching obtained from the gold lexicon.
# dictionary has format dict of key, to dict of values
# source words is a dictionary of source words and target words is a dictionary of target words
def filterLexicon(lex, source_words, target_words):
    F = defaultdict(set)
    total_items = 0
    # keep only pairs (u,w) from dictionary that are both in the source and target lists.
    for u in lex:
        if u not in source_words:
            continue  # u is not among the source words
        for w in lex[u]:
            if w in target_words:
                F[u].add(w)
                total_items += 1
    return F, total_items


# filtered dictionary is a dictionary with the key as a source word and the value
# is another dictionary with the key as the target words
# matches is a list of lists. Each row in the list has the format
# [source_word,matched_target_word,score]. The items are sorted according to score
def getScores(lex, source_words, target_words, weights):
    M = len(lex)
    # sort according to weights (increasing)
    (_, pi) = perm.sort(weights, reverse=False)
    source_words = source_words[pi]
    target_words = target_words[pi]
    N = len(source_words)
    assert N == len(target_words)
    C = np.zeros((N, 3))  # [1, exists in source, target matches]
    dict_keys = lex.keys()
    for i, (source_word) in enumerate(source_words):
        target_word = target_words[i]
        C[i, 0] = 1  # always 1
        if source_word in dict_keys:
            C[i, 1] = 1  # word exists as a source word
            #if target_word in lex[source_word]:
            if is_valid_match(lex, source_word, target_word):
                C[i, 2] = 1  # (source, target) words are correctly matched according to dict

    C = np.cumsum(C, 0)  # cumulative sum per column
    scores = common.Struct()
    scores.M = M
    scores.precision = C[:, 2] / C[:, 1]
    scores.recall = C[:, 2] / M
    scores.F1 = F1(scores.precision, scores.recall)

    return scores  # C should allow computing precision/recall/F1 for any cutoff value.


def is_valid_match(lex, source_word, target_word):
    return target_word in lex[source_word]


def format(x, format='.3f'):
    if np.isscalar(x):
        return ('{:'+format+'}').format(x)
    else:
        return np.array([('{:'+format+'}').format(i) for i in x])


def outputScores(scores, title):
    cutoff = [0.1, 0.25, 1.0/3, 0.5, 2.0/3, 0.75, 0.77, 0.8]
    p = []
    r = []
    f = []

    d = 3
    cutoff = np.around(cutoff, decimals=d)
    scores.precision = np.around(scores.precision, decimals=d)
    scores.recall = np.around(scores.recall, decimals=d)
    scores.F1 = np.around(scores.F1, decimals=d)

    for i, c in enumerate(cutoff):
        J = np.argwhere(scores.recall >= c)

        if len(J) > 0:
            j = J[0]
            p.append(scores.precision[j][0])
            r.append(scores.recall[j][0])
            f.append(scores.F1[j][0])
        else:
            p.append(0)
            r.append(0)
            f.append(0)

    print 'Final Scores', title, ':'
    print '============='
    print 'Prec''  :', np.array(p)
    print 'Recall:', np.array(cutoff)
    #print 'Recall:', np.array(r)
    print 'F1    :', np.array(f)

    (maxF1, i) = common.max(scores.F1)
    precision_i = scores.precision[i]
    recall_i = scores.recall[i]
    print 'max F1:', (maxF1), 'precision: ', (precision_i), 'recall:', (recall_i)


# computes the F1 score
def F1(p, r, beta=1):
    sqr_beta = beta**2
    D = sqr_beta*p + r
    score = (1 + sqr_beta) * (p*r) / D
    score = [0 if math.isnan(s) else s for s in score]  # fix nans
    return score


if __name__ == '__main__':
    B = defaultdict(defaultdict)
    B['_big_']['_grande_'] = 1
    B['_the_']['_el_'] = 1
    B['_the_']['_la_'] = 1
    B['_the_']['_un_'] = 1
    B['_thin_']['_delgado0_'] = 1
    B['_thin_']['_delgado1_'] = 1
    B['_city_']['_ciudad_'] = 1
    B['_country_']['_pais_'] = 1
    B['_beautiful_']['_hermosa_'] = 1


    matching = []
    matching.append(['_big_', '_delgado1_'])
    matching.append(['_city_', '_el_'])
    matching.append(['_thin_', '_delgado0_'])
    matching.append(['_country_', '_hermosa_'])
    matching.append(['_beautiful_', '_pais_'])
    matching.append(['_unmatched_', '_unmatched_'])
    matching = np.array(matching)

    S = matching[:, 0]
    T = matching[:, 1]

    weights = xrange(len(matching)) # just some simple weights
            
    # filter with ground truth.
    gt, total_items = filterLexicon(B, S, T)
    scores = getScores(gt, S, T, weights)

    outputScores(scores,  'test run')
