# computes precision and recall scores
from collections import defaultdict
from common import *


# dictionary contains the list of all matching obtained from the gold lexicon.
# dictionary has format dict of key, to dict of values
# source words is a dictionary of source words and target words is a dictionary of target words
def filterDictionary(dictionary, source_words, target_words):
    F = defaultdict(dict)
    total_items = 0
    # keep only pairs (u,w) from dictionary that are both in the source and target lists.
    for u in dictionary:
        if u not in source_words:
            continue  # u is not among the source words
        for w in dictionary[u]:
            if w in target_words:
                F[u][w] = 1
                total_items += 1
    return F, total_items


# filtered dictionary is a dictionary with the key as a source word and the value
# is another dictionary with the key as the target words
# matches is a list of lists. Each row in the list has the format
# [source_word,matched_target_word,score]. The items are sorted according to score
def getScores(dict, matches, weights):
    M = len(dict)
    # sort according to weights (increasing)
    (_, pi) = perm.sort(weights)
    matches = matches[pi, :]
    (N, _) = matches.shape
    C = np.zeros((N, 3))  # [1, exists in source, target matches]
    dict_keys = dict.keys()
    for i, (source_word, target_word) in enumerate(matches):
        C[i, 0] = 1  # always 1
        if source_word in dict_keys:
            C[i, 1] = 1  # word exists as a source word
            if target_word in dict[source_word]:
                C[i, 2] = 1  # (source, target) words are correctly matched according to dict

    C = np.cumsum(C, 0)  # cumulative sum per column
    scores = Struct()
    scores.M = M
    scores.precision = C[:, 2] / C[:, 1]
    scores.recall = C[:, 2] / M
    scores.F1 = F1(scores.precision, scores.recall)

    return scores  # C should allow computing precision/recall/F1 for any cutoff value.


def outputScores(scores, title):
    np.argwhere(scores.recall)

    cutoff = [0.05, 0.1, 0.25, 1/3, 0.4, 0.5, 0.6]
    p = []
    r = []
    f = []
    for i, c in enumerate(cutoff):
        J = np.argwhere(scores.recall > c)

        if len(J) > 0:
            j = J[0]
            p.append(scores.precision[j][0])
            r.append(scores.recall[j][0])
            f.append(scores.F1[j][0])
        else:
            p.append(-0.01)
            r.append(-0.01)
            f.append(-0.01)

    print 'Final Scores', title, ':'
    print '============='
    print 'P :', np.array(p)
    print 'R :', np.array(r)
    print 'F1:', np.array(f)


# computes the F1 score
def F1(p, r, beta=1):
    sqr_beta = beta**2
    D = sqr_beta*p + r
    score = (1 + sqr_beta) * (p*r) / D
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
    gt, total_items = filterDictionary(B, S, T)
    scores = getScores(gt, matching, weights)

    outputScores(scores,  'test run')
