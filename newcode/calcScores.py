# computes precision and recall scores
from common import *
from collections import defaultdict


# dictionary contains the list of all matching obtained from the gold lexicon.
# dictionary has format dict of key, to dict of values
# source words is a dictionary of source words and target words is a dictionary of target words
def filterDictionary(dictionary, source_words, target_words):
    F = defaultdict(defaultdict)
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
def computePrecisionRecall(dict, matches, weights):
    recall_cutoffs = [0.05, 0.1, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6]

    M = len(dict)
    # sort according to weights (increasing)
    (_, pi) = perm.sort(weights)
    matches = matches[pi, :]
    (N, _) = matches.shape
    C = np.zeros((N, 3))  # [1, exists in source, target matches]
    for i, (source_word, target_word) in enumerate(matches):
        C[i, 1] = 1  # always 1
        if source_word in dict:
            C[i, 2] = 1  # word exists as a source word
            if target_word in dict[source_word]:
                C[i, 3] = 1  # (source, target) words are correctly matched according to dict

    C = np.cumsum(C, 0)  # cumulative sum per column
    scores = Struct()
    scores.M = M
    scores.precision = C[:, 3] / C[:, 2]
    scores.recall = C[:, 3] / M
    scores.F1 = F1(scores.precision, scores.recall)

    return scores  # C should allow computing precision/recall/F1 for any cutoff value.



# computes the F1 score
def F1(p, r, beta=1):
    sqr_beta = beta**2
    D = sqr_beta*p + r
    score = (1 + sqr_beta) * (p*r) / D
    return score


if __name__ == '__main__':
    print 1
