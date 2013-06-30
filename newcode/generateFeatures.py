from collections import Counter
from collections import defaultdict
from common import *
import nltk
import sys


def getTopNouns(filename, tags, K):
    counter_all = Counter()
    j = 0
    corpus = []
    for line in open(filename):
        # tokenize each line, lower case words and count
        # note that the tags are static, so we don't really know if a word is used as a noun in each context
        # we just take it if it CAN be a noun
        line = line.strip()
        if len(line) == 0:
            continue
        tokens = [word for word in nltk.word_tokenize(line)]
        words = filter(lambda word: word not in ',-.', tokens)
        for word in words:
            word = word.lower()
            corpus.append(word)
            counter_all[word] += 1
        j += 1
        # if j > 10000:
        #     break
        if j % 10000 == 0:
            log(100, 'Parsed sentence:', j)
    log(100, "parsed ", j, 'sentences')

    # extract only nouns
    counter_noun = Counter()
    for k in counter_all.keys():
        if k in tags:
            counter_noun[k] = counter_all[k]

    return Counter(dict(counter_noun.most_common(K))), corpus


def readTags(filename, accept_tags=None):
    tags = dict()
    for line in open(filename):
        line = line.strip()
        S = line.split('\t')
        tag = S[-1]
        word = S[0]
        if accept_tags is not None and tag in accept_tags and len(word) > 1:
            tags[word.lower()] = tag

    return tags


def extractContextFeatures(corpus, top_nouns):
    context_features = defaultdict(lambda: defaultdict(int))
    d = 4
    pair_count = 0
    for (i, word) in enumerate(corpus):
        is_top_noun = word in top_nouns
        if is_top_noun:
            window = corpus[i-4:i] + corpus[i+1:i+1+d]
            for other_word in window:
                if other_word in top_nouns:
                    context_features[word][other_word] += 1
                    #print 'Added pair', word, '-', other_word
                    pair_count += 1
        if i % 5000 == 0:
            log(1000,'word', i, word, is_top_noun)
    log(100, 'pair count', pair_count)
    return context_features


def extract(filename_text, filename_tags, accept_tags, K):
    # read corpus,
    # extract the top most frequent K nouns
    # for those words, get the co-occurances features
    # TODO: get the orhtographic features

    tags = readTags(filename_tags, accept_tags)
    (top_nouns, corpus) = getTopNouns(filename_text, tags, K)  # top_noun is a [nounfrequency] dict

    context_features = extractContextFeatures(corpus, top_nouns)
    return top_nouns, context_features


if __name__ == '__main__':
    global verbosity
    verbosity = 0
    filename_text = sys.argv[1]
    filename_tags = sys.argv[2]
    K = int(sys.argv[3])
    lang = (sys.argv[4])
    assert lang == 'en' or lang == 'es'
    if lang == 'en':
        accept_tags = ['NN', 'NNS', 'NP', 'NPS']
    elif lang == 'es':
        accept_tags = ['NC', 'NP']


    log(100, 'Extracting', K, 'top nouns', '-- accepted tags:', accept_tags)
    top_nouns, context_features = extract(filename_text, filename_tags, accept_tags, K)

    noun_keys = sorted(top_nouns, key=top_nouns.get, reverse=True)
    for word in noun_keys:
        sys.stdout.write(word)
        sys.stdout.write(',')
        freq = top_nouns[word]
        sys.stdout.write(str(freq))
        sys.stdout.write(',')
        V = [context_features[word][other_word] for other_word in noun_keys]
        print ','.join([str(v) for v in V])
