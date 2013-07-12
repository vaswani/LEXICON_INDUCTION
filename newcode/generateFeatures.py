from collections import Counter
from common import *
import nltk
import sys
import IO
import re


def getTopNouns(filename, tags, K):
    counter_all = defaultdict(int)
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
        #tokens = re.split("[ ,.?!]", line)
        words = filter(lambda word: word not in ',-.', tokens)
        for word in words:
            word = word.lower()
            corpus.append(word)
            counter_all[word] += 1
        j += 1

        if j % 10000 == 0:
            log(100, 'Parsed sentence:', j)
    log(100, "parsed", j, 'sentences')

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
        tag = S[-1]  # some words appear with space. TODO: check how many, concat?
        word = S[0]
        if accept_tags is not None and tag in accept_tags and len(word) > 1:
            tags[word.lower()] = tag

    return tags


def extractContextFeatures(corpus, nouns, top_nouns, window_size):
    context_features = defaultdict(dd)
    d = window_size
    pair_count = 0
    print >> sys.stderr, 'Using', len(nouns), 'words'
    Nc = len(corpus)
    print >> sys.stderr, 'number of word in corpus', Nc
    for (i, word) in enumerate(corpus):     
        if word in top_nouns:
            window = corpus[i-d:i] + corpus[i+1:i+1+d]
            for other_word in window:
                if other_word in nouns:
                    context_features[word][other_word] += 1
                    # print >> sys.stderr, 'Added pair', word, '-', other_word
                    pair_count += 1
        if i % 100000 == 0:
            print >> sys.stderr, 'at word', i, 'out of', Nc
    log(100, 'pair count', pair_count)
    return context_features


def extract(filename_text, filename_tags, accept_tags, K):
    # read corpus,
    # extract the top most frequent K nouns
    # for those words, get the co-occurances features
    # TODO: get the orhtographic features
    print >> sys.stderr,'reading tag file', filename_tags
    tags = readTags(filename_tags, accept_tags)
    print >> sys.stderr, 'filtering in accepted tas', accept_tags
    nouns = filter(lambda word: tags[word] in accept_tags, tags)  # get all nouns
    nouns = set(nouns)    
    (top_nouns, corpus) = getTopNouns(filename_text, tags, K)  # top_noun is a [nounfrequency] dict
    window_size = 4
    context_features = extractContextFeatures(corpus, nouns, top_nouns, window_size)
    return top_nouns, context_features
    

def outputWords(top_nouns, context_features):
    noun_keys = sorted(top_nouns, key=top_nouns.get, reverse=True)
    for word in noun_keys:
        sys.stdout.write(word)
        sys.stdout.write(',')
        freq = top_nouns[word]
        sys.stdout.write(str(freq))
        sys.stdout.write(',')
        V = [context_features[word][other_word] for other_word in noun_keys]
        print ','.join([str(v) for v in V])


if __name__ == '__main__':
    global verbosity
    verbosity = 0
    filename_text = sys.argv[1]
    filename_tags = sys.argv[2]
    N = int(sys.argv[3])
    lang = (sys.argv[4])
    assert lang == 'en' or lang == 'es'
    if lang == 'en':
        accept_tags = ['NN', 'NNS', 'NP', 'NPS']
    elif lang == 'es':
        accept_tags = ['NC', 'NP']
    outfile = lang + '_' + 'pickled.txt'

    log(100, 'Extracting', N, 'top nouns', '-- accepted tags:', accept_tags)
    top_nouns, context_features = extract(filename_text, filename_tags, accept_tags, N)
    print top_nouns
    IO.writePickledWords(outfile, top_nouns, context_features)