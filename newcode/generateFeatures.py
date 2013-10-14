from Counter0 import Counter
from collections import OrderedDict
from collections import defaultdict
import common
import nltk
import sys
import IO


def getTopNouns(filename, nouns, K, mode):
    counter_noun = Counter()
    j = 0
    corpus = []
    for line in open(filename):
        j += 1
        if mode == 1 and j >= 50000:
            continue
        if mode == 2 and j < 50000:
            continue

        # tokenize each line, lower case words and count
        # note that the tags are static, so we don't really know if a word is used as a noun in each context
        # we just take it if it CAN be a noun
        line = line.strip()
        if len(line) == 0:
            continue
        tokens = [word for word in nltk.word_tokenize(line)]  # was word.lower()
        #tokens = re.split("[ ,.?!]", line)
        nouns_in_line = filter(lambda t: t not in ',-.' and t in nouns, tokens)  # was "t.lower() in nouns"
        for noun in nouns_in_line:
            counter_noun[noun] += 1
        corpus.append(nouns_in_line)

        if j % 10000 == 0:
            common.log(100, 'Parsed sentence:', j)

    #print corpus
    common.log(100, "parsed", j, 'sentences')
    # count only nouns
    return OrderedDict(counter_noun.most_common(K)), corpus


# read the "word, POS-tag" file
def readTags(filename, accept_tags=None):
    tags = dict()
    for line in open(filename):
        line = line.strip()
        S = line.split('\t')
        tag = S[-1]  # some words appear with space. TODO: check how many, concat?
        word = S[0]
        if accept_tags is not None and tag in accept_tags and len(word) > 1:
            tags[word] = tag  # was word.lower()
    return tags


def extractContextFeatures(corpus, nouns, top_nouns, window_size):
    context_features = defaultdict(common.dd)
    d = window_size
    pair_count = 0
    sum_L0 = 0
    print >> sys.stderr, 'Using', len(nouns), 'words as co-occurrence features.'
    Nc = len(corpus)
    print >> sys.stderr, 'number of lines in corpus', Nc
    for (l, nouns_in_line) in enumerate(corpus):
        for (i, noun) in enumerate(nouns_in_line):
            if noun in top_nouns:
                window = nouns_in_line[max(i-d, 0):i] + nouns_in_line[i+1:i+1+d]
                for other_noun in window:
                    context_features[noun][other_noun] += 1
                    # print >> sys.stderr, 'Added pair', word, '-', other_word
                    pair_count += 1
        if l % 10000 == 0:
            print >> sys.stderr, 'at line', l, 'out of', Nc

    for noun in context_features:
        sum_L0 += len(context_features[noun])

    common.log(100, 'pair count', pair_count)
    common.log(100, 'avg non-zero', sum_L0 / len(top_nouns))
    return context_features


def extract(filename_text, filename_tags, accept_tags, K):
    # read corpus,
    # extract the top most frequent K nouns
    # for those words, get the co-occurances features
    # TODO: generate the orthographic features
    select_lines_mode = 0
    print >> sys.stderr, 'reading tag file', filename_tags
    tags = readTags(filename_tags, accept_tags)

    # form a set of all unique nouns (set allows fast look-up)
    print >> sys.stderr, 'filtering in accepted tags', accept_tags
    feature_names = set(filter(lambda word: tags[word] in accept_tags, tags))

    # top_noun is a {noun: frequency} dict
    (top_nouns, corpus) = getTopNouns(filename_text, feature_names, K, select_lines_mode)
    window_size = 4
    context_features = extractContextFeatures(corpus, feature_names, top_nouns, window_size)

    feature_names = list(feature_names)  # subsequently, order matters.
    return top_nouns, context_features, feature_names
    

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
    out_filename = lang + '_' + 'pickled_N='+str(N)+'.txt'

    common.log(100, 'Extracting', N, 'top nouns', '-- accepted tags:', accept_tags)
    top_nouns_freq, context_features, feature_names = extract(filename_text, filename_tags, accept_tags, N)

    # sort by frequency (descending)
    context_features0 = OrderedDict()
    for noun in top_nouns_freq:
        context_features0[noun] = context_features[noun]

    IO.writePickledWords(out_filename, top_nouns_freq, context_features0, feature_names)