import cPickle
import common
import sys
import csv
import MatchingUtil as MU
import numpy as np
import words
import scipy.io


def readCSV(filename):
    lines = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append(row)
    return lines


def readWords(filename):  # read the Word format from a CSV (word, frequency, feature1 ... featureD)
    common.log(50, 'reading Words:', filename)
    i = 0
    W = []
    freq = []
    features = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            #print row
            W.append(row[0])
            freq.append(int(row[1]))
            features.append(np.array(row[2:]).astype(np.float))  # skip frequency
            i += 1
    X = words.Words(filename)
    X.words = np.array(W)
    X.freq = np.array(freq)
    X.features = np.array(features)
    common.log(50, 'read', len(X.freq), 'words')
    csvfile.close()
    return X


def readPickledWords(filename):
    obj = unpickle(filename)
    N = len(obj['freq'])
    D = len(obj['featureNames'])
    # pre-allocate
    W = [0] * N
    freq = [0] * N
    common.log(50, 'reading pickled words N =', N, 'and D =', D)
    for i, w in enumerate(obj['freq']):
        W[i] = w
        freq[i] = obj['freq'][w]
    X = words.Words(filename)
    X.words = W # np.array(W)
    X.freq = np.array(freq)
    X.repr = obj['features']  # a dict to dict to count
    X.featureNames = obj['featureNames']
    assert set(X.repr.keys()) == set(X.words)
    return X


def writePickledWords(filename, freq, features, featureNames):
    obj = {'freq': freq, 'features': features, 'featureNames': featureNames}
    print >> sys.stderr, "Pickling features to", filename, 'N =', len(freq), 'D =', len(featureNames)
    pickle(filename, obj)


# write X into filename in the following format
# (word,frequency,features)
# since no frequency is given, just use 0
def writeWords(filename, X):
    (N, D) = X.features.shape
    common.log(50, 'writing', N, 'word in plaintext')
    features = np.asarray(X.features)
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in xrange(N):
            writer.writerow([str(X.words[i]),X.freq[i]] + [j for j in features[i, :]])
    csvfile.close()
    print >> sys.stderr, 'saved NxD', (N, D), 'words in:\t', filename


def getHash(X, Y):
    s = hashlib.sha224("_".join(X.tolist() + Y.tolist())).hexdigest()
    return s[1:10] # just take the first 10 letters.


def writeString(filename, string):
    with open(filename, 'wb') as f:
        f.write(string)



# def getMatchingFilename(options, X, Y):
#     # computes a canonical name for a matching, based on the original lists of words
#     h = getHash(X, Y)
#     filename = 'cache/matching=' + h + '_expid=' + str(options.exp_id) + '.csv'
#     return filename
#
#
# def writeMatching(options, X, Y, pi, edge_cost):  # writes a matching pi to a csv file.
#     filename = getMatchingFilename(options, X, Y)
#     print >> sys.stderr, 'writing matching into file ', filename
#     with open(filename, 'wb') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         writer.writerow(pi)
#         writer.writerow(edge_cost)
#         matching = MU.getMatching(X, Y, pi, edge_cost)
#         writer.writerow(matching[0, :])
#         writer.writerow(matching[1, :])


def readMatching(options, X, Y):  # reads a matching from a csv file.
    filename = getMatchingFilename(options, X, Y)
    print >> sys.stderr,'reading matching file', filename
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        rows = [row for row in reader]
        pi = [int(i) for i in np.array(rows[0])]
        edge_cost = [float(i) for i in np.array(rows[1])]
    pi = np.array(pi)
    edge_cost = np.array(edge_cost)
    return pi, edge_cost


def getEditDistFilename(X, Y):
    h = getHash(X, Y)
    filename = 'cache/edit_dist=' + h + '.npy'
    return filename


def readNumpyArray(filename):
    return np.load(filename)


def writeNumpyArray(filename, D):
    print >> sys.stderr, 'Saved array in:', filename
    np.save(filename, D)


def writeSeed(filename, seed):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for (i, v) in enumerate(seed):
            writer.writerow(v)
    print >> sys.stderr, 'Saved seed:', filename


def readSeed(filename):
    wordsX = []
    wordsY = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            #print row
            wordsX.append(row[0])
            wordsY.append(row[1])
    return wordsX, wordsY
    
    
def pickle(filename, obj):
    outfile = open(filename, 'wb')
    cPickle.dump(obj, outfile, protocol=2)
    outfile.close()


def unpickle(filename):
    infile = open(filename, 'rb')
    obj = cPickle.load(infile)
    infile.close()
    return obj


def readPy(filename):
    with open(filename, 'r') as f:
        data = f.read()
    A = eval(data)
    return A


def exportMatlab(filename, varname, A):
    scipy.io.savemat('/tmp/matrices/' + filename, mdict={varname: A})


if __name__ == '__main__':
    readPickledWords('data/mock/pockX.txt')
    readWords('data/mock/mockX.txt')
