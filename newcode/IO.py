from common import *
import hashlib

def readWords(filename):
    print 'reading:', filename
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        words = []
        freq = []
        features = []
        for row in reader:
            #print row
            words.append(row[0])
            freq.append(int(row[1]))
            features.append(np.array(row[2:]).astype(np.float))  # skip frequency
            i = i + 1
        X = Words()
        X.words = np.array(words)
        X.freq = np.array(freq)
        X.features = np.array(features)
        return X

# write X into filename in the following format
# word,frequency,features
# since no frequency is given, just use 0
def writeWords(filename, X):
    (N,D) = X.features.shape
    features = np.asarray(X.features)
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in xrange(N):
            writer.writerow([str(X.words[i]),0] + [j for j in features[i,:]])
    print 'saved ', filename


def getHash(X, Y):
    s = hashlib.sha224("_".join(X.tolist() + Y.tolist())).hexdigest()
    return s[1:10] # just take the first 10 letters.

def getMatchingFilename(options, X, Y):
    h = getHash(X, Y)
    filename = 'cache/matching=' + h + '_expid=' + str(options.exp_id) + '.csv'
    return filename

def writeMatching(options, X, Y, pi):
    filename = getMatchingFilename(options, X, Y)
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(pi)

def readMatching(options, X, Y):
    filename = getMatchingFilename(options, X, Y)
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            pi = [int(i) for i in np.array(row)]
    return pi
