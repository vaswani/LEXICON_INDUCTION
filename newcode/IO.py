from common import *

def readFeatures(filename):
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
def writeCSV(filename, X):
    (N,D) = X.features.shape
    features = np.asarray(X.features)
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in xrange(N):
            writer.writerow([str(X.words[i]),0] + [j for j in features[i,:]])
    print 'saved ', filename