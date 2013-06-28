from common import *
import numpy as np
import strings
import graphs


# Utility classes
class Words:
    def __init__(self):
        self.words = []
        self.freq = []
        self.features = []
        self.G = None

    def toNP(self):  # to numpy array the fields
        self.words = np.array(self.words)
        self.freq = np.array(self.freq)
        self.features = np.array(self.features)
        self.G = np.array(self.G)

    def setupFeatures(self):
        #logFr = np.log(X.freq)
        L = strings.strlen(self.words)
        self.features = normalize_rows(self.features)
        # TODO: should be add logFr and L ?

    # this method permutes all the fields of X according to pi.
    # if pi is shorter than X.words, than only the first entries are permuted,
    # and the last remain in their position.
    def permuteFirstWords(self, pi):
        pi = np.array(pi)
        M = len(pi)
        id = perm.ID(M)
        self.words[id] = self.words[pi]
        self.features[id, :] = self.features[pi, :]
        self.freq[id] = self.freq[pi]
        if self.G is not None:
            self.G = graphs.permute(self.G, pi)


    @staticmethod
    def concat(A, B):
        C = Words()
        C.words = np.append(A.words, B.words)
        C.freq = np.append(A.freq, B.freq)
        C.features = np.vstack((A.features, B.features))
        return C