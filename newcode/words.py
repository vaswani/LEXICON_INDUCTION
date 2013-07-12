from common import *
import numpy as np
import strings
import graphs
import copy
from ICD import *


# Utility classes
class Words:
    def __init__(self):
        self.words = []
        self.freq = []
        self.features = []
        self.G = None
        self.repr = {}

    def toNP(self):  # to numpy array the fields
        self.words = np.array(self.words)
        self.freq = np.array(self.freq)
        self.features = np.array(self.features)
        self.G = np.array(self.G)

    def setupFeatures(self):
        #logFr = np.log(X.freq)
        L = strings.strlen(self.words)

        # normalize the features
        if self.isPickled():
            for word in self.words:
                v = SparseFeatures(self.repr[word])  # wrap the dictionary
                if v.norm() == 0:
                    print '0 norm for word', word, '?'
                v.scale(1/v.norm())  # scale it (no need to assign since this is done in place)
        else:
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

    def isPickled(self):
        return len(self.repr) > 0

    def ICD_representation(self, Nt, eta):
        # step 1: calculate ICD model based on the last N-Nt words
        # (those are supposed to be well aligned - the seed is at the end)
        # step 2: projected all the data based on the model.
        model = ICD.ichol_words(self.repr, self.words[Nt:], eta)
        self.features = ICD.getRepresentations_words(model, self.repr, self.words)


    @staticmethod
    def concat(A, B):
        C = Words()
        C.words = np.append(A.words, B.words)
        C.freq = np.append(A.freq, B.freq)
        C.features = np.vstack((A.features, B.features))
        # union the two dictionaries (python!, what magic you have)
        C.repr = dict({}, **copy.deepcopy(A.repr))
        C.repr = dict(C.repr, **copy.deepcopy(B.repr))
        return C