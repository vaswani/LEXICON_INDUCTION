import common
import numpy as np
import strings
import graphs
import copy
from SparseFeatures import *
from ICD import ICD
import perm
from kernels import DictDictKernel
from MatrixStringKeys import MSK


# Utility classes
class Words:
    def __init__(self):
        self.words = []
        self.freq = []
        self.features = []
        self.featureNames = []
        self.G = None
        self.repr = {}

    def toNP(self):  # to numpy array the fields
        #self.words = np.array(self.words)
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
                    raise Exception('word with 0 co-occurrence')
                v.scale(1/v.norm())  # scale it (no need to assign since this is done in place)
            #self.DDK = DictDictKernel(self.repr)
            print >> sys.stderr, 'Computing kernel'
            self.msk = MSK(self.repr, self.words, self.featureNames)
            self.msk.computeKernel()
        else:
            self.features = common.normalize_rows(self.features)
        # TODO: should be add logFr and L ?

    # this method permutes all the fields of X according to pi.
    # if pi is shorter than X.words, than only the first entries are permuted,
    # and the last remain in their position.
    def permuteFirstWords(self, pi):
        pi = np.array(pi)
        M = len(pi)
        id = perm.ID(M)
        self.words[id] = self.words[pi]
        self.freq[id] = self.freq[pi]
        if not common.isEmpty(self.features):
            self.features[id, :] = self.features[pi, :]
        if self.G is not None:
            self.G = graphs.permute(self.G, pi)

    def isPickled(self):
        return len(self.repr) > 0

    def ICD_representation(self, Nt, eta):
        # step 1: calculate ICD model based on the last N-Nt words
        # (those are supposed to be well aligned - the seed is at the end)
        # step 2: projected all the data based on the model.
        # use_ICD = True
        # if use_ICD:
        print >> sys.stderr, "computing ICD model."
        keys = self.words[Nt:]
        #self.DDK.compute(keys, keys)
        #G = self.DDK.materialize(keys, keys)
        K = self.msk.materializeKernel(keys, keys)
        #print 'norm(G-K)', common.norm(K-G, 2)
        model = ICD.ichol_words(K, keys, eta)
        print >> sys.stderr, "Computing representations"
        #self.DDK.compute(keys, self.words)
        #G = self.DDK.materialize(self.words, model.keys)
        K = self.msk.materializeKernel(self.words, model.keys)
        #print 'norm(G-K)', common.norm(K-G, 2)
        self.features = ICD.getRepresentations_words(model, K)
        print >> sys.stderr, "Done with ICD."
        return model
        # else:
        #     for k in self.repr:
        #         keys = self.repr[k].keys()
        #         break
        #
        #     F = []
        #     for word in self.words:
        #         F.append(SparseFeatures(self.repr[word]).asVector(keys))
        #     self.features = np.array(F)
        # note that self.features will be sorted (row-wise0 by self.words

    def addReprNoise(self, noise):
        for k in self.repr.keys():
            d = self.repr[k]
            for kk in d.keys():
                d[kk] = d[kk] + noise*common.randn((1, 1))

    def asTuple(self):
        return self.words, self.freq, self.features

    @staticmethod
    def concat(A, B):
        C = Words()
        C.words = np.append(A.words, B.words)
        C.freq = np.append(A.freq, B.freq)
        C.features = np.vstack((A.features, B.features))
        # union the two dictionaries (python!, what magic you have)
        C.repr = dict({}, **copy.deepcopy(A.repr))
        C.repr = dict(C.repr, **copy.deepcopy(B.repr))
        C.featureNames = A.featureNames
        assert A.featureNames == B.featureNames
        return C