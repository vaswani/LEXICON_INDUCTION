import common
import numpy as np
import strings
import graphs
import copy
from ICD import ICD
import perm
from MatrixStringKeys import MSK
import DictDict
import sys


# Utility classes
class Words:
    def __init__(self, name):
        self.name = name
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
        # L = strings.strlen(self.words)

        # normalize the features
        if self.isPickled():
            M0 = MSK(self.repr, self.words, self.featureNames)
            M0.normalize()
            M0.computeKernel()
            self.msk = M0

            # M1 = MSK(self.repr, self.words, self.featureNames)
            # M1.computeKernel()
            # print np.linalg.norm(M1.K - M0.K)
            # asd
            # orthoDD, orthoFeatures = strings.to_ngram_dictionary(self.words)
            # mapOrthoFeature = lambda f: "O_" + f
            # mapContextFeature = lambda f: "C_" + f
            # allFeatures = [mapContextFeature(f) for f in self.featureNames] + [mapOrthoFeature(f) for f in orthoFeatures]
            # orthoDD = DictDict.mapInnerKeys(orthoDD, mapOrthoFeature)
            # contextDD = DictDict.mapInnerKeys(self.repr, mapContextFeature)
            # print >> sys.stderr, 'Joining with Context Features'
            # allDD = DictDict.add(contextDD, orthoDD)
            # print >> sys.stderr, 'Preparing Sparse Features'
            #self.msk = MSK(self.repr, self.words, self.featureNames)
            #print >> sys.stderr, 'Computing Kernel'
            #self.msk.computeKernel()
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
        print >> sys.stderr, "Computing ICD model."
        keys = self.words[Nt:]
        K = self.msk.materializeKernel(keys, keys)
        model = ICD.ichol_words(K, keys, eta)
        print >> sys.stderr, "Computing Representations"
        K = self.msk.materializeKernel(self.words, model.keys)
        self.features = ICD.getRepresentations_words(model, K)
        print >> sys.stderr, "Done ICD."
        return model

    def addReprNoise(self, noise):
        for k in self.repr.keys():
            d = self.repr[k]
            for kk in d.keys():
                d[kk] = d[kk] + noise*common.randn((1, 1))

    def asTuple(self):
        return self.words, self.freq, self.features

    @staticmethod
    def concat(A, B):
        C = Words(A.name)  # take the name of A
        C.words = np.append(A.words, B.words)
        C.freq = np.append(A.freq, B.freq)
        C.features = np.vstack((A.features, B.features))
        # union the two dictionaries (python!, what magic you have)
        C.repr = dict({}, **copy.deepcopy(A.repr))
        C.repr = dict(C.repr, **copy.deepcopy(B.repr))
        C.featureNames = A.featureNames
        assert A.featureNames == B.featureNames
        return C