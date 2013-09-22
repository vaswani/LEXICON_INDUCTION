import common
import numpy as np
import strings
import graphs
# import copy
from ICD import ICD
import perm
from MatrixStringKeys import MSK
import sys
import os.path
import IO


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
        self.options = None

    def setOptions(self, options):
        self.options = options

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
            (orthoDD, orthoFeatures) = strings.to_ngram_dictionary(self.words, affix=True)
            self.orthoMSK = MSK(orthoDD, self.words, orthoFeatures).normalize(norm='l2')
            self.contextMSK = MSK(self.repr, self.words, self.featureNames).normalize(norm='l2')
        else:
            self.features = common.normalize_rows(self.features)
        # TODO: should be add logFr and L ?

    def computeOrthographicKernel(self):
        print >> sys.stderr, 'Computing Orthographic Kernel for', self.name
        return self.orthoMSK.makeLinearKernel()

    def computeContextKernel(self):
        print >> sys.stderr, 'Computing Context Kernel for', self.name
        return self.contextMSK.makeLinearKernel()

    def cacheOrComputeKernel(self, filename, f):
        if os.path.exists(filename):
            print >> sys.stderr, 'Loading kernel from file:', filename
            return IO.unpickle(filename)
        else:
            K = f(self)
            print >> sys.stderr, 'Saving kernel to file:', filename
            IO.pickle(filename, K)
            return K

    def computeKernel(self, options):
        print >> sys.stderr, 'Computing Kernel for', self.name

        K_context = None
        K_ortho = None
        if options.useContextFeatures == 1:
            filename_ck = self.name.replace('.', '_context_kernel.')

            K_context = self.cacheOrComputeKernel(filename_ck, lambda self: self.computeContextKernel())
            #K_context0 = self.computeContextKernel()
            #print 'AAA: ', np.linalg.norm(K_context.K - K_context0.K)

        if options.useOrthoFeatures == 1:
            filename_ok = self.name.replace('.', '_ortho_kernel.')
            K_ortho = self.cacheOrComputeKernel(filename_ok, lambda self: self.computeOrthographicKernel())
            #K_ortho0 = self.computeOrthographicKernel()
            #print 'BBB: ', np.linalg.norm(K_ortho.K - K_ortho0.K)

        if K_ortho is None:
            self.kernel = K_context
        elif K_context is None:
            self.kernel = K_ortho
        else:
            assert K_context.strings == K_ortho.strings  # strings should be numbered the same.
            K_context.K += K_ortho.K
            self.kernel = K_context

        return self.kernel

    def getKernel(self):
        return self.kernel.K

    def materializeGraph(self):
        if self.G is None:
            return None
        # otherwise, return the graph,
        # permuted according to the order of self.words
        return self.G.materialize(self.words, self.words)

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
        # note that there is no need to permute the graph, since it is stored as an MSK
        # and will be materialized according to the order of self.words

    def isPickled(self):
        return len(self.repr) > 0

    def pushSeedToEnd(self, seed):
        # push the seed to the end of the list. The order of seed matter, but the rest doesn't.
        S = set(seed)
        non_seed = filter(lambda x: x not in S, self.words)
        self.words = np.array(non_seed + seed)

    def ICD_representation(self, Nt, eta):
        # step 1: calculate ICD model based on the last N-Nt words
        # (those are supposed to be well aligned - the seed is at the end)
        # step 2: projected all the data based on the model.
        # use_ICD = True
        # if use_ICD:
        print >> sys.stderr, "Computing ICD model for",  self.name
        keys = self.words[Nt:]
        K = self.kernel.materialize(keys, keys)
        model = ICD.ichol_words(K, keys, eta)
        print >> sys.stderr, "Computing Representations"
        K = self.kernel.materialize(self.words, model.keys)
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

    # @staticmethod
    # def concat(A, B):
    #     C = Words(A.name)  # take the name of A
    #     C.words = np.append(A.words, B.words)
    #     C.freq = np.append(A.freq, B.freq)
    #     C.features = np.vstack((A.features, B.features))
    #     # union the two dictionaries (python!, what magic you have)
    #     C.repr = dict({}, **copy.deepcopy(A.repr))
    #     C.repr = dict(C.repr, **copy.deepcopy(B.repr))
    #     C.featureNames = A.featureNames
    #     assert A.featureNames == B.featureNames
    #     return C