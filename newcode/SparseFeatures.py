import numpy as np
import IO
import sys
import time


# This class wraps a dictionary of {keys ==> numeric} and provides
# means to calculate the inner produce between two such dictionaries.
class SparseFeatures:
    def __init__(self, dict=dict()):
        self.dict = dict

    def __getitem__(self, k):
        return self.dict[k]
        
    def __setitem__(self, k, v):
        self.dict[k] = v

    def __mul__(self, other):    
        return self.mult(other)

    def __repr__(self):
        return "<SparseFeatures: " + self.dict.__str__() + ">"

    def L0(self):  # number of non-zero elements
        return len(self.dict)

    def inc(self, feature):
        if feature not in self.dict:
            self.dict[feature] = 1
        else: 
            self.dict[feature] += 1

    def set(self, feature, value):
        self[feature] = value

    def scale(self, scalar):  # scale the entire "vector" by a scalar
        for k in self.dict.keys():
            self.dict[k] *= scalar
            
    def mult(self, other):
        if other.L0() < self.L0():
            return other.mult(self)
        else:
            sum = 0
            for k in self.dict:
                if k in other.dict:
                    sum += self.dict[k] * other.dict[k]
        return sum

    def norm(self):
        return np.sqrt(self * self)

    def getDict(self):
        return self.dict

    def asVector(self, keys):
        A = []
        for k in keys:
            if k not in self.dict:
                A.append(0)
            else:
                A.append(self.dict[k])
        return A


    @staticmethod
    def faster_mult(d_i, d_j):

        if len(d_i) < len(d_j):
            A = d_i
            B = d_j
        else:
            A = d_j
            B = d_i

        sum = 0
        for k in A:
            if k in B:
                sum += A[k] * B[k]
        return sum


    @staticmethod
    def getKernel(double_dict, words, other_words=None):
        if other_words is None:
            other_words = words
        N = len(words)
        M = len(other_words)
        G = np.mat(np.zeros((N, M)))
        t = time.time()
        # naive matrix multiplication
        if len(words) == len(other_words) and (words == other_words).all():
            for i in xrange(N):
                #word_i = words[i]
                #v_i = SparseFeatures(double_dict[word_i])
                # G[i, i] = v_i.mult(v_i)  # diagonal
                d_i = double_dict[words[i]]
                G[i, i] = SparseFeatures.faster_mult(d_i, d_i)
                for j in xrange(i):
                    # word_j = other_words[j]  # NOTE: other_words!
                    # v_j = SparseFeatures(double_dict[word_j])
                    # G[i, j] = v_i.mult(v_j)
                    # G[j, i] = G[i, j]
                    d_j = double_dict[words[j]]
                    G[i, j] = SparseFeatures.faster_mult(d_i, d_j)
                    G[j, i] = G[i, j]
        else:
            for i in xrange(N):
                word_i = words[i]
                v_i = SparseFeatures(double_dict[word_i])
                for j in xrange(M):
                    word_j = other_words[j]  # NOTE: other_words!
                    v_j = SparseFeatures(double_dict[word_j])
                    G[i, j] = v_i.mult(v_j)
        print 'elapsed', time.time() - t
        return G


if __name__ == '__main__':
    v = SparseFeatures()
    v.inc('a')
    v.inc('b')
    v.inc('b')
    v['a'] = 3
    assert v['a'] == 3
    assert v['b'] == 2
    v.scale(1./3)
    assert v['b'] == 2/3.

    u = SparseFeatures()
    print u.asVector(['a', 'a1', 'b', 'c'])
    u['a'] = 2
    u['b'] = 3
    u['c'] = 4

    print v.dict
    print 'norm(v):', v.norm()
    print v['a']
    print u
    print v
    print u*v

    d = {}
    d['a'] = {'c': 1, 'd': np.sqrt(2)}
    d['b'] = {'c': 0, 'd': np.sqrt(1/2.0)}

    A = SparseFeatures.getKernel(d, ['a', 'b'])
    print 'kernel:\n', A

    if len(sys.argv) > 2:
        mock = sys.argv[1]
        pock = sys.argv[2]
        print mock, pock
        m = IO.readWords(mock)
        p = IO.readPickledWords(pock)

        print "words equal =", "True" if all(m.words == p.words) else "False"
        print "freq equal =", "True" if all(m.freq == p.freq) else "False"

        for k in p.repr:
            keys = p.repr[k].keys()
            break
        F = []
        for word in p.words:
            F.append(SparseFeatures(p.repr[word]).asVector(keys))
        P = np.array(F)
        print 'norm = 0: ', "True" if all(m.freq == p.freq) else "False"





