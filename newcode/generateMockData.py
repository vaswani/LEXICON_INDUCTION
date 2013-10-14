#generates the data from the generative story
from OrderedDict0 import OrderedDict
from common import *
import IO
import perm
import words


def random_proj_shift(F):
    (_, D) = F.shape
    m = randn((1, D))
    U = randn((D, D))
    [U, _, _] = np.linalg.svd(U)  # a unitary=rotation matrix
    #F = F.dot(U)
    F = F + m
    return F


def make(N, Nseed, D):
    # generate random data in latent space
    Nboth = N + Nseed
    Z = randn((Nboth, D))
    # generate X and Y from Z.
    # they should be random transformations of Z with added noise.
    X = words.Words()
    Y = words.Words()
    # create a set of words with which it will be easy to run edit distance
    #X.words = ['a' + str(i) for i in xrange(Ns)] + ['b' + str(i) for i in xrange(Nr)]
    #Y.words = ['a' + str(i) for i in xrange(Ns)] + ['c' + str(i*2) for i in xrange(Nr)]
    X.words = np.array([str(i) for i in xrange(Nboth)])
    Y.words = np.array([str(i) for i in xrange(Nboth)])
    X.freq = np.array([i for i in xrange(Nboth)])  # mock frequencies
    Y.freq = np.array([i for i in xrange(Nboth)])
    X.features = Z
    Y.features = Z
    # create random permutation, but keep last (1-q) in place

    #X.G = np.zeros((N, N), dtype=np.float)
    #Y.G = np.zeros((N, N), dtype=np.float)

    X.features = random_proj_shift(X.features)
    Y.features = random_proj_shift(Y.features)

    # permute Y randomly
    pi = perm.ID(N)
    topX = Nboth  # int(0.92*N)
    pi = np.append(perm.randperm(pi[:topX]), pi[topX:])
    Y.permuteFirstWords(pi)
    return X, Y, pi


def getFreqRepr(X, D):
    freqX = OrderedDict()
    reprX = OrderedDict()
    for i, w in enumerate(X.words):
        freqX[w] = X.freq[i]
        reprX[w] = {j: X.features[i, j] for j in xrange(D)}
    return freqX, reprX

# read input parameters
if __name__ == '__main__':
    N = int(sys.argv[1])
    D = int(sys.argv[2])
    Nseed = int(sys.argv[3])
    # make X,Y mock data
    (X, Y, pi) = make(N, Nseed, D)
    seed = [(i, i) for i in xrange(Nseed)]
    # write to CSV files

    freqX, reprX = getFreqRepr(X, D)
    freqY, reprY = getFreqRepr(Y, D)
    IO.writePickledWords('pockX.txt', freqX, reprX)
    IO.writePickledWords('pockY.txt', freqY, reprY)

    IO.writeWords('mockX.txt', X)
    IO.writeWords('mockY.txt', Y)
    IO.writeSeed('seedXY.txt', seed)
    print X.asTuple()
    # now need to save


