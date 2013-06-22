#generates the data from the generative story
from common import *
import IO
import perm
import MatchingUtil as MU


def random_proj_shift(F):
    (_, D) = F.shape
    m = randn((1, D))
    S = randn((D, D))
    F = F.dot(S) + m
    return F


def make(N, D, eps):
    # generate random data in latent space
    Z = randn((N, D))
    # generate X and Y from Z.
    # they should be random transformations of Z with added noise.
    X = Words()
    Y = Words()
    # create a set of words with which it will be easy to run edit distance
    #X.words = ['a' + str(i) for i in xrange(Ns)] + ['b' + str(i) for i in xrange(Nr)]
    #Y.words = ['a' + str(i) for i in xrange(Ns)] + ['c' + str(i*2) for i in xrange(Nr)]
    X.words = np.array([str(i) for i in xrange(N)])
    Y.words = np.array([str(i) for i in xrange(N)])
    X.freq = np.array([i for i in xrange(N)])  # mock frequencies
    Y.freq = np.array([i for i in xrange(N)])
    X.features = Z + eps*randn((N, D))
    Y.features = Z + eps*randn((N, D))
    # create random permutation, but keep last (1-q) in place
    pi = perm.ID(N)
    topX = N  # int(0.92*N)
    pi = np.append(perm.randperm(pi[:topX]), pi[topX:])
    #X.G = np.zeros((N, N), dtype=np.float)
    #Y.G = np.zeros((N, N), dtype=np.float)

    X.features = random_proj_shift(X.features)
    Y.features = random_proj_shift(Y.features)

    # permute Y randomly
    Y = MU.permuteFirstWords(Y, pi)
    return X, Y, pi


# read input parameters
if __name__ == '__main__':
    N = int(sys.argv[1])
    D = int(sys.argv[2])
    eps = float(sys.argv[3])
    Nseed = int(sys.argv[4])
    # make X,Y mock data
    (X, Y, pi) = make(N, D, eps)
    seed = [(i, i) for i in xrange(Nseed)]
    # write to CSV files
    IO.writeWords('data/mockX.txt', X)
    IO.writeWords('data/mockY.txt', Y)
    IO.writeSeed('data/seedXY.txt', seed)

    # now need to save