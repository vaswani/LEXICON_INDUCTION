#generates the data from the generative story
from common import *
import IO
import perm


def random_rotate_shift(F):
    (_,D) = F.shape
    m = randn(1,D)
    S = randn(D,D)
    F = F.dot(S) + m
    return F

def make(N,D,eps):
    # generate random data in latent space
    Z = randn(N, D)
    # generate X and Y from Z.
    # they should be random transformations of Z with added noise.
    X = Words();
    Y = Words();

    X.words = np.array(xrange(N))
    Y.words = np.array(xrange(N))
    X.features = Z + eps*randn(N,D)
    Y.features = Z + eps*randn(N,D)
    # create random permutation, but keep last (1-q) in place
    pi = np.array(xrange(N))
    topX = int(0.92*N);
    pi = np.append(perm.randperm(pi[:topX]), pi[topX:])
    # TODO, genenrate graph. for now, keep it empty.
    X.G = np.zeros((N,N),dtype=np.float)
    Y.G = np.zeros((N,N),dtype=np.float)

    X.features = random_rotate_shift(X.features)
    Y.features = random_rotate_shift(Y.features)

    # permute Y
    Y.words = perm.permuteList(Y.words, pi)
    Y.features = Y.features[pi, ]
    Y.G = Y.G[pi,pi]
    return (X, Y, pi)


# read input parameters
N   = int(sys.argv[1])
D   = int(sys.argv[2])
eps = float(sys.argv[3])
# make X,Y mock data
(X, Y, pi) = make(N, D, eps)
# write to CSV files
IO.writeCSV('mockX.txt', X)
IO.writeCSV('mockY.txt', Y)
# now need to save