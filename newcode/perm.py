from common import *


def ID(N):
    return np.arange(N)


def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


# computes the Hamming distance between p and q
# = counts the number of indices on which p and q differ
def hamming(p, q):
    s = 0
    N = len(p)
    for i in xrange(N):
        if p[i] != q[i]:
            s += 1
    return s


def argsort(seq):  # np.argsort was not stable?!
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    return sorted(range(len(seq)), key=seq.__getitem__)


def sort(v, reverse=False):
    v = np.array(v)
    I = argsort(v)
    if reverse:
        I = I[::-1]
    u = v[I]
    return u, I


def isID(pi):
    id = ID(len(pi))
    return np.linalg.norm(id-pi) == 0


def permuteList(v, pi):
    v = [v[i] for i in pi]
    return v


def randperm(list):
    list = np.random.permutation(list)
    return list


def isperm(pi):
    N = len(pi)
    u = set(pi)
    return len(u) == N and np.min(pi) == 0 and np.max(pi) == (N-1)


if __name__ == '__main__':
    I = [3, 4, 2, 5, 1]
    (U, J) = sort(I)
    (Urev, I) = sort(I, reverse=True)
    print U
    print Urev





