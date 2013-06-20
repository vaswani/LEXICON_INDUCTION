from common import *


def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def hamming(p, q):
    s = 0
    N = len(p)
    for i in xrange(N):
        if p[i] != q[i]:
            s += 1
    return s


def sort(V, reverse=False):
    N = len(V)
    Z = zip(V, np.arange(N))
    Z.sort()
    U, I = zip(*Z)
    if reverse:
        I = I[::-1]
        U = U[::-1]
    return np.array(U), np.array(I)


def permuteList(v, pi):
    v = [v[i] for i in pi]
    return v


def randperm(list):
    list = np.random.permutation(list)
    return list


def isperm(pi):
    N = len(pi)
    u = uniq(pi)
    return len(u) == N and np.min(pi) == 0 and np.max(pi) == (N-1)


def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]


if __name__ == '__main__':
    I = [3, 4, 2, 5, 1];
    (U, J) = sort(I)
    (Urev,I ) = sort(I, reverse=True)
    print U
    print Urev




