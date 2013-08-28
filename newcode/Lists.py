
def zeros2(N, D):
    return [[0] * D for i in xrange(N)]


def T(A): ## (transpose) warning! returns a list of tuples!
    return zip(*A)


#def fromMat(M):
#    return M.tolist()
