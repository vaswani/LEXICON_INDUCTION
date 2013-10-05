#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np
import numpy as np

DEF INFTY = float('inf')

# Approximate minimum weighted matching on the input C
# returns the cost = \sum C[ i, pi[i] ]
def cy_ApproxMatch(np.ndarray[double, ndim=2] C):
    cdef unsigned int N = C.shape[1]
    cdef np.ndarray[np.int32_t, ndim=1] left = np.zeros(N, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] right = np.zeros(N, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] I = np.tile(np.arange(N, dtype=np.int32), [N, 1]).flatten()
    cdef np.ndarray[np.int32_t, ndim=1] J = np.tile(np.arange(N, dtype=np.int32), [N, 1]).T.flatten()

    cdef np.ndarray[np.int32_t, ndim=1] sigma = np.array(np.argsort(C.flatten()), dtype=np.int32)
    #sigma = sigma.flat
    cdef np.ndarray[np.int32_t, ndim=1] pi = np.zeros(N, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] edge_cost = np.zeros(N)

    cdef size_t e
    cdef size_t i, j

    #for element in sigma.flat:
    for e in xrange(sigma.shape[0]):
        i = I[sigma[e]]
        j = J[sigma[e]]
        if left[i] == 0 and right[j] == 0:
            pi[j] = i
            edge_cost[j] = C[j, i]
            left[i] = 1
            right[j] = 1
    cdef double cost = np.sum(edge_cost)
    #print cost
    return cost, pi, edge_cost


# Approximate minimum weighted matching on the input C
# returns the cost = \sum C[ i, pi[i] ]
def cy_min_submatrix(np.ndarray[double, ndim=2] U, rowX, rowY):
    cdef size_t i, j, m, n
    cdef double current_min = INFTY

    for i in rowX:
        for j in rowY:
            if current_min > U[i, j]:
                current_min = U[i, j]
                m = i
                n = j
    return current_min, m, n

# this code is much faster than the one above only when rowX and rowY are not small arrays
# Approximate minimum weighted matching on the input C
# returns the cost = \sum C[ i, pi[i] ]
def cy_min_submatrix2(np.ndarray[double, ndim=2] U, np.ndarray[np.int32_t, ndim=1] rowX, np.ndarray[np.int32_t, ndim=1] rowY):
    cdef size_t i, j, ii, jj
    cdef double current_min = INFTY
    cdef double v

    cdef long NX = rowX.shape[0]
    cdef long NY = rowY.shape[0]
    for i in xrange(NX):
        ii = rowX[i]
        for j in xrange(NY):
            jj = rowY[j]
            v = U[ii, jj]
            if current_min > v:
                current_min = v
    return current_min

def cy_getGraphMinDist(np.ndarray[double, ndim=2] GX, np.ndarray[double, ndim=2] GY,  np.ndarray[double, ndim=2] U):
    cdef size_t n, m
    cdef unsigned int N = GX.shape[0]
    rows_X = [[m for m in np.nonzero(GX[n, :])[0]] for n in xrange(N)]
    rows_Y = [[m for m in np.nonzero(GY[n, :])[0]] for n in xrange(N)]
    cdef np.ndarray[double, ndim=2] Z = np.mat(np.zeros((N, N)))
    cdef np.ndarray[np.int32_t, ndim=2] IX = np.mat(np.zeros((N, N)), dtype='i4')
    cdef np.ndarray[np.int32_t, ndim=2] IY = np.mat(np.zeros((N, N)), dtype='i4')
    #rows_X = [np.array([j for j in np.nonzero(GX[i, :])[0]], dtype='i4') for i in xrange(N)]
    #rows_Y = [np.array([j for j in np.nonzero(GY[i, :])[0]], dtype='i4') for i in xrange(N)]
    for n in xrange(N):
        rowX = rows_X[n]
        for m in xrange(N):
            #Z[n, m] = cy_min_submatrix2(U, rowX, rows_Y[m])
            Z[n, m], IX[n,m], IY[n,m] = cy_min_submatrix(U, rowX, rows_Y[m])
    return Z, IX, IY