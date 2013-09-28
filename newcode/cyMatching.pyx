#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np
import numpy as np

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
    cdef unsigned int i, j
    cdef double m = U[0, 0]

    for i in rowX:
        for j in rowY:
            if m > U[i, j]:
                m = U[i, j]
    return m


# Approximate minimum weighted matching on the input C
# returns the cost = \sum C[ i, pi[i] ]
def cy_min_submatrix2(np.ndarray[double, ndim=2] U, np.ndarray[long, ndim=1] rowX, np.ndarray[long, ndim=1] rowY):
    cdef unsigned int i, j, ii, jj
    cdef double m = U[0, 0]

    for i in range(rowX.shape[0]):
        ii = rowX[i]
        for j in range(rowY.shape[0]):
            jj = rowY[j]
            if m > U[rowX[i], rowY[j]]:
                m = U[rowX[i], rowY[j]]
    return m