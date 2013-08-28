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
    cdef np.ndarray[np.int32_t, ndim=2] I0 = np.tile(np.arange(N, dtype=np.int32), [N, 1])
    cdef np.ndarray[np.int32_t, ndim=2] J0 = I0.T
    cdef np.ndarray[np.int32_t, ndim=1] I = I0.flatten()
    cdef np.ndarray[np.int32_t, ndim=1] J = J0.flatten()

    cdef np.ndarray[np.int32_t, ndim=1] sigma = np.array(np.argsort(C.flatten()), dtype=np.int32)
    #sigma = sigma.flat
    cdef np.ndarray[np.int32_t, ndim=1] pi = np.zeros(N, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] edge_cost = np.zeros(N)

    cdef unsigned int element, i, j

    for element in sigma.flat:
        #element = sigma[m]
        i = I[element]
        j = J[element]
        if left[i] == 0 and right[j] == 0:
            pi[j] = i
            edge_cost[j] = C[j, i]
            left[i] = 1
            right[j] = 1
    cdef double cost = np.sum(edge_cost)
    #print cost
    return cost, pi, edge_cost