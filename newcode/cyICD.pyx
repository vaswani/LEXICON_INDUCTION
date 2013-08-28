#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np
import numpy as np

def cy_ichol(np.ndarray[double, ndim=2] K, double eta):
    cdef unsigned int i = 0
    cdef unsigned int j = 0
    cdef unsigned int jj = 0
    cdef unsigned int D
    cdef unsigned int best_i
    cdef unsigned int p_j

    cdef unsigned int N = len(K)
    cdef np.ndarray[double, ndim=1] d = np.diag(K).copy()

    cdef np.ndarray[np.int32_t, ndim=1] perm = np.zeros(N, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] nu = np.zeros(N)

    cdef np.ndarray[double, ndim=2] R = np.zeros((N, N))
    while j < N:
        best_i = np.argmax(d)
        if d[best_i] <= eta:
            break
        perm[j] = best_i
        p_j = perm[j]
        nu[j] = np.sqrt(d[p_j])
        for i in xrange(N):
            R[j,i] = K[i, p_j]
            for jj in xrange(j):  # inner summation.  R
                R[j, i] -= R[jj, i]*R[jj, p_j]
            R[j, i] /= nu[j]
            d[i] -= R[j, i]*R[j, i]
        # finally
        j += 1
    D = j
    return D, R, perm, nu

def cy_getRepresentations(np.ndarray[double, ndim=2] K, unsigned int D, np.ndarray[double, ndim=2] R, np.ndarray[np.int32_t, ndim=1] perm,  np.ndarray[double, ndim=1] nu):
    cdef unsigned int i = 0
    cdef unsigned int j = 0
    cdef unsigned int jj = 0
    cdef unsigned int p_j
    cdef unsigned int N = len(K)
    cdef np.ndarray[double, ndim=2] r = np.zeros((N, D))

    for j in xrange(D):
        p_j = perm[j]
        for i in xrange(N):
            r[i, j] = K[i, p_j]
            for jj in xrange(j):
                r[i, j] -= r[i, jj] * R[jj, p_j]
            r[i, j] /= nu[j]

    return r

