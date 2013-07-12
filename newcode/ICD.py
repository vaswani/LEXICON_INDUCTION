from common import *
from SparseFeatures import * # Auxiliary class for computing kernels from sparse features representations


# A static class that implements Hardoon et al.'s  Incomplete Cholesky Decomposition of a kernel,
# as well as projection of out-of-sample points.
#
# One possible use case is to find a lower-dimensional representation of the data that preserves inner products.
# That is, we are given (x1..xN) -  N samples in a high dimensional feature space of dimension D, where D >> N
# we want to find Z, a new representation of X in a lower dimension K <= N such that
# the linear kernels ZZ' ~= K (for example, the linear kernel K = XX')
#
# Furthermore, given an unseen sample x0, we want to compute its new representation z0
#
# The first is done using ichol() (model.RT is the new NxK representation), which performs
# Incomplete Cholesky Decomposition of the kernel (Rte that the kernel can be non-PSD)
# See Algorithm 2 in the reference below.
#
# The second is done with getRepresentations(). See Algorithm 3 in the reference below.
#
# Reference:
# Canonical Correlation Analysis: An Overview with Application to Learning Methods
# David R. Hardoon, Sandor Szedmak, John Shawe-Taylor
# http://www.davidroihardoon.com/Professional/Publications_files/NC_Hardoon_2817_reg.pdf
#
#
# July 2013, Tomer Levinboim
class ICD:
    @staticmethod
    def ichol_words(W, keys, eta):
        # W is a dict=>dict=>number
        # keys defines the sample ordering for the kernel (so, the first row of K corresponds o keys[1]
        # SparseFeatures wraps dict=>numbers as sparse feature vectors (providing functions like inner products)
        N = len(keys)
        print >> sys.stderr, "ichol of", N, "words"
        print keys
        K = SparseFeatures.getKernel(W, keys)
        model = ICD.ichol(K, eta)
        model.keys = keys
        assert norm(model.RT.dot(model.R) - K) < 1e-5
        return model

    @staticmethod
    def ichol(K, eta):
        d = np.diag(K).copy()
        N = K.shape[0]
        j = 0
        perm = [0] * N
        nu = [0] * N
        R = np.zeros((N, N))
        while j < N:
            best_i = np.argmax(d)
            if d[best_i] <= eta:
                break
            #R.append(np.zeros((1, N))) # it is already initialized.
            perm[j] = best_i
            p_j = perm[j]
            nu[j] = np.sqrt(d[p_j])
            for i in xrange(N):
                R[j, i] = K[i, p_j]
                for jj in xrange(j):  # inner summation.  R
                    R[j, i] -= R[jj, i]*R[jj, p_j]

                R[j, i] /= nu[j]
                d[i] -= R[j, i]*R[j, i]
            # finally
            j += 1
        D = j

        model = Struct()
        model.D = D
        model.R = R[:D, :]
        model.perm = perm[:D]
        model.nu = nu[:D]
        model.K = K
        model.RT = model.R.T
        return model

    @staticmethod
    def getRepresentations_words(model, W, keys):
        K_X = SparseFeatures.getKernel(W, keys, model.keys)
        return ICD.getRepresentations(model, K_X)

    @staticmethod
    def getRepresentations(model, K_X):
        N = K_X.shape[0]
        D = model.D
        r = np.mat(np.zeros((N, D)))

        for j in xrange(D):
            p_j = model.perm[j]
            r[:, j] = K_X[:, p_j]
            for jj in xrange(j):
                r[:, j] -= r[:, jj] * model.RT[p_j, jj]
            r[:, j] /= model.nu[j]

        return r

if __name__ == '__main__':
    # Test
    # setup data
    data = np.matrix('[1,3,1;1,4,1;1,-3,-5;2,2.5,2]')
    K = data*data.T
    eta = 0.01
    # ichol
    model = ICD.ichol(K, eta)
    # setup out of sample data
    z0 = [1, 3, 1]  # same as the first row in data
    z1 = [3, -2, 1]
    z2 = [2, 1, 1]
    K_x0 = z0 * data.T
    K_x1 = z1 * data.T
    K_x2 = z2 * data.T
    K_X = np.mat([z0, z1, z2]) * data.T
    # project
    r0 = ICD.getRepresentations(model, K_x0)
    r1 = ICD.getRepresentations(model, K_x1)
    r2 = ICD.getRepresentations(model, K_x2)
    r = ICD.getRepresentations(model, K_X)
    #output
    print 'r0', r0
    print 'r1', r1
    print 'r2', r2
    print 'r', r
    print 'eta', eta
    print 'nu', model.nu
    print 'perm', model.perm
    print 'RT', model.RT
    R = model.R
    # both should be roughly the same (maybe even exactly the same for r0-R0)
    print 'norm', norm(R.T.dot(R)-K)
    print 'norm r0-R0', norm(model.RT[0, :] - r0)
    # 7/7/2013 TL: implementation seems to be working correctly.