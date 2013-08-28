import common
import sys
import numpy as np
#from SparseFeatures import *  # Auxiliary class for computing kernels from sparse features representations
import CCAUtil as CU
import Lists
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from cyICD import cy_ichol
from cyICD import cy_getRepresentations


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
    def ichol_words(K, keys, eta):
        # W is a dict=>dict=>number
        # keys defines the sample ordering for the kernel (so, the first row of K corresponds o keys[1]
        # SparseFeatures wraps dict=>numbers as sparse feature vectors (providing functions like inner products)
        N = len(keys)
        print >> sys.stderr, keys
        print >> sys.stderr, "computing kernel for N =", N, "words"
        #import cProfile
        ##cProfile.runctx("K_X = SparseFeatures.getKernel(W, keys)", globals(), locals())
        #K = SparseFeatures.getKernel(W, keys) # computing kernel
        #print 'norm', common.norm(G-K)
        print >> sys.stderr, "ichol of", N, "words"
        #IO.writeNumpyArray('K0.txt', K)
        model = ICD.fast_ichol(K, eta)
        #model = ICD.ichol(K, eta)
        model.keys = keys
        R = np.mat(model.R)
        assert common.norm(R.T * R - K) < 1e-5
        return model

    @staticmethod
    def ichol(K, eta):
        #K1 = np.mat(IO.readPy('/tmp/matrices/K' + str(counter) + '.txt'))
        #K = K1
        #d = np.round(np.diag(K).copy())
        N = len(K)  # shape[0]
        #IO.pickle('/tmp/matrices/K' + str(N) + '.txt', K)
        d = np.diag(K).copy()

        j = 0
        perm = [0] * N
        nu = [0] * N
        R = Lists.zeros2(N, N)
        while j < N:
            best_i = np.argmax(d)
            if d[best_i] <= eta:
                break
            perm[j] = best_i
            p_j = perm[j]
            nu[j] = np.sqrt(d[p_j])
            for i in xrange(N):
                R[j][i] = K[i][p_j]
                for jj in xrange(j):  # inner summation.  R
                    R[j][i] -= R[jj][i]*R[jj][p_j]
                R[j][i] /= nu[j]
                d[i] -= R[j][i]*R[j][i]
            # finally
            j += 1
        D = j

        model = common.Struct()
        model.D = D
        model.R = R[:D]
        model.perm = perm[:D]
        model.nu = np.array(nu[:D])
        model.K = K
        return model


    @staticmethod
    def fast_ichol(K, eta):
        D, R, perm, nu = cy_ichol(np.array(K), eta)
        model = common.Struct()
        model.D = D
        model.R = R[:D]
        model.perm = perm[:D]
        model.nu = np.array(nu[:D])
        model.K = K
        return model


    @staticmethod
    def getRepresentations_words(model, K):
        #K = SparseFeatures.getKernel(W, keys, model.keys)
        #print 'norm', common.norm(G-K)
        F0 = ICD.fast_getRepresentations(model, K)
        #G0 = ICD.getRepresentations(model, K)
        #asd
        # F1 = np.zeros(F0.shape)
        # for i in xrange(l-en(keys)):
        #     word_i = keys[i]
        #     F1[i, :] = ICD.getRepn(model, K_X, i)
        return F0

    # @staticmethod
    # def getRepn(model, K_X, i):
    #     r = np.zeros(model.D)
    #     for j in xrange(model.D):
    #         r[j] = K_X[i, model.perm[j]]
    #         for jj in xrange(j):
    #             r[j] -= r[jj] * model.R[jj, model.perm[j]]  # R[jj][perm[j]];
    #         r[j] /= model.nu[j]
    #     return r

    @staticmethod
    def fast_getRepresentations(model, K):
        r = cy_getRepresentations(np.array(K), model.D, model.R, model.perm, model.nu)
        return np.mat(r)

    @staticmethod
    def getRepresentations(model, K):
        N = len(K)
        D = model.D
        r = Lists.zeros2(N, D)

        for j in xrange(D):
            p_j = model.perm[j]
            for i in xrange(N):
                r[i][j] = K[i][p_j]
                for jj in xrange(j):
                    r[i][j] -= r[i][jj] * model.R[jj][p_j]
                r[i][j] /= model.nu[j]

        return np.mat(r)
    #
    # @staticmethod
    # def getRepresentations_old(model, K_X):
    #     N = K_X.shape[0]
    #     D = model.D
    #     r = np.mat(np.zeros((N, D)))
    #
    #     for j in xrange(D):
    #         p_j = model.perm[j]
    #         r[:, j] = K_X[:, p_j]
    #         for jj in xrange(j):
    #             r[:, j] -= r[:, jj] * model.R[jj][p_j]
    #         r[:, j] /= model.nu[j]
    #
    #     return r

if __name__ == '__main__':
    nargs = len(sys.argv)
    if nargs == 1:
        # Test
        # setup data
        data = np.matrix('[1,3,1;1,4,1;1,-3,-5;2,2.5,2]')
        K = (data*data.T)
        K = K.tolist()
        eta = 0.01
        # ichol
        model = ICD.ichol(K, eta)
        model2 = ICD.fast_ichol(np.array(K), eta)

        print 'K', common.norm(model2.K - model.K)
        print 'R', common.norm(model2.R - model.R)
        print 'perm', np.linalg.norm(model2.perm - model.perm)
        print 'D', np.linalg.norm(model2.D - model.D)
        print 'nu', np.linalg.norm(model2.nu - model.nu)
        # setup out of sample data
        z0 = [1, 3, 1]  # same as the first row in data
        z1 = [3, -2, 1]
        z2 = [2, 1, 1]
        K_x0 = z0 * data.T
        K_x1 = z1 * data.T
        K_x2 = z2 * data.T
        K_X = np.mat([z0, z1, z2]) * data.T
        # project
        r0 = ICD.getRepresentations(model, K_x0.tolist())
        r1 = ICD.getRepresentations(model, K_x1.tolist())
        r2 = ICD.getRepresentations(model, K_x2.tolist())
        r = ICD.getRepresentations(model, K_X.tolist())
        t0 = ICD.fast_getRepresentations(model2, K_x0)
        t1 = ICD.fast_getRepresentations(model2, K_x1)
        t2 = ICD.fast_getRepresentations(model2, K_x2)
        t = ICD.fast_getRepresentations(model2, K_X)
        #output
        print 'r0', np.linalg.norm(r0-t0), r0
        print 'r1', np.linalg.norm(r1-t1), r1
        print 'r2', np.linalg.norm(r2-t2), r2
        print 'r', np.linalg.norm(r-t), r
        print 'eta', eta

        R = np.mat(model.R)
        model.RT = R.T
        # both should be roughly the same (maybe even exactly the same for r0-R0)
        assert common.norm(R.T.dot(R)-K) < 1e-10
        assert common.norm(model.RT[0, :] - r0) < 1e-20
        # 7/7/2013 TL: implementation seems to be working correctly.
    elif nargs == 2:
        print 'here'
    elif nargs == 3:
        filename1 = sys.argv[1]
        filename2 = sys.argv[2]
        W1 = IO.readPickledWords(filename1)
        W1.setupFeatures()
        eta = 0.001
        model1 = W1.ICD_representation(0, eta, 0)

        W2 = IO.readPickledWords(filename2)
        W2.setupFeatures()
        eta = 0.001
        model2 = W2.ICD_representation(0, eta, 1)

        tau = 0.001
        cca_model = CU.learn0(W1.features, W2.features, tau)
