import common
import numpy as np
import scipy.linalg
import perm


def symmetricSqrt(A):
    D, P = scipy.linalg.eigh(A)
    D = np.mat(common.diag(D))
    P = np.mat(P)
    assert common.isPSD(D, 0), 'given matrix is not PSD'
    return P * np.sqrt(D) * P.T


def computeCorr(X, Y, covar_type, tau):
    X = np.mat(X)
    Y = np.mat(Y)
    (NX, Dx) = X.shape
    (NY, Dy) = Y.shape
    assert NX == NY
    N = NX

    if covar_type == 'outer':
        Cxx = X.T*X/N  # C[:Dx, :Dx]
        Cxy = X.T*Y/N  # C[:Dx, Dx:]
        Cyy = Y.T*Y/N  # C[Dx:, Dx:]
    else:
        Z = np.c_[X, Y]  # stack X Y by rows
        C = np.cov(Z.T)
        Cxx = C[:Dx, :Dx]
        Cxy = C[:Dx, Dx:]
        Cyy = C[Dx:, Dx:]

    Cxx = np.mat((1-tau)*Cxx + tau*np.eye(Dx))
    Cxy = np.mat(Cxy)
    Cyy = np.mat((1-tau)*Cyy + tau*np.eye(Dy))
    return Cxx, Cxy, Cyy


# primal CCA, with regularizer tau
def learn(X, Y, options):
    (Sxx, Sxy, Syy) = computeCorr(X, Y, options.covar_type, options.tau)

    # compute the matrix (Sxx)^-0.5 * Sxy * (Syy)^-0.5
    Ry = symmetricSqrt(Syy)
    Rx = symmetricSqrt(Sxx)
    E = Rx.I * Sxy * Ry.I

    Vx, p, VyT = scipy.linalg.svd(E)

    # note that Vx*diag(s, DX, DY)*VyT =~ E. To test run:
    # np.linalg.norm(U*common.diag(s, 3,4)*Vh - E)
    model = common.Struct()
    # compute and truncate the number of correlation coefficients
    Ux = (scipy.linalg.solve(Rx, Vx)[:, :len(p)])  # Rx\Vx
    Uy = (scipy.linalg.solve(Ry, VyT.T)[:, :len(p)])  # Ry\Vy

    #print common.norm(Ux.T*Sxx*Ux - np.eye(Ux.shape[0]), 2) < 1e-10
    #assert common.norm(Uy.T*Syy*Uy - np.eye(Uy.shape[0]), 2) < 1e-10
    #assert common.norm(Uy.T*Sxy.T*Ux - np.diag(p), 2) < 1e-10

    model.Ux = np.mat(Ux)
    model.Uy = np.mat(Uy)
    model.mux = np.mean(X, 0)  # they don't use the mean?
    model.muy = np.mean(Y, 0)
    model.p = p
    model.P = np.mat(np.diag(p))
    return model


def learn0(X, Y, options=None):
    # compute the covariance of [X;Y]
    (Sxx, Sxy, Syy) = computeCorr(X, Y, options.covar_type, options.tau)

    BC = Syy.I * Sxy.T
    E = Sxx.I * Sxy * BC

    [eigs, V] = np.linalg.eig(E)
    (eigs, I) = perm.sort(eigs, reverse=True)
    p = np.sqrt(eigs + 1e-10)
    V = V[:, I]
    Ux = V.T
    Uy = (BC * V * np.diag(1/p)).T
    model = common.Struct()
    model.p = p
    model.Ux = Ux
    model.Uy = Uy
    model.P = np.mat(np.diag(p))
    model.mux = np.mean(X, 0)  # currently unused
    model.muy = np.mean(Y, 0)

    return model


def project(options, model, X, Y):
    Z = common.Struct()
    M = np.mat(np.diag(np.sqrt(model.p)))
    Z.X = projectSingle(options, X, model.Ux, M, model.mux)
    Z.Y = projectSingle(options, Y, model.Uy, M, model.muy)
    return Z


def projectSingle(options, X, U,  P, mu):
    # compute latent representation z of each x
    # z = P' U'(x - mu)
    if options.projection_type == 'ignore_P_mu':
        Z = U.T*X.T
    else:
        Z = P.T*U.T*(X - mu).T  # TODO: can make this faster by using P as a vector

    return Z.T


def myTest_model():
    N = 4
    Dx = 3
    Dy = 4
    tau = 0.01
    X = common.randn((N, Dx))
    Y = common.randn((N, Dy))
    #X = np.eye(Dx)
    #Y = diag(np.ones(Dx), Dx, Dy)
    model = learn(X, Y, tau)
    # it should be the case that Ux*Sxx*Ux' = I and by symmetry for y too.
    (Ux, Uy) = (model.Ux, model.Uy)
    Z = np.c_[X, Y]  # stack X Y by rows
    C = np.cov(Z.T)
    Sxx = np.mat((1-tau)*C[:Dx, :Dx] + tau*np.eye(Dx))
    Syy = np.mat((1-tau)*C[Dx:, Dx:] + tau*np.eye(Dy))
    Sxy = np.mat(C[:Dx, Dx:])

    # validate that Ux is Sxx-orthonormal and Uy is Syy-orthonormal
    Jx = Ux.T * Sxx * Ux
    Jy = Uy.T * Syy * Uy
    Q = Uy.T * Sxy.T * Ux
    Np = len(model.p)
    bx = np.allclose(Jx, np.eye(Np))
    by = np.allclose(Jy, np.eye(Np))
    bp = np.allclose(Q, model.P.T)
    print 'pcca model test passed?', bx, by, bp

    Z = project(model, X, Y)
    print Z

if __name__ == '__main__':
    Z = myTest_model()
    # TODO: check the embedding is correct. How?



