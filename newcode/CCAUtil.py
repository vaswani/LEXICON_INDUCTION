from common import *
import numpy as np
import scipy.linalg


def symmetricSqrt(A):
    D, P = scipy.linalg.eigh(A)
    D = np.mat(diag(D))
    P = np.mat(P)
    assert isPSD(D, 0), 'given matrix is not PSD'

    return P * np.sqrt(D) * P.T


# primal CCA, with regularizer tau
def learn(X, Y, tau):
    (NX, Dx) = X.shape
    (NY, Dy) = Y.shape
    assert NX == NY

    # compute the covariance of [X;Y]
    Z = np.c_[X, Y]  # stack X Y by rows
    C = np.cov(Z.T)
    Sxx = np.mat((1-tau)*C[:Dx, :Dx] + tau*np.eye(Dx))
    Sxy = np.mat(C[:Dx, Dx:])
    Syy = np.mat((1-tau)*C[Dx:, Dx:] + tau*np.eye(Dy))

    # compute the matrix (Sxx)^-0.5 * Sxy * (Syy)^-0.5
    Rx = symmetricSqrt(Sxx)
    Ry = symmetricSqrt(Syy)
    E = Rx.I * Sxy * Ry.I
    Vx, p, VyT = scipy.linalg.svd(E)
    # note that Vx*diag(s, DX, DY)*VyT =~ E. To test run:
    # np.linalg.norm(U*common.diag(s, 3,4)*Vh - E)
    model = Struct()
    # compute and truncate the number of correlation coefficients
    model.Ux = scipy.linalg.solve(Rx, Vx)[:, :len(p)]  # Rx\Vx
    model.Uy = scipy.linalg.solve(Ry, VyT.T)[:, :len(p)]  # Ry\Vy
    model.mux = np.mean(X, 0)
    model.muy = np.mean(Y, 0)
    model.p = p
    model.P = np.mat(np.diag(p))
    return model


def project(model, X, Y):
    Z = Struct()
    M = np.mat(np.diag(model.p))
    Z.X = projectSingle(X, model.Ux, M, model.mux)
    Z.Y = projectSingle(Y, model.Uy, M, model.muy)
    return Z


def projectSingle(X, U,  P, mu):
    # compute latent representation z of each x
    # z = P' U'(x - mu)
    Z = P.T*U.T*(X - mu).T
    return Z.T


def myTest_model():
    N = 4
    Dx = 3
    Dy = 4
    tau = 0.01
    X = randn((N, Dx))
    Y = randn((N, Dy))
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



