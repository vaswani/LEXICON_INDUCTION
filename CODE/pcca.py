import math
import numpy
import scipy
import scipy.linalg
from collections import defaultdict

class MVNormal(object):
    def __init__(self, mu, sigma):
        """
        cholesky decomposition

        sigma = LL'
        det(sigma) = det(L) det(L')
        det(sigma) = 2 det(L)
        det(sigma) = 2 \prod diagoal(L)
        log det(sigma) = 2 \sum log diagoal(L)
        """
        self.mu = mu
        self.dim = len(mu)
        self.sigma = numpy.real_if_close(sigma, tol=10000)
        self.lower = numpy.linalg.cholesky(self.sigma)
        self.lower_inv = scipy.linalg.solve_triangular(self.lower, numpy.eye(self.dim), lower=True)
        self.log_z = 0.5 * self.dim * math.log(2*math.pi) + numpy.sum(numpy.log(numpy.diag(self.lower)))

    def __getitem__(self, x):
        return self.value(x)

    def value(self, x):
        return math.exp(self.log_value(x))

    def log_value(self, x):
        """
        triangular matrix problem

        (x-mu)' sigma^{-1} (x-mu)
        = (x-mu)' (LL')^{-1} (x-mu)
        = (x-mu)' L^{-1}'L^{-1} (x-mu)
        = (L^{-1}(x-mu))' L^{-1}(x-mu)
        = v'v
        where v = L-1(x - mu)

        """
        v = self.lower_inv.dot(x-self.mu)
        return -0.5 * v.T.dot(v) - self.log_z


    def marginal(self, i, j):
        return MVNormal(self.mu[i:j], self.sigma[i:j,i:j])

def sqrtm(M): 
    D, P = numpy.linalg.eig(M) # M ~ symmetric positive-definite

    """
    Eigendecomposition
    
    M = P * D * P-1 
    sqrt(M) = P * sqrt(D) * P-1 
    """

    P_inv = numpy.linalg.inv(P)
    D = numpy.diag(D)
    D_sqrt = numpy.sqrt(D)

    return P.dot(D_sqrt).dot(P_inv)

def pcca(mu_sigma, regularization_coefficient = 0.0, dim_latent = 0):
    mu_x = mu_sigma['mu_x']
    mu_y = mu_sigma['mu_y']
    dim_x = mu_x.shape[0]
    dim_y = mu_y.shape[0]
    dim_latent = dim_latent and dim_latent or min(dim_x, dim_y)

    sigma_xx = mu_sigma['sigma_xx'] + regularization_coefficient * numpy.eye(dim_x)
    sigma_xy = mu_sigma['sigma_xy']
    sigma_yy = mu_sigma['sigma_yy'] + regularization_coefficient * numpy.eye(dim_y)
    
    sigma_xx_inv = numpy.linalg.inv(sigma_xx)
    sigma_yy_inv = numpy.linalg.inv(sigma_yy)
    sigma_xx_invsqrt = sqrtm(sigma_xx_inv)
    sigma_yy_invsqrt = sqrtm(sigma_yy_inv)


    m = sigma_xx_invsqrt.dot(sigma_xy.dot(sigma_yy_invsqrt))
    v_x, correlations, v_y = scipy.linalg.svd(m)
    print correlations
    correlations = numpy.diag(correlations[0: dim_latent])
    v_y = v_y.T
    
    u_x = numpy.real_if_close(sigma_xx_invsqrt.dot(v_x))
    u_y = numpy.real_if_close(sigma_yy_invsqrt.dot(v_y))
    u_x = u_x[:, :dim_latent]
    u_y = u_y[:, :dim_latent]

    """
    Probabilistic interpretation of CCA

    z ~ N(0, I)
    x|z ~ N(w_x z + mu_x, psi_x)
    y|z ~ N(w_y z + mu_y, psi_y)
    """

    w_x = sigma_xx.dot(u_x.dot(numpy.sqrt(correlations)))
    w_y = sigma_yy.dot(u_y.dot(numpy.sqrt(correlations)))
    psi_x = sigma_xx - w_x.dot(w_x.T)
    psi_y = sigma_yy - w_y.dot(w_y.T)

    """
    Marginalize out z
    """

    sigma_xy = w_x.dot(w_y.T)
    sigma_yx = w_y.dot(w_x.T)

    mu = numpy.vstack((mu_x, mu_y))
    sigma = numpy.asarray(numpy.bmat([[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]]))

    """
    Debug

    u_x = u_x[:, :2]
    u_y = u_y[:, :2]
    correlations =  correlations[:2, :2]
    w_x_inv = correlations.T.dot(u_x.T)
    w_y_inv = correlations.T.dot(u_y.T)
    LatentX = lambda x : w_x_inv.dot(x - mu_x)
    LatentY = lambda y : w_y_inv.dot(y - mu_y)
    """

    return MVNormal(mu, sigma)

def estimate_means_covariances(x_vectors, y_vectors, weights):
    dim_x = x_vectors.itervalues().next().shape[0]
    dim_y = y_vectors.itervalues().next().shape[0]
    weight_x = defaultdict(float)
    weight_y = defaultdict(float)
    weight_total = 0.0
    for x in weights:
        for y in weights[x]:
            weight_x[x] += weights[x][y]
            weight_y[y] += weights[x][y]
            weight_total += weights[x][y]

    # estimate the means
    mu_x = numpy.zeros((dim_x, 1))
    mu_y = numpy.zeros((dim_y, 1))
    for x in weight_x:
        mu_x += weight_x[x] * x_vectors[x]
    for y in weight_y:
        mu_y += weight_y[y] * y_vectors[y]
    mu_x /= weight_total
    mu_y /= weight_total


    # estimate the covariances
    sigma_xx = numpy.zeros((dim_x, dim_x))
    sigma_xy = numpy.zeros((dim_x, dim_y))
    sigma_yy = numpy.zeros((dim_y, dim_y))
    for x in weight_x:
        dev_x = x_vectors[x] - mu_x
        sigma_xx += weight_x[x] * numpy.outer(dev_x, dev_x)

    for y in weight_y:
        dev_y = y_vectors[y] - mu_y
        sigma_yy += weight_y[y] * numpy.outer(dev_y, dev_y)

    for x in weights:
        for y in weights[x]:
            dev_x = x_vectors[x] - mu_x
            dev_y = y_vectors[y] - mu_y
            sigma_xy += weights[x][y] * numpy.outer(dev_x, dev_y)


    sigma_xx /= weight_total
    sigma_xy /= weight_total
    sigma_yy /= weight_total
    

    # pack the parameters
    parameters = {}
    parameters['mu_x'] = mu_x
    parameters['mu_y'] = mu_y
    parameters['sigma_xx'] = sigma_xx
    parameters['sigma_xy'] = sigma_xy
    parameters['sigma_yy'] = sigma_yy
    
    return parameters
