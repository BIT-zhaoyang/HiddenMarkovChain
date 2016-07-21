# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:23:51 2016

@author: yangzhao
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from scipy.stats import multivariate_normal as multiGaussian
from numpy import newaxis

def estimateCov(X, mean, assign):
    numComponent, numDim = mean.shape
    cov = np.zeros((numComponent, numDim, numDim))
    nEleInCluster = assign.sum(axis=0)
    for i in range(numComponent):
        _X = X[assign[:, i] == 1]
        diff = _X - mean[i]
        cov[i] = diff.T.dot(diff) / nEleInCluster[i]

    return cov

def init(numComponent):
    _A = np.random.random((numComponent, numComponent))
    row_sum = _A.sum(axis=1);
    _A = _A / row_sum[:, newaxis]

    _pi = np.random.random((numComponent))
    _pi = _pi / _pi.sum()

    return _A, _pi

def E_Step(X, _mean, _cov, _A, _pi):
    numPoint = X.shape[0]
    numComponent = _pi.shape[0]

    ## compute multiGaussian.pdf(X, numComponent) at once which will save some computation
    pdf = np.zeros((numPoint, numComponent))
    for i in range(numPoint):
        for j in range(numComponent):
            pdf[i][j] = multiGaussian.pdf(X[i], mean=_mean[j], cov=_cov[j])

    ## we use rescaling technique in E-Step
    scaling = np.zeros(numPoint)

    ## compute alpha
    alpha = np.zeros((numPoint, numComponent))
    alpha[0] = pdf[0] * _pi  # compute starting condition alpha[0]
    scaling[0] = alpha[0].sum()  # scaling alpha[i]'s for numerical stability
    alpha[0] /= scaling[0]
    for i in range(1, numPoint):    # compute rest alpha
        alpha[i] = pdf[i] * (alpha[i-1].dot(_A))
        scaling[i] = alpha[i].sum()
        alpha[i] /= scaling[i]

    ## compute beta
    beta = np.ones((numPoint, numComponent))
    for i in range(numPoint-2, -1, -1):
        beta[i] = (beta[i+1]*pdf[i+1]).dot(_A.T)
        beta[i] /= scaling[i+1]

    ## compute gamma(z_n)
    gamma = alpha * beta
    # normalize gamma for numerical stability
    gamma /= gamma.sum(axis=1)[:, newaxis]

    ## compute epsilon(z_{n-1}, z_n)
    epsilon = np.zeros((numPoint-1, numComponent, numComponent))
    for i in range(numPoint-1):
        epsilon[i] = (alpha[i][:, newaxis] * _A) * pdf[i+1] * beta[i+1] / scaling[i+1]

    return gamma, epsilon

def M_Step(X, gamma, epsilon):
    numPoint, numDim = X.shape
    numComponent = gamma.shape[1]

    ## compute _pi
    _pi = gamma[0]

    ## compute _A
    numerator = epsilon.sum(axis=0)  # compute numerator
    denominator = numerator.sum(axis=1)  # compute denominator
    _A = numerator / denominator[:, newaxis]

    ## compute denominator in _mean and _cov
    denominator = gamma.sum(axis=0)

    ## compute _mean
    numerator = (gamma.T).dot(X)
    _mean = numerator / denominator[:, newaxis]

    ## compute _cov
    _cov = np.zeros((numComponent, numDim, numDim))
    for i in range(numComponent):
        numer1 = X - _mean[i]
        numer2 = gamma[:, i][:, newaxis] * numer1
        _cov[i] = numer1.T.dot(numer2) / denominator[i]

    return _mean, _cov, _A, _pi
