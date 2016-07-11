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
    
    # compute multiGaussian.pdf(X, numComponent) at once which will save some computation    
    pdf = np.zeros((numPoint, numComponent))
    for i in range(numPoint):
        for j in range(numComponent):
            pdf[i][j] = multiGaussian.pdf(X[i], mean=_mean[j], cov=_cov[j])
    
    # compute alpha
    alpha = np.zeros((numPoint, numComponent))
    alpha[0] = pdf[0] * _pi  # compute starting condition alpha[0]    
    for i in range(1, numPoint):    # compute rest alpha
        alpha[i] = (pdf[i] * alpha[i-1]).dot(_A)
    
    # compute beta
    beta = np.ones((numPoint, numComponent))
    for i in range(numPoint-2, -1, -1):
        beta[i] = (beta[i+1]*pdf[i+1]).dot(_A.T)
    
    # compute gamma(z)
    