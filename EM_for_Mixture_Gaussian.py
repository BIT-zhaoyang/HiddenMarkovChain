# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:26:49 2016

@author: yangzhao
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from scipy.stats import multivariate_normal as multiGaussian
from numpy import newaxis

def compute_responsibility(mean, cov, pi, x):
    ''' Compute responsiblity according to (9.13).
    The mean should be an array of 2 dimensions. The last dimension contains
    the mean of each component. These means are listed vertically in the first
    dimension. Similiarly, cov is an array of 3 dimensions. pi(prior probability)
    is an array of 1 dimension. x is the data to be fitted.

    The returned value is a one dimensional array. Containing the gamma for each
    component of the given data x.
    '''
    nComponent, dataDim = mean.shape
    gamma = np.zeros(nComponent)

    for i in range(nComponent):
        gamma[i] = multiGaussian.pdf(x, mean[i], cov[i])

    normalizer = gamma.sum()
    gamma = gamma / normalizer
    return gamma

def compute_Nk(respon):
    nComponent = respon[0].size
    N_k = np.zeros(nComponent)
    N_k = respon.sum(axis = 0)

    return N_k

def update_mean(respon, X, N_k):
    nComponent = respon[0].size
    dataDim = X[0].size
    mean = np.zeros((nComponent, dataDim))

    mean = respon.T.dot(X)
    for i in range(nComponent):
        mean[i] = mean[i] / N_k[i]

    return mean

def update_cov(respon, mean, X, N_k):
    nComponent = respon[0].size

    N, dataDim = X.shape
    mean = np.repeat(mean, N, axis = 0)
    cov = np.zeros( (nComponent, dataDim, dataDim) )
    for i in range(nComponent):
        Y = X - mean[i*N:(i+1)*N]
        cov[i] = (Y.T * respon[:, i]).dot(Y) / N_k[i]

    return cov

def update_pi(X, N_k):
    N, ignore = X.shape
    N_k = N_k / N
    return N_k

#%% Generate data
mean = np.array([0, 0])
cov = np.array([ [1, 1], [1, 3] ])
X1 = np.random.multivariate_normal(mean, cov, 300)
plt.scatter(X1[:, 0], X1[:, 1], c = 'r')

mean = np.array([3, 0])
cov = np.array([ [1, -1], [-1, 3] ])
X2 = np.random.multivariate_normal(mean, cov, 500)
plt.scatter(X2[:, 0], X2[:, 1], c = 'g')

mean = np.array([6, 3])
cov = np.array([ [1, 1], [1, 3] ])
X3 = np.random.multivariate_normal(mean, cov, 200)
plt.scatter(X3[:, 0], X3[:, 1], c = 'b')

X = np.vstack( (X1, X2, X3) )

#%% Train model

# Initialize parameters
mean = np.random.random([3, 2])
cov = np.identity(2)
cov = cov[newaxis, :]
cov = np.vstack((cov, cov, cov))
pi = [0.3, 0.3, 0.4]
respon = np.zeros([1000, 3])

# EM
for iter in range(51):
    # E step
    for i in range(1000):
        respon[i] = compute_responsibility(mean, cov, pi, X[i])

    if iter % 10 == 0:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c = respon)

    # M step
    N_k = compute_Nk(respon)
    mean = update_mean(respon, X, N_k)
    cov = update_cov(respon, mean, X, N_k)
    pi = update_pi(X, N_k)
