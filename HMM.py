# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:05:12 2016

@author: yangzhao

This file is a detailed implementation of HMM described in PRML. THe files
contains three modules: 
    1. data generation
    2. K-means for preprocess parameter selection
    3. Alpha-Beta algorithm for learning parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import helpers
from scipy.stats import multivariate_normal as multiGaussian
from numpy import newaxis
from KMeans import KMeans


#%% Data Generation Module
"""
The generated data consists of 'numPoint' points from 'numComponent' components. The transition matrix and emission probability are given by the user. Here we use 2-dimensional gaussian as the emission pdf.
"""

numPoint = 1000
numComponent = 3
numDim = 2

# transition matrix and emission parameters are given by predifined
A = np.array([[0.9, 0.1, 0.0],
              [0.0, 0.9, 0.1],
              [0.1, 0.0, 0.9]])
mean = np.array([ [-2, 0],
                  [3, 0],
                  [7, 3] ])
cov = np.array([ [[1, 1],
                  [1, 3]],
                 [[1, -1],
                  [-1, 3]],
                 [[1, 1],
                  [1, 3]] ])

# generate data
# use the uniform distribution for the choosing the first component
Z = np.zeros((numPoint), dtype = np.int8)
X = np.zeros((numPoint, numDim))

Z[0] = np.random.uniform(low = 0, high = numComponent)
X[0] = np.random.multivariate_normal(mean[Z[0]], cov[Z[0]])

for i in range(1, numPoint):
    last = Z[i-1]
    curr = np.random.choice(numComponent, 1, p = A[last])
    Z[i] = curr
    X[i] = np.random.multivariate_normal(mean[Z[i]], cov[Z[i]])

plt.scatter(X[:, 0], X[:, 1], c = Z)
plt.title("Original Data and Their Classes")

#%% K-Means for parameter initialization
"""
Next part is training the HMM. Training consists of two phases: E Step and M Step. To start, we need to initialize the estimated parameters before our first iteration for E Step.
This module uses K-Means to initialize mean and cov. Transition matrix A and distribution for the first component is initialized randomly. All estimated parameters starts with an underscore then follows the same name of the parameters as we used previously.
"""

# K-Means to estimate _mean and _cov
_mean, assign = KMeans(X, numComponent)
_cov = helpers.estimateCov(X, _mean, assign)

# show clustering result
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = assign)
plt.scatter(_mean[:, 0], _mean[:, 1], s = 40, color = 'y', marker = 'D')
plt.title("Clustering Result by K-Means")

#%% Train HMM
# initialize parameters
_A, _pi = helpers.init(numComponent)

# E Step
# compute alpha
helpers.E_Step(X, _mean, _cov, _A, _pi)


# S Step
