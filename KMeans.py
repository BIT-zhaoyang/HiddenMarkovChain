# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:42:26 2016

@author: yangzhao
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from scipy.stats import multivariate_normal as multiGaussian
from numpy import newaxis

"""
    This is a K-Means implementation by myself. I tried to optimize the algorithm by using matrix multiplications rather than iterations.
    This implementation has given me some surprise so that I realized I was wrong about K-Means before. I used to think one limitation of K-Means is that it forms clusters in a ball shape. Namely, the border points should roughly form a circle focused on the center of this cluster(outliers are not considered). However, after this implementation, I realized that it is not the case. The border can be a straight line and this straight line is perpendicular to the link between two cluster centers! This is in fact very intuitive. Because when we assign a point to a cluster, we just choose the nearest center. Thus, points on this line have the same distance to both centers!
"""



def computeError(X, mean, numCluster, numDim, numPoint):
    """ Compute error and update assignment in K-Means. """
    nEleInCluster = np.zeros(numCluster)
    totalXY = np.zeros((numCluster, numDim))
    error = 0
    for i in range(numPoint):
        ele = X[i]
        dist = mean - ele
        dist = np.sqrt((dist * dist).sum(axis=1))

        choose = np.argmin(dist)
        nEleInCluster[choose] += 1
        totalXY[choose] += ele

        error += dist[choose]

    mean[:] = totalXY / nEleInCluster[:, newaxis]
    return error

def getAssign(X, assign, mean):
    numPoint = X.shape[0]
    numCluster = mean.shape[0]
    for i in range(numPoint):
        ele = X[i]
        dist = mean - ele
        dist = np.sqrt((dist * dist).sum(axis=1))
        choose = np.argmin(dist)
        assign[i][choose] = 1

    return assign

def KMeans(X, numCluster):
    """
    Given an N*D matrix X consists of N datapoints of D dimensions and an intial guess of mean about each cluster, return the optimal mean. This function is a 0-1 hard assignment of K-Means without  the ability to evaluate number of clusters.
    """
    numPoint, numDim = X.shape
    randIdx = np.random.random_integers(low=0, high=numPoint, size = numCluster)
    mean = X[randIdx, :]
#    mean = np.random.random((numCluster, numDim))
#    nIter = 40
    assign = np.zeros((numPoint, numCluster))


    lastError = computeError(X, mean, numCluster, numDim, numPoint)
    while True:
        currError = computeError(X, mean, numCluster, numDim, numPoint)
        diff = currError - lastError
        if(diff / (lastError + 1.e-9) < 0.01):
            break
        else:
            lastError = currError

    getAssign(X, assign, mean)

    return mean, assign
