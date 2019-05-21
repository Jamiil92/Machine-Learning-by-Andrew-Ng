from __future__ import division
import numpy as np


def featureNormalize(X):
# featureNormalize returns a normalized version of X
# where the mean value of each feature is 0 and the standard deviation is 1.
# This is often good preprocessing step to do when working with learning algorithms.

    X_norm = X
    mu  = np.zeros((X.shape[-1],))
    sigma = np.zeros((X.shape[-1],))

    # Compute the mean

    mu = np.mean(X, axis = 0)

    # Compute the standard deviation

    sigma = np.std(X, axis = 0)

    # returns the number of rows in X

    m = X.shape[-1]

    # Compute the normalization

    mu_matrix = np.ones((m,))*(mu)

    sigma_matrix = np.ones((m,))*(sigma)

    X_substract = X - mu_matrix
    X_norm = X_substract / sigma_matrix
 
    return (X_norm, mu, sigma)

