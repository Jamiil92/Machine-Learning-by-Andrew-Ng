from __future__ import division, absolute_import, print_function
import numpy as np
from numpy import *
from sigmoid import sigmoid

def predict(theta, X):

# predict function computes the predictions for X using a threshold at 0.5
# (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    m = X.shape[0] # Number of training examples

    p = np.zeros((m,))

    p = (sigmoid(X.dot(theta)) >= 0.5).astype(np.int) # you should set p to a vector of 0's and 1's

    return p
