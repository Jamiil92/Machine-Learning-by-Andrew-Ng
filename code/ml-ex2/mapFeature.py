from __future__ import division
import numpy as np
from numpy import *

def mapFeature(X1, X2):
# mapFeature feature mapping function to polynomial features
#
#   mapFeature (X1, X2) maps the two input features
#   to quadratic features used in regularization exeercise

#   Returns a new feature array with more features, comprising of
#   X1, X2, X1^2, X2^2, X1*X2, X1*X2^2, etc..
#
#   Inputs X1, X2 must be the same size

    degree = 6
    out = np.ones((X1.shape[0],1))
    for i in range (1,degree+1,1):
        for j in range(0,i+1,1):
            out = np.append(out, np.multiply(pow(X1,(i-j)),pow(X2, j)).reshape((X1.shape[0],1)), axis = 1)
            
    return out         
