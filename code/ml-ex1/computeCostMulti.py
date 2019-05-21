from __future__ import division 
from numpy import *
import numpy as np 

def computeCostMulti(X, y, theta):
# computeCost compute cost for linear regression to fit the data point in X and Y

    m = y.size
    J = 0
    

    hypothesis = X.dot(theta)
    error = hypothesis - y 
    error_square = np.square(error, dtype = float)
    J = (1/(2*m))* np.sum(error_square) 
    return(J)
