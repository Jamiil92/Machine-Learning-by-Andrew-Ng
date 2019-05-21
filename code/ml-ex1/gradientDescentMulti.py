from __future__ import division 
from numpy import *
import numpy as np 

from computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, num_iters) :
#gradientDescentMulti(X, y, theta, alpha, num_iters) performs gradient descent to learn theta

    # Initialize some useful values
    m = y.size  # number of training examples
    J_history = np.zeros(num_iters) 

    for iter in  np.arange(num_iters) :

        hypothesis = X.dot(theta)

        error_vector = hypothesis - y
        
        theta = theta - alpha*(1/m)*(X.T.dot( error_vector))      

        # Save the cost in every iteration
        J_history[iter] = computeCostMulti(X, y, theta)

    return(theta, J_history)
