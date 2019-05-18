from __future__ import division 
from computeCost import computeCost
from numpy import *
import numpy as np 

def gradientDescent(X, y, theta=array([0,0]), alpha=0.01, num_iters=1500) :
#gradientDescent(X, y, theta, alpha, num_iters) performs gradient descent to learn theta

    # Initialize some useful values
    m = y.size  # number of training examples
    J_history = np.zeros(num_iters) 

    for iter in  np.arange(num_iters) :

        hypothesis = X.dot(theta)

        error_vector = hypothesis - y
        
        theta = theta - alpha*(1/m)*(X.T.dot( error_vector))      

        # Save the cost in every iteration
        J_history[iter] = computeCost(X, y, theta)

    return(theta, J_history)
