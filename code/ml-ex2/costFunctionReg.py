from __future__ import division, absolute_import, print_function
import numpy as np
from numpy import *

from sigmoid import sigmoid

def costFunctionReg(theta, X, y, lambdaa):

# costFunction computes the cost of using theta as the parameter for
# regularized logistic regression 

    # initialize some useful values
    m = y.size  # number of training examples

    J = 0  

    # Compute the cost of a particular choice of theta and set J to be the cost.

    h = sigmoid(X.dot(theta))

    theta[0] = 0

    J = (1/m)*np.sum(-y*np.log(h) - (1-y)*np.log(1-h)) + (lambdaa/(2*m))*(np.dot(theta.T, theta))
    grad = (1/m)*(X.T.dot(h-y)) + (lambdaa/m)*theta

    return J,grad

def costFunctionReg_0(theta, X, y, lambdaa):

# costFunction computes the cost of using theta as the parameter for
# regularized logistic regression 

    # initialize some useful values
    m = y.size  # number of training examples

    J = 0  

    # Compute the cost of a particular choice of theta and set J to be the cost.

    h = sigmoid(X.dot(theta))

    theta[0] = 0

    J = (1/m)*np.sum(-y*np.log(h) - (1-y)*np.log(1-h)) + (lambdaa/(2*m))*(np.dot(theta.T, theta))
    
    return J

def gradFunctionReg(theta, X, y, lambdaa) :

# gradFunction compute the gradient of the cost w.r.t to the parameters.

    m = y.size  # number of training examples

    grad = np.zeros(shape(theta))

    # Compute the partial derivatives and set grad to the partial derivatives
    # of the cost w.r.t each parameter in theta

    # Note grad should have the same dimension as theta

    h = sigmoid(X.dot(theta))

    #theta[0] = 0

    grad = (1/m)*(X.T.dot(h-y)) + (lambdaa/m)*theta

    return grad
