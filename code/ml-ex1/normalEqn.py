from __future__ import division
import numpy as np
from numpy import *


def normalEqn (X,y) :
# normalEqn computes the closed-form solution to linear regression
# using the normal equations


    theta = np.zeros((X.shape[-1],))

    multi = X.T.dot(X)
    theta = np.linalg.pinv(multi).dot(X.T).dot(y)


    return theta
