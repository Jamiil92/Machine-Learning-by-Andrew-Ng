from __future__ import division
import numpy as np
from numpy import *

def sigmoid(z) :

# sigmoid function computes the sigmoid of z
# z can be a matrix vector or scalar

    g = np.zeros(shape(z))

    g = 1/(1+ np.exp(-z))

    return g
