from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import pylab
from pylab import *
import warnings
warnings.filterwarnings("ignore")

def plotData(X,y):

# plotData plots the data points X and y into a new figure
# with + for the positive examples and o for the negative examples.
# X is assumed to be a Mx2 matrix.

    plt.figure()
    # find the indices of positive and negative examples
    pos = find(y==1) ; neg = find(y == 0)
    # Plot examples for exam score 1(pos) and 2(neg)
    plt.plot(X[pos,0], X[pos,1], 'k+', linewidth = 2, markersize = 7)
    plt.plot(X[neg,0], X[neg,1], 'ko', color = "y", markersize = 7)
    
