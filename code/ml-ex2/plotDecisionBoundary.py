from __future__ import division, absolute_import, print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
#from mpl_toolkits.mplot3d import Axes3D

from plotData import plotData
from mapFeature import mapFeature

def plotDecisionBoundary(theta, X, y):
# plotDecisionBoundary plots the data points with + for the positive examples
# and o for the negative examples. X is assumed to be a either
# 1) Mx3 matrix, where the first column is an all-ones column for the intercept.
# 2) MxN, N>3 matrix, where the first column is all-ones

# Plot the data

    plotData(X[:,1:3], y)

    if X.shape[1] <= 3:
        # only need 2 points to define a line, so choose two endpoints
        plot_x = array([np.min(X[:,1])-2, np.max(X[:,1])+2])

        # Calculate the decision boundary line

        plot_y = array((-1/theta[2])*(theta[1]*plot_x + theta[0]))

        # Plot and adjust axes for better viewing

        plt.plot(plot_x, plot_y)

        # Legend, specific

        # plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 105, 30, 105])
    else:
        # Here is the grid range

        u = linspace(-1,1.5,50)
        v = linspace(-1,1.5,50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid

        for i in range(u.size):
            for j in range(v.size):                
                z[i,j] = np.dot(mapFeature(u[i].reshape((1,1)), v[j].reshape(1,1)),theta)

        z = z.T # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify range [0,0]
        
        plt.contour (u, v, z,[0], linewidth = 2)
        
        

