##### Linear Regression #####

from __future__ import division 
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent



##### Part 1 : Basic Function

print("Running warmUpExercise .... \n")

print("5x5 Identity Matrix : \n")

warmUpExercise()

##### Part 2 : Scatter Plotting of our data

print('Plotting Data ... \n')

data = np.loadtxt("C:\\Python37\\Data\\ex1data1.txt", delimiter = ',') # read comma seperated data
X = data[:,0]; y = data[:,1] 
m = y.size  # number of training examples

#Plot Data
plt.figure(1)
plotData(X,y)

print('Program paused. Press enter to contine \n')
time.sleep(2) 


##### Part 3 : Cost and Gradient Descent

X = np.c_[np.ones((m,1)), data[:,0]] # add a column of ones to x
theta = np.zeros((2,1))  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ... \n')

# Compute and display initial cost

J = computeCost(X, y) 

print('With theta = [0 ; 0]\nCost computed = %f\n' %J)
print('Expected cost value (approx) 32.07\n') 


# further testing of the cost function

J = computeCost(X, y, [-1, 2])

print('With theta = [-1 ; 2]\nCost computed = %f\n' %J)
print('Expected cost value (approx) 52.24\n') 

print('Program paused. Press enter to contine \n')
time.sleep(2) 

print('Running Gradient Descent ... \n') 

# run gradient descent

theta, J_history = gradientDescent(X, y)

#print theta to screen
print('Theta found by gradient descent:\n')
print(theta.flatten()) 
print('Expected theat values (approx)\n') 
print(' -3.6303\t 1.1664\n\n') ;

# Plot the linear fit
plt.plot(X[:,1], X.dot(theta), 'b-')
plt.legend(['Training data', 'Linear regression'])

# Predict values for population sizes of 35,000 and 70,000

predict1 = theta.dot([1,3.5]) 

print("For population = 35,000, we predict a profit of \n" ,(predict1*10000))


predict2 = theta.dot([1,7]) 

print("For population = 70,000, we predict a profit of \n" ,(predict2*10000))

print('Program paused. Press enter to contine \n')
time.sleep(2) 

#### Part 4: Visualizing J(theta_0, theta_1)

print("Visualizing J(theta_0, theta_1) ... \n")

# Grid over which we will calculate J

theta0_vals = np.linspace(-10,10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's

J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

# Fill out J_vals

for i in np.arange(theta0_vals.size):
    for j in np.arange(theta1_vals.size) :
        t = [theta0_vals[i], theta1_vals[j]];
        J_vals[i,j] = computeCost(X,y,t)

# Because of the way meshgrids work in the surf command, we need to transpose
# J_vals before calling surf, or else the axes will be fillped

J_vals = J_vals.T        
     
# Surface plot

fig = plt.figure(2, figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

X,Y = np.meshgrid(theta0_vals, theta1_vals, indexing = 'xy')
Z = J_vals

ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.jet)
ax2.set_xlabel(r'$\theta_0$'); ax2.set_ylabel(r'$\theta_1$')
ax2.view_init(elev=15, azim=230)

# Contour plot

# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100

ax1.contour(X, Y, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(theta[0], theta[1], c='r', marker ='x')
ax1.set_xlabel(r'$\theta_0$'); ax1.set_ylabel(r'$\theta_1$')

plt.show()
