from __future__ import division
import time
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from scipy import stats

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn


### Part 1 : Feature Normalization

# Clear and close figures

plt.clf(); plt.close('all')

print('Loading data ... \n')

# Load Data

data = loadtxt('C:\\Python37\\Data\\ex1data2.txt', delimiter = ',')

X = data[:, 0:2] 
y = data[:, 2]
m = y.size

# Print some data points

print('First 10 examples from the dataset: \n')
list(map(lambda x,y: print("x = [%0.f %0.f], y=%0.f"%(x[0],x[1],y)),X[0:11,:],y[0:11]))

print('Program paused. Press enter to contine \n')
time.sleep(2)

# Scale features and set them to zero mean

print("Normalizing features ... \n")

X, mu, sigma = featureNormalize(X)  #X = stats.zscore(X) #(You could also use this instead of our featureNormalize function)

# Add intercept term to X

X = np.c_[np.ones((m,)), X]

### Part 2 : Gradient Descent

print("Running gradient Descent ... \n")

# Choose some alpha value

alpha = 0.01
num_iters = 400

# Init Theta and run Gradient Descent

theta = np.zeros((3,))

theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph

plt.figure(1)

plt.plot(np.arange(400), J_history[0:400], 'b', 2)
plt.xlabel("Number of iterations"); plt.ylabel("Cost J")
plt.show()

# Display gradient descent's result

print("Theta computed from gradient descent : \n")
print(theta)
print('\n')

### (Begin Optional) Selecting appropriate learning rates alpha

learning_rate = logspace(-2,0,9)
colors = list(mcd.CSS4_COLORS.values())  # try CSS4_COLORS OR TABLEAU_COLORS 
plt.figure(2)

for i, rate in  enumerate(learning_rate):
    theta = np.zeros((3,))
    theta,J = gradientDescentMulti(X,y,theta,rate,num_iters)
    plt.plot(np.arange(50),J[0:50],'-',c=colors[i], label ='alpha '+str(rate)) 
    plt.xlabel("Number of iterations"); plt.ylabel("Cost J")
    plt.legend(loc='best')

plt.show()

### (End optional)

# Estimate the price of a 1650 sq-ft, 3 br house
test = np.array([1650,3],float)
test_normalized = (test - mu)/sigma
test_normalized_augumented = np.append(1, test_normalized)
price = theta.dot(test_normalized_augumented)
print("Predicted price of 1650, sq-ft, 3 br house using gradient descent:\n", price)

print('Program paused. Press enter to contine \n')
time.sleep(2)

### Part 3 : Normal Equation

print("Solving with normal equation ...\n")

# Load Data

data = loadtxt("C:\\Python37\\Data\\ex1data2.txt", delimiter = ',')

X = data[:, 0:2]
y = data[:, 2]
m = y.size

# Add intercept term to X

X = np.c_[np.ones((m,)), X]

# Calculate the parameters from the normal equation

theta = normalEqn(X,y)

# Display normal equation's result

print("Theta computed from the normal equations: \n")
print(theta)
print("\n")

# Estimate the price of a 1650 sq-ft , 3 br house

price = np.array([1,1650,3],float).dot(theta)

print("Predicted price of a 1650 sq-ft , 3 br house using normal equations :\n", price)
