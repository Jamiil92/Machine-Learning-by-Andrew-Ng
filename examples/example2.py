from __future__ import division
import time
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from plot_learning_curve import plot_learning_curve

#Loading our data

# Download the data from the folder data in the repository and
# set your own directory    

plt.clf() ;  plt.close('all')
print("Loading our data ... \n")

data2 = np.loadtxt("C:\\Python37\\Data\\ex1data2.txt", delimiter = ",")

print('Program paused. Press enter to continue \n')
time.sleep(2)


print("Visualizing our data ...\n")

plt.figure(figsize =(14,6))
plt.subplot(121)
plt.scatter(data2[:,0],data2[:,2], s=80, c='b', marker = "x" )
plt.xlabel("Size of the house (in square feet)")
plt.ylabel("Price of the house")

plt.subplot(122)
plt.scatter(data2[:,1], data2[:,2], s=80, c='g', marker = "+" )
plt.xlabel("Number of bedroom")
plt.ylabel("Price of the house")


print("Starting our analysis with sklearn ....\n")

from sklearn.preprocessing import StandardScaler

feature_to_be_scaled = data2[:,0:2] # or simply we are removing the target column because they are the one we want to predict
scaler = StandardScaler()
print(scaler.fit(feature_to_be_scaled))
print("The mean per feature is : \n", scaler.mean_)

print("\nThe variance per feature is : \n", scaler.scale_)

print("\nThe scaled data is : \n", scaler.transform(feature_to_be_scaled)[0:10])

# Why do we need to normalize at all ?
# Simple reason the value are not of the same scale. So using features together the raw way,
# We might face a problem with the feature with heavier weight biasing our model .
# In a simple term, most of our prediction would be biased by the feature with big values
# More arguments if our data was skewed normalizing it better our result
# That's why we normalize
# Here is an histogram of our two features for some visual.

plt.figure(figsize =(14,6))

plt.subplot(121)
plt.hist(data2[:,0], bins = 10, density =True, facecolor='b', data =data2)
plt.xlabel("Size of the house (in square feet)")
plt.ylabel("Probability")

plt.subplot(122)
plt.hist(data2[:,1], bins = 5, density =True, facecolor='g', data =data2)
plt.xlabel("Number of bedroom")
plt.ylabel("Probability")

# From the plot you could notice that the size of the house feature is positively skewed whereas number of bedroom is not
# and the main reason we are really normalizing here is the different scale
# between both features.

print("\nExtracting the normalized features and targets ...\n")

X = scaler.transform(feature_to_be_scaled)
y = data2[:,2]

print("The shape of our features is :", X.shape)
print("\nThe shape of our target is : ", y.shape)

lineReg = LinearRegression(fit_intercept = True, normalize = False).fit(X, y)

print("\nWeight coefficients: ", lineReg.coef_)
print("y-axis intercept: ", lineReg.intercept_)

print("\nR^2 on data set : %f" %  lineReg.score(X, y))

# We got an R^2 of 0.732945 This is quite good but could be better

print("\nPredicting some values of our data ... \n")
y_pred = lineReg.predict(X)
print(y_pred[0:5])

test_normalized = (np.array([1650,3],float) - scaler.mean_ ) / scaler.scale_
test_normalized_augumented = np.append(1, test_normalized) # we add the ones
price = np.append(lineReg.intercept_, lineReg.coef_).dot(test_normalized_augumented)
print("\nPredicted price of 1650, sq-ft, 3 br house :", price)

plt.figure()
plot_learning_curve(LinearRegression(), X,y)
plt.show()


### From our learning curve we see that our data is suffering from overfitting
### What to do next ? We might need to generate more data to improve our model.
### In the end this is very detail way of how linearRegression works
### Let me explain what I meant


### Here is another simpler way of implementing this multivariate regression using sklearn
X_n = data2[:,0:2]
y_n = data2[:,2]

print("The shape of our features is :", X_n.shape)
print("\nThe shape of our target is : ", y_n.shape)

# setting LinearRegression that way do all the job for you :) :)
# It saves you time to do the normalization by yourself.

lineReg_n = LinearRegression(fit_intercept = True, normalize = False).fit(X_n, y_n)

print("\nTo make it visual look at the features again :\n", X_n[0:10])
print("\nWeight coefficients: ", lineReg_n.coef_)
print("y-axis intercept: ", lineReg_n.intercept_)

# Notice how different is our intercept
# This is to tell you again that the LinearRegression function is well cooked
# in choosing the right solver in the case here it has used Normal Equation
# Comparing to the previous analysis it has used Gradient Descent (check back again it intercept value)

print("\nR^2 on data set : %f" %  lineReg_n.score(X_n, y_n))

price_n = lineReg_n.predict([[1650, 3]])

print("\nPredicted price of 1650, sq-ft, 3 br house :", price_n)

#The End#
