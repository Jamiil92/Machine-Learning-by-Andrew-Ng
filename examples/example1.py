from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from plot_learning_curve import plot_learning_curve

#Loading our data

# Download the data from the folder data in the repository and
# set your own directory    

data1 = np.loadtxt("C:\\Python37\\Data\\ex1data1.txt", delimiter = ",")


### Analysis with sklearn for data 1 ###

X = data1[:,0] # extracting the feature (only  one feature : profit)
y = data1[:,1] # extracting the target (only one target : Population city)

print(X.shape)
print(y.shape)

# Visualizing our dataset

plt.figure(1)
plt.plot(X,y, "rx")
plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of city in 10,000s")
plt.xlim([4 , 24])
plt.ylim([-5 , 25])
plt.xticks(np.arange(4,28,4))
plt.yticks(np.arange(-5,30,5))
plt.title("Visualizing our dataset")

# Adding the bias 
X = np.c_[np.ones(X.shape), X]

# Performing linear regression with sklearn

lineReg = LinearRegression()
lineReg.fit(X,y)

print("r^2 on data set:%f" % lineReg.score(X, y)) # we obtain a score of 0.702032 which is not quite bad

print("Weight coefficients:", lineReg.coef_)
print("y-axis intercept:" , lineReg.intercept_)

# Predicting the profit
y_pred = lineReg.predict(X)

# Below here we plot our dataset with prediction highlighted
# along the line of best fit

plt.figure(2)
plt.plot(X, y, 'rx', label = "data")
plt.xlim([4 , 24])
plt.ylim([-5 , 25])
plt.xticks(np.arange(4,28,4))
plt.yticks(np.arange(-5,30,5))

plt.plot(X, y_pred, 'gx', label = "prediction")
plt.xlim([4 , 24])
plt.ylim([-5 , 25])
plt.xticks(np.arange(4,28,4))
plt.yticks(np.arange(-5,30,5))
plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of city in 10,000s")

# Plugging min max values to plot the regression fit to our training data

max_point = X.max() * lineReg.coef_[1] + lineReg.intercept_
min_point = X.min() * lineReg.coef_[1] + lineReg.intercept_

plt.plot([X.min(), X.max()],[min_point, max_point], label = 'fit')
plt.title("Visualizing our data, prediction and line of best fit")

# From the plot we could see that our model is suffering from overfitting
# Well, we couldn't have expected a better result with r^2 score = 0.7
# Let's plot the learning curve to verify that the model is overfitting

plt.figure(3)
plot_learning_curve(LinearRegression(), X, y)

plt.show()

# What to do next ? we need more data
# to improve the accuracy of our model on the dataset
# This was just a quick review of linear regression with one variable 
# using sklearn
