import matplotlib.pyplot as plt
import numpy as np
def plotData(x,y) :
    plt.plot(x,y, 'rx', 10) # Plot the data
    plt.xlabel('Profit in $10,000s'); # set the x-axis label
    plt.ylabel('Population of City in 10,000s'); #Set the y-axis label
    plt.xlim([4, 24])
    plt.ylim([-5, 25])
    plt.xticks(np.arange(4, 28, 4))
    plt.yticks(np.arange(-5, 30, 5))
