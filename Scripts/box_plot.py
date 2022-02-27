#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:34:08 2022

@author: Sophia
"""

from load_data import *

import matplotlib.pyplot as plt

# Normalise data to see distribution and identify outliers
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)
#To standardize, we dividing by the standard deviation
Y = Y*(1/np.std(Y,0))


# We start with a box plot of each attribute
plt.figure(figsize=(15,5))
plt.title('Boxplots of attributes')
plt.boxplot(Y)
plt.xticks(range(1,M+1), list(attributeNames), rotation=45)


# Next, we plot histograms of all attributes.

plt.figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    plt.subplot(u,v,i+1)
    plt.hist(X[:,i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: plt.yticks([])
    if i==0: plt.title('Wine: Histogram')
    

# This confirms our belief about outliers in attributes Na and K.
# To take a closer look at this, we next plot histograms of the 
# attributes we suspect contains outliers

h2 = plt.figure(figsize=(14,9))
m = [1, 5]
for i in range(len(m)):
    plt.subplot(1,len(m),i+1)
    plt.hist(X[:,m[i]],50)
    plt.xlabel(attributeNames[m[i]])
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i>0: plt.yticks([])
    if i==0: plt.title('Wine: Histogram (selected attributes)')

# TODO: analyse af outliers baseret på domænet, fjerne outliers
# The histograms show that there are a few very extreme values in these
# three attributes. To identify these values as outliers, we must use our
# knowledge about the data set and the attributes. Say we expect volatide
# acidity to be around 0-2 g/dm^3, density to be close to 1 g/cm^3, and
# alcohol percentage to be somewhere between 5-20 % vol. Then we can safely
# identify the following outliers, which are a factor of 10 greater than
# the largest we expect.
#outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
#valid_mask = np.logical_not(outlier_mask)

# Finally we will remove these from the data set
#X = X[valid_mask,:]
#y = y[valid_mask]
#N = len(y)


# Now, we can repeat the process to see if there are any more outliers
# present in the data. We take a look at a histogram of all attributes:
plt.figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    plt.subplot(u,v,i+1)
    plt.hist(X[:,i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: plt.yticks([])
    if i==0: plt.title('Wine: Histogram (after outlier detection)')

# This reveals no further outliers, and we conclude that all outliers have
# been detected and removed.

plt.show()

print('Ran Exercise 4.3.1')