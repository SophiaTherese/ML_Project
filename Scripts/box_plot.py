#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:34:08 2022

@author: Sophia
"""

from load_data import *


from matplotlib.pyplot import boxplot, title, figure, subplot, plot, legend, show,  xlabel, ylabel, xticks, yticks
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt



#Jeg ved godt at du siger man ikke skal definere det sådan her, men gør det
#indtil vi får ryttet op. 
cols = range(1, 10) 
X = raw_data[:, cols]
attributeNames = np.array(['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
N, M = X.shape


#KUNNE IKKE FÅ DERES NORMALISERING TIL AT VIRKE SÅ BRUGTE DEN FRA PCA:
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)
#To standardize, we dividing by the standard deviation
Y = Y*(1/np.std(Y,0))


# We start with a box plot of each attribute
plt.figure(figsize=(15,5))
title('Boxplots of attributes')
boxplot(Y)
plt.xticks(range(1,M+1), list(attributeNames), rotation=45)






# From this it is clear that there are some outliers in the Alcohol
# attribute (10x10^14 is clearly not a proper value for alcohol content)
# However, it is impossible to see the distribution of the data, because
# the axis is dominated by these extreme outliers. To avoid this, we plot a
# box plot of standardized data (using the zscore function).

#plt.figure(figsize=(12,6))
#title('Glass: Boxplot (standarized)')
#boxplot(zscore(X, ddof=1), attributeNames)
#xticks(range(1,M+1), attributeNames, rotation=45)



# Next, we plot histograms of all attributes.

figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M-1):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('Wine: Histogram')
    

# This confirms our belief about outliers in attributes 2, 8, and 11.
# To take a closer look at this, we next plot histograms of the 
# attributes we suspect contains outliers

h2 = plt.figure(figsize=(14,9))
m = [1, 7, 10]
for i in range(len(m)):
    subplot(1,len(m),i+1)
    hist(X[:,m[i]],50)
    xlabel(attributeNames[m[i]])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i==0: title('Wine: Histogram (selected attributes)')


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
figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X_PCA[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('Wine: Histogram (after outlier detection)')

# This reveals no further outliers, and we conclude that all outliers have
# been detected and removed.

show()

print('Ran Exercise 4.3.1')