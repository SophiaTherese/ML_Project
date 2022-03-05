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
x = plt.figure(figsize=(15,5))
plt.title('Boxplots of attributes')
plt.grid(visible=True, axis='y')
plt.boxplot(Y, patch_artist=True, medianprops=dict(color='deeppink'), boxprops=dict(facecolor='white'))
plt.xticks(range(1,M+1), list(attributeNames), rotation=45)


# Next, we plot histograms of all attributes.

plt.figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    plt.subplot(u,v,i+1)
    plt.hist(X[:,i], bins=20, color='tab:pink')
    plt.xlabel(attributeNames[i], verticalalignment='center_baseline')
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: plt.yticks([])
    if i==0: plt.title('Distribution of Attributes')
    
