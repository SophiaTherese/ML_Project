#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:53:32 2022

@author: Sophia Wesche s173828, Simone Engelbrecht s174276, Aidana Nursultanova s212994
"""
from load_data import *

from scipy.linalg import svd
import matplotlib.pyplot as plt



# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)
#To standardize, we dividing by the standard deviation
Y = Y*(1/np.std(Y,0))

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.90

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-', color='tab:pink')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-', color='tab:green')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
plt.figure()
plt.set_cmap('Greens')
plt.rcParams['image.cmap']='jet'

plt.title('Glass data projected onto first two principal components')

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C9']
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==classNames[c]
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', color=colors[c], alpha=.5)
plt.legend(classNames, title="Glass type")
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()


# Plot PCA component coefficients
f = plt.figure(figsize=(15, 5))
pcs = [0,1,2,3,4]
legendStrs = ['PC'+str(e+1) for e in pcs]

bw = .2
r = np.arange(0,M)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=0.2)
plt.xticks(r-0.1, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()
