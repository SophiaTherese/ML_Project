#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:53:32 2022

@author: Sophia
"""
from load_data import *



from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend

from scipy.linalg import svd

import matplotlib.pyplot as plt



cols = range(1, 10) 
X_PCA = X[:, cols]
attributeNames_PCA = np.array(attributeNames[1:]);
N_PCA, M_PCA = X_PCA.shape


# Subtract mean value from data
Y = X_PCA - np.ones((N_PCA,1))*X_PCA.mean(0)
#To standardize, we dividing by the standard deviation
Y = Y*(1/np.std(Y,0))

# Subtract mean value from data
#Y = X_PCA - np.ones((N,1))*X_PCA.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.90

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#print('Ran PCA')


# exercise 2.1.4
# (requires data structures from ex. 2.2.1 and 2.2.3)
#from ex2_1_1 import *


# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('NanoNose data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()



pcs = [0,1,2,3]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b','y']
bw = .2
r = np.arange(1,M_PCA+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames_PCA)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()

# Inspecting the plot, we see that the 2nd principal component has large
# (in magnitude) coefficients for attributes A, E and H. We can confirm
# this by looking at it's numerical values directly, too:
print('PC1:')
print(V[:,0].T)

# How does this translate to the actual data and its projections?
# Looking at the data for water:

# Projection of water class onto the 2nd principal component.
#all_water_data = Y[y==4,:]

#print('First water observation')
#print(all_water_data[0,:])

# Based on the coefficients and the attribute values for the observation
# displayed, would you expect the projection onto PC2 to be positive or
# negative - why? Consider *both* the magnitude and sign of *both* the
# coefficient and the attribute!

# You can determine the projection by (remove comments):
#print('...and its projection onto PC2')
#print(all_water_data[0,:]@V[:,1])
# Try to explain why?
