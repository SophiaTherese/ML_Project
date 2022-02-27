#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:52:43 2022

@author: Sophia
"""

from load_data import *

import matplotlib.pyplot as plt


plt.figure(figsize=(12,12))
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plt.plot(X[class_mask,m2], X[class_mask,m1], '.')
            if m1==M-1:
                plt.xlabel(attributeNames[m2])
            else:
                plt.xticks([])
            if m2==0:
                plt.ylabel(attributeNames[m1])
            else:
                plt.yticks([])
            #plt.ylim(0,X.max()*1.1)
            #plt.xlim(0,X.max()*1.1)
plt.legend(classNames)
plt.show()


plt.figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); 
v = np.ceil(float(M)/u)
for i in range(1,M):
    plt.subplot(u,v,i+1)
    plt.hist(X[:,i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: plt.yticks([])
    if i==0: title('Wine: Histogram')
