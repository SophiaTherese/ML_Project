#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:52:43 2022

@author: Sophia
"""

from load_data import *


from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show)
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore

#Jeg ved godt at du siger man ikke skal definere det sådan her, men gør det
#indtil vi får ryttet op. 

cols = range(1, 10) 
X = raw_data[:, cols]
attributeNames = np.array(['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
N, M = X.shape
# Subtract mean value from data
Xnorm = X - np.ones((N,1))*X.mean(0)
#To standardize, we dividing by the standard deviation
Xnorm = Xnorm*(1/np.std(Y,0))



Attributes = [0,1,2,3,4,5,6,7,8]
NumAtr = len(Attributes)

figure(figsize=(12,12))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
            if m1==NumAtr-1:
                xlabel(attributeNames[Attributes[m2]])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[Attributes[m1]])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)
show()


figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); 
v = np.ceil(float(M)/u)
for i in range(1,M):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('Wine: Histogram')
