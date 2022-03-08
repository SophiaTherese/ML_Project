#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:20:39 2022

@author: Sophia
"""

from load_data import *

from scipy.linalg import svd
import matplotlib.pyplot as plt



# trying to one-out-of-K code it - but not successfully 
glass_type = np.array(X[:, -1], dtype=int).T
K = glass_type.max() + 1
glass_type_encoding = np.zeros((glass_type.size, K))
glass_type_encoding[np.arange(glass_type.size), glass_type] = 1
X = np.concatenate( (X[:, :-1], glass_type_encoding), axis=1) 



#Featuretransformation to make mean = 0 and std = 1
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)
#To standardize, we dividing by the standard deviation
Y = Y*(1/np.std(Y,0))
np.set_printoptions(precision=2, suppress=True)
mean = Y.mean(0).round(8)+0.0
std = np.std(Y,0)
print('Mean:', mean, '- std:', std)




