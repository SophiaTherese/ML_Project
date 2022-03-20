#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:24:50 2022

@author: Sophia
"""

from load_data import *

from scipy.linalg import svd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics as mt



# ------------------------ REGULIZATION PARAMETER -------------------------------

#change matrix so y = RI and X consists of chemical components
#cols = range(1, 9)
X_reg = X[:,cols]
y_reg = X[:,0]



