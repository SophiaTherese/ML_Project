#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:20:39 2022

@author: Sophia Wesche & Simone Engelbrecht
"""

from load_data import *

from scipy.linalg import svd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt




# trying to one-out-of-K code it - but not successfully 
glass_type = np.array(X[:, -1], dtype=int).T
K = glass_type.max() + 1
glass_type_encoding = np.zeros((glass_type.size, K))
glass_type_encoding[np.arange(glass_type.size), glass_type] = 1
#X = np.concatenate( (X[:, :-1], glass_type_encoding), axis=1) 



#Featuretransformation to make mean = 0 and std = 1
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)
#To standardize, we dividing by the standard deviation
Y = Y*(1/np.std(Y,0))
np.set_printoptions(precision=2, suppress=True)
mean = Y.mean(0).round(8)+0.0
std = np.std(Y,0)
print('Mean:', mean, '- std:', std)


## Crossvalidation
# Create crossvalidation partition for evaluation
#K = 10
#CV = model_selection.KFold(n_splits=K,shuffle=True)



cols = range(1, 9) 
X_reg = X[:, cols]
y_reg = X[:,0] # -1 takes the last column


# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X[:, 6],y_reg)
# Compute model output:
y_est = model.predict(X[:, 6])
# Or equivalently:
#y_est = model.intercept_ + X @ model.coef_
#w0_est = model.intercept_
#w1_est = model.model.coef_

# Plot original data and the model output
attr = 5


f = plt.figure()


plt.plot(X[:, 6],y_reg,'.')
#plot(X,y_true,'-')
plt.plot(X[:, 6],y_est,'-')
plt.xlabel(attributeNames[6]); plt.ylabel('y')
plt.legend(['Training data', 'Regression fit (model)'])

plt.show()

