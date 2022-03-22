#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:20:39 2022

@author: Sophia Wesche & Simone Engelbrecht
"""

from load_data import *

from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn import metrics as mt
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate


# one-out-of-K encode
glass_type = y.T
glass_type_encoding = np.zeros((glass_type.size, C+2))
glass_type_encoding[np.arange(glass_type.size), glass_type] = 1
glass_type_encoding = np.delete(glass_type_encoding, [0, 4], 1)
X_reg = np.concatenate( (X[:, :-1], glass_type_encoding), axis=1) 


#Featuretransformation to make mean = 0 and std = 1
# Subtract mean value from data
X_reg = X_reg - np.ones((X.shape[0],1))*X_reg.mean(0)
#To standardize, we dividing by the standard deviation
X_reg = X_reg*(1/np.std(X_reg,0))
#np.set_printoptions(precision=2, suppress=True)
#mean = Y.mean(0).round(8)+0.0
#std = np.std(Y,0)
#print('Mean:', mean, '- std:', std)

# Extract RI as new y
cols = range(1, 14)
y_reg = X_reg[:,0]
X_reg = X_reg[:,cols]


# ------------------------ REGULARIZATION PARAMETER -------------------------------

lambdas = np.power(10.,np.arange(-3,2, step=0.1))

# 10-fold cross validation
K = 10

opt_lambda_err, opt_lambda, weights, train_err, test_err = rlr_validate(X_reg, y_reg, lambdas, cvf=K)

print(weights.shape)

plt.plot(lambdas, train_err, '-o', label='Training error')
plt.plot(lambdas, test_err, '-o', label='Validation error')
plt.axvline(x=opt_lambda, ls='--', lw=2, color='y', label='Optimal lambda')
plt.xscale('log')
plt.xlabel('Regularization factor')
plt.ylabel('Mean squared error of cross-validation')
plt.legend(loc='upper left')
plt.title('Optimal lambda: ' + str(opt_lambda))
plt.show()











# CV = model_selection.KFold(K, shuffle=True)

# Error_train = np.empty((K,lambdas.size))
# Error_test = np.empty((K,lambdas.size))

# for i in range(lambdas.size):
#     k=0
#     for train_index, test_index in CV.split(X_reg,y_reg):
#         # extract training and test set for current CV fold
#         X_train = X_reg[train_index,:]
#         y_train = y_reg[train_index]
#         X_test = X_reg[test_index,:]
#         y_test = y_reg[test_index]
        
#         # Compute squared error
#         m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
#         Error_train[k, i] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
#         Error_test[k, i] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]


# opt_lambda = 10
# plt.plot(lambdas, np.ndarray.mean(Error_train,0), '-o', label='Training error')
# plt.plot(lambdas, np.ndarray.mean(Error_test,0), '-o', label='Validation error')
# plt.axvline(x=opt_lambda, ls='--', lw=2, color='y', label='Optimal lambda')
# plt.xscale('log')
# plt.xlabel('Regularization factor')
# plt.ylabel('Mean squared error of cross-validation')
# plt.legend(loc='upper left')
# plt.title('Optimal lambda: ' + str(opt_lambda))

        
    
    
    
    
    
    


















# #
# #y = X[:,0]
# #y = y.reshape(-1,1)
# #%%
# #---------------------- Normal linear regression using test 10 fold --------------------

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X_reg[:,cols],X_reg[:,0].squeeze(),test_size=test_proportion)

# # Fit ordinary least squares regression model
# model = lm.LinearRegression(fit_intercept=True)
# model = model.fit(X_train,y_train)
# # Compute model output:
# y_est = model.predict(X_test)
# # Or equivalently:
# #y_est = model.intercept_ + X @ model.coef_
# #w0_est = model.intercept_
# #w1_est = model.model.coef_

# # Plot original data and the model output

# r2 = mt.r2_score(y_test, y_est)
# print('r2')
# print(r2)
# #%%
# #f = plt.figure()


# #plt.plot(X_train,y_train,'.')
# #plot(X,y_true,'-')
# #plt.plot(X_reg,y_est,'-')
# #plt.xlabel(attributeNames[1]); plt.ylabel('y')
# #plt.legend(['Training data', 'Regression fit (model)'])

# #plt.show()

# #%%
# # ------------------------ REGULIZATION PARAMETER -------------------------------

# ## Crossvalidation
# # Create crossvalidation partition for evaluation
# M_reg = M-1
# K = 10
# CV = model_selection.KFold(K, shuffle=True)
# #CV = model_selection.KFold(K, shuffle=False)

# # Values of lambda
# lambdas = np.power(10.,range(-3,2))

# # Initialize variables
# #T = len(lambdas)
# Error_train = np.empty((K,1))
# Error_test = np.empty((K,1))
# Error_train_rlr = np.empty((K,1))
# Error_test_rlr = np.empty((K,1))
# Error_train_nofeatures = np.empty((K,1))
# Error_test_nofeatures = np.empty((K,1))
# w_rlr = np.empty((M_reg,K))
# mu = np.empty((K, M_reg-1))
# sigma = np.empty((K, M_reg-1))
# w_noreg = np.empty((M_reg,K))


# k=0
# for train_index, test_index in CV.split(X,y):
    
#     # extract training and test set for current CV fold
#     X_train = X_reg[train_index]
#     y_train = y_reg[train_index]
#     X_test = X_reg[test_index]
#     y_test = y[test_index]
#     internal_cross_validation = 10    
    
#     opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

#     # Standardize outer fold based on training set, and save the mean and standard
#     # deviations since they're part of the model (they would be needed for
#     # making new predictions) - for brevity we won't always store these in the scripts
#     mu[k, :] = np.mean(X_train[:, 1:], 0)
#     sigma[k, :] = np.std(X_train[:, 1:], 0)
    
#     X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
#     X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
#     Xty = X_train.T @ y_train
#     XtX = X_train.T @ X_train
    
#     # Compute mean squared error without using the input data at all
#     Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
#     Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

#     # Estimate weights for the optimal value of lambda, on entire training set
#     lambdaI = opt_lambda * np.eye(M_reg)
#     lambdaI[0,0] = 0 # Do no regularize the bias term
#     w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
#     # Compute mean squared error with regularization with optimal lambda
#     Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    # Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # # Estimate weights for unregularized linear regression, on entire training set
    # w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # # Compute mean squared error without regularization
    # Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    # Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # #m = lm.LinearRegression().fit(X_train, y_train)
    # #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

#     # Display the results for the last cross-validation fold
#     if k == K-1:
#         figure(k, figsize=(12,8))
#         subplot(1,2,1)
#         semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
#         xlabel('Regularization factor')
#         ylabel('Mean Coefficient Values')
#         grid()
#         # You can choose to display the legend, but it's omitted for a cleaner 
#         # plot, since there are many attributes
#         #legend(attributeNames[1:], loc='best')
        
#         subplot(1,2,2)
#         title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
#         loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
#         xlabel('Regularization factor')
#         ylabel('Squared error (crossvalidation)')
#         legend(['Train error','Validation error'])
#         grid()
    
#     # To inspect the used indices, use these print statements
#     #print('Cross validation fold {0}/{1}:'.format(k+1,K))
#     #print('Train indices: {0}'.format(train_index))
#     #print('Test indices: {0}\n'.format(test_index))

#     k+=1

# show()
# # Display results
# print('Linear regression without feature selection:')
# print('- Training error: {0}'.format(Error_train.mean()))
# print('- Test error:     {0}'.format(Error_test.mean()))
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
# print('Regularized linear regression:')
# print('- Training error: {0}'.format(Error_train_rlr.mean()))
# print('- Test error:     {0}'.format(Error_test_rlr.mean()))
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

# print('Weights in last fold:')
# for m in range(M_reg):
#     print('{:>15} {:>15}'.format(attributeNames[m+1], np.round(w_rlr[m,-1],2)))

# print('Ran Exercise 8.1.1')

