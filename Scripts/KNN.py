#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 09:52:36 2022

@author: Sophia
"""

#from toolbox_02450 import jeffrey_interval
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

from load_data import *

# This script creates predictions from three KNN classifiers using cross-validation


# Maximum number of neighbors
L=[1, 20,200]

CV = model_selection.LeaveOneOut()
i=0

# store predictions.
yhat = []
y_true = []
for train_index, test_index in CV.split(X, y):
    #print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    dy = []
    for l in L:
        knclassifier = KNeighborsClassifier(n_neighbors=l)
        knclassifier.fit(X_train, y_train)
        y_est = knclassifier.predict(X_test)

        dy.append( y_est )
        # errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    i+=1

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)
yhat[:,0] # predictions made by first classifier.
# Compute accuracy here.

for l in range(len(L)):
    #accuracy of each mpdel is calculated: 
    print("M_{} =".format(l+1), sum(yhat[:,l] == y_true)/len(y_true))

# M_1 = sum(yhat[:,0] == y_true)/len(y_true)
# M_2 = sum(yhat[:,1] == y_true)/len(y_true)
# M_3 = sum(yhat[:,2] == y_true)/len(y_true)

# print('M_1 =', M_1)
# print('M_2 =', M_2)
# print('M_3 =', M_3)



#Jeffrey inteval for K = 1 in KNN. 
#alpha = 0.05
#[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)

#print("Theta point estimate", thetahatA, " CI: ", CIA)