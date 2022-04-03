#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:53:12 2022

@author: Sophia Wesche and Simone Engelbrcht
"""

from load_data import *

from scipy.linalg import svd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics as mt


from pandas import read_csv
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder

import imblearn
from imblearn.over_sampling import SMOTE



# ------------------------ Oversampling of dataset

#Original dataset is called X and y
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	#print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
#pyplot.bar(counter.keys(), counter.values())
#pyplot.show()



# transform (oversampling) the dataset (called X_OS and y_OS)
oversample = SMOTE()
X_OS, y_OS = oversample.fit_resample(X, y)
# summarize distribution
counter = Counter(y_OS)
for k,v in counter.items():
	per = v / len(y_OS) * 100
	#print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
#pyplot.bar(counter.keys(), counter.values())
#pyplot.show()



# ------------------------- KNN  (Skal cross valideres sammen med andre modelle)

# Maximum number of neighbors

# skal laves til array med forskellige værdier af K i stedet:
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


