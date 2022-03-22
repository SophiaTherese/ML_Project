# -*- coding: utf-8 -*-
from load_data import *

from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics as mt

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from toolbox_02450 import rlr_validate



# One-out-of-K

# Regularization

#Init: best error matrix (10,3), lambdas, h-values, ANN model
# lambdas = np.power(10.,np.arange(-3,2, step=0.1))
# h = [1, 2, 3, 4, 5]

# Outer-layer CV
# for split in outer layer CV
    
    # 10-fold linear reg / baseline:
    # for split in new 10-layer cv
        # lin reg w error
        # m = lm.LinearRegression().fit(X_train, y_train)
        # Error_test = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        # w = m.coef_

    # 10-fold lambda
    # her skal vi bruge opt_lambda_error
    # opt_lambda_err, opt_lambda, _, _, _ = rlr_validate(X, y, lambdas, cvf=10)


    # 10-fold ANN for each value of h
    # for range of hidden units h
        # for split in new 10-fold CV
            # ANN
    

    # Select best fold for each model type


# Create table