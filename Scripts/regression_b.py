# -*- coding: utf-8 -*-
from load_data import *

import scipy.stats as st
from sklearn import model_selection
import sklearn.linear_model as lm
import torch

from toolbox_02450 import rlr_validate, train_neural_net


# Initialize: lambdas, cross-validation parameters, h-values
lambdas = np.power(10.,np.arange(-3,2, step=0.1))
K1 = 10
K2 = 3
# Parameters for neural network classifier
h = range(1, 11)

# RESULTS
ann = np.ndarray((K1,2))
ann.fill(np.finfo('float64').max)
lin_reg = np.ndarray((K1,2))
baseline = np.zeros((K1,1))

# Outer-layer CV
CV_1 = model_selection.KFold(K1, shuffle=True)
k_1 = 0
for train_index, test_index in CV_1.split(X_reg,y_reg):
    
    X_train_1 = X_reg[train_index,:]
    y_train_1 = y_reg[train_index]
    X_test_1 = X_reg[test_index,:]
    y_test_1 = y_reg[test_index]
    
    # 10-fold linear reg / baseline:
    CV_2 = model_selection.KFold(K2, shuffle=True)
    best_error_baseline = np.finfo('float64').max
    for train_index, test_index in CV_2.split(X_train_1, y_train_1):
        X_train_2 = X_train_1[train_index,:]
        y_train_2 = y_train_1[train_index]
        X_test_2 = X_train_1[test_index,:]
        y_test_2 = y_train_1[test_index]
        # lin reg w error
        m = lm.LinearRegression().fit(X_train_2, y_train_2)
        # select outer error based on best inner error    
        test_error = np.square(y_test_2-m.predict(X_test_2)).sum()/y_test_2.shape[0]
        if test_error < best_error_baseline:
            best_error_baseline = test_error
            # save outer error
            baseline[k_1] = np.square(y_test_1-m.predict(X_test_1)).sum()/y_test_1.shape[0]
    
    # 10-fold lambda
    _, opt_lambda, _, _, _ = rlr_validate(X_train_1, y_train_1, lambdas, cvf=10)
    
    # Estimate weights for the optimal value of lambda, on entire training set
    Xty = X_train_1.T @ y_train_1
    XtX = X_train_1.T @ X_train_1
    lambdaI = opt_lambda * np.eye(M_reg)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    # Compute mean squared error with regularization with optimal lambda
    test_error = np.square(y_test_1-X_test_1 @ w).sum(axis=0)/y_test_1.shape[0]
    lin_reg[k_1] = (opt_lambda,test_error)

    # 10-fold ANN for each value of h
    best_error_ann = np.finfo('float64').max
    for n_hidden_units in h:
        n_replicates = 2        # number of networks trained in each k-fold
        max_iter = 8000         # stop criterion 2 (max epochs in training)
        
        # Define the model, see also Exercise 8.2.2-script for more information.
        # Define the model
        model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M_reg, n_hidden_units), #M features to n_hidden_units
                            torch.nn.Tanh(),   # 1st transfer function,
                            torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                            # no final tranfer function, i.e. "linear output"
                            )
        loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
        
        # ANN in 10-fold CV
        for train_index, test_index in CV_2.split(X_train_1,y_train_1):
            X_train_2 = torch.Tensor(X_train_1[train_index,:])
            y_train_2 = torch.Tensor(y_train_1[train_index]).unsqueeze(1)
            X_test_2 = torch.Tensor(X_train_1[test_index,:])
            y_test_2 = torch.Tensor(y_train_1[test_index]).unsqueeze(1)
            
            # ANN
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_2,
                                                               y=y_train_2,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
                        
                
            # Determine estimated class labels for test set
            y_test_est_2 = net(X_test_2)
            
            # Determine mean square error
            se_2 = (y_test_est_2.float()-y_test_2.float())**2 # squared error
            mse_2 = (sum(se_2).type(torch.float)/len(y_test_2)).data.numpy() #mean
            
            # store best error rate along with h
            if mse_2 < best_error_ann:
                best_error_ann = mse_2
                
                y_test = torch.Tensor(y_test_1).unsqueeze(1)
                # Determine error for outer test set
                y_test_est_1 = net(torch.Tensor(X_test_1))
                se_1 = (y_test_est_1.float()-y_test.float())**2 # squared error
                mse_1 = (sum(se_1).type(torch.float)/len(y_test)).data.numpy() #mean
                ann[k_1] = (n_hidden_units, mse_1)

    k_1+=1

# Create table
print("Baseline:")
print(baseline)
print("Linear Regression:")
print(lin_reg)
print("ANN:")
print(ann)

def setupI(zA, zB):
    # Initialize parameters and run test appropriate for setup I
    alpha = 0.05

    z = zA - zB
    p_setupI = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
    CI_setupI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    print ("p-value:")
    print(p_setupI)
    print ("confidence interval")
    print(CI_setupI)

print("Setup I")
print("Baseline / Linear Regression")
setupI(baseline.squeeze(1), lin_reg[:,1])
print("Baseline / ANN")
setupI(baseline.squeeze(1), ann[:,1])
print("ANN / Linear Regression")
setupI(ann[:,1], lin_reg[:,1])





