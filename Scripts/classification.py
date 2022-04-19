# -*- coding: utf-8 -*-
from load_data import *

from imblearn.over_sampling import SMOTE
import scipy.stats as st
from sklearn import model_selection
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier

# Initialize: lambdas, cross-validation parameters, L-values
lambdas = np.power(10.,np.arange(-10,2, step=0.2))
K1 = 10
K2 = 10
# Maximum number of neighbors
L = range(1, 101)

# transform (oversampling) the dataset (called X_OS and y_OS)
oversample = SMOTE()
X_os, y_os = oversample.fit_resample(X, y)

#Featuretransformation to make mean = 0 and std = 1
# Subtract mean value from data
X_os = X_os - np.ones((X_os.shape[0],1))*X_os.mean(0)
# #To standardize, we dividing by the standard deviation
X_os = X_os*(1/np.std(X_os,0))

# RESULTS
baseline = np.zeros((K1,1))
log_reg = np.ndarray((K1,2))
knn = np.ndarray((K1,2))
knn.fill(np.finfo('float64').max)

# Outer-layer CV
CV_1 = model_selection.KFold(K1, shuffle=True)
k_1 = 0
alltime_best_error_log_reg = np.finfo('float64').max
alltime_best_log_reg = (0, np.ndarray((C,M)))
for train_index, test_index in CV_1.split(X_os,y_os):
    
    X_train_1 = X_os[train_index,:]
    y_train_1 = y_os[train_index]
    X_test_1 = X_os[test_index,:]
    y_test_1 = y_os[test_index]

    # Inner-layer CV
    CV_2 = model_selection.KFold(K2, shuffle=True)
    
    best_error_baseline = np.finfo('float64').max
    best_error_log_reg = np.finfo('float64').max
    best_error_knn = np.finfo('float64').max
    for train_index, test_index in CV_2.split(X_train_1, y_train_1):
        X_train_2 = X_train_1[train_index,:]
        y_train_2 = y_train_1[train_index]
        X_test_2 = X_train_1[test_index,:]
        y_test_2 = y_train_1[test_index]
        
        #%% Model fitting and prediction
        # Standardize data based on training set
        mu = np.mean(X_train_2, 0)
        sigma = np.std(X_train_2, 0)
        X_train_2 = (X_train_2 - mu) / sigma
        X_test_2 = (X_test_2 - mu) / sigma

        
        # Baseline
        mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                       tol=1e-4, random_state=1, 
                                       penalty='l2')
        mdl.fit(X_train_2,y_train_2)
        
        y_test_est_2 = mdl.predict(X_test_2)
        test_error_rate_2 = np.sum(y_test_est_2!=y_test_2) / len(y_test_2)
        
        # store best error rate
        if test_error_rate_2 < best_error_baseline:
            best_error_baseline = test_error_rate_2
            # save outer error
            y_test_est_1 = mdl.predict(X_test_1)
            test_error_rate_1 = np.sum(y_test_est_1!=y_test_1) / len(y_test_1)
            baseline[k_1] = test_error_rate_1
        

        # Logistic Regression
        for l in lambdas:
            mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                           tol=1e-4, random_state=1, 
                                           penalty='l2', C=1/l, max_iter=3000)
            mdl.fit(X_train_2,y_train_2)
            
            y_test_est_2 = mdl.predict(X_test_2)
            test_error_rate_2 = np.sum(y_test_est_2!=y_test_2) / len(y_test_2)
            
            # store best error rate along with lambda
            if test_error_rate_2 < best_error_log_reg:
                best_error_log_reg = test_error_rate_2
                
                # Determine error for outer test set
                y_test_est_1 = mdl.predict(X_test_1)
                test_error_rate_1 = np.sum(y_test_est_1!=y_test_1) / len(y_test_1)
                
                log_reg[k_1] = (l, test_error_rate_1)
                
                if test_error_rate_1 < alltime_best_error_log_reg:
                    alltime_best_error_log_reg = test_error_rate_1
                    alltime_best_log_reg = (l, mdl.coef_)

        # KNN
        for l in L:
            knclassifier = KNeighborsClassifier(n_neighbors=l)
            knclassifier.fit(X_train_2, y_train_2)
            
            y_test_est_2 = knclassifier.predict(X_test_2)
            test_error_rate_2 = np.sum(y_test_est_2!=y_test_2) / len(y_test_2)
            
            # store best error rate along with lambda
            if test_error_rate_2 < best_error_knn:
                best_error_knn = test_error_rate_2
                
                # Determine error for outer test set
                y_test_est_1 = knclassifier.predict(X_test_1)
                test_error_rate_1 = np.sum(y_test_est_1!=y_test_1) / len(y_test_1)
                
                knn[k_1] = (l, test_error_rate_1)

    k_1+=1

# Create table
print("Baseline:")
print(baseline)
print("Logistic Regression:")
print(log_reg)
print("KNN:")
print(knn)

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
print("Baseline / Logistic Regression")
setupI(baseline.squeeze(1), log_reg[:,1])
print("Baseline / KNN")
setupI(baseline.squeeze(1), knn[:,1])
print("KNN / Logistic Regression")
setupI(knn[:,1], log_reg[:,1])

print("Prediction of logistic regression model")
print("Value of lambda:", alltime_best_log_reg[0])
print("Weights:")
print(alltime_best_log_reg[1])
