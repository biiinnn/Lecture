# -*- coding: utf-8 -*-

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston

def ftest(X,y):
    # X: input variables
    # y: target
    n, p = X.shape
    
    ## add intercept
    b0 = np.ones(n).reshape(-1,1)
    X_new = np.concatenate((b0, X), axis=1)
    
    # calculate Beta hat
    Xt = X_new.T
    XtX = np.matmul(Xt, X_new)
    XtX_inv = np.linalg.inv(XtX)
    beta = np.matmul(np.matmul(XtX_inv, Xt), y)
    
    # calculate y predict value (y hat)
    y_pred = np.matmul(X_new, beta)
    
    # calculate SSR(MSR), SSE(MSE), SST
    SSR = sum((y_pred - np.mean(y))**2)
    SSE = sum((y - y_pred)**2)
    SST = SSR + SSE
    MSR = SSR / p
    MSE = SSE / (n-p-1)
    
    # calculate F-value, p-value
    F_value = MSR / MSE
    p_value = 1 - stats.f.cdf(F_value, p, n-p-1)
    
    # print result
    print("-" * 70)
    print("{:<10}  {:^10} {:^10} {:^10} {:>10} {:>10}".format("Factor", "SS", "DF", "MS", "F-value", "Pr > F"))
    print(" {:<10} {:^10.4f} {:^10} {:^10.4f} {:>10.4f} {:>10.4f}".format("Model", SSR, p, MSR, F_value, p_value)) 
    print(" {:<10} {:^10.4f} {:^10} {:^10.4f}".format("Error", SSE, n-p-1, MSE))
    print("-" * 70)
    print(" {:<10} {:^10.4f} {:^10}".format("Total", SST, n-1))
    print("-" * 70)
    
    return 0

def ttest(X,y,varname=None):
    # X: inpute variables
    # y: target
    n, p = X.shape
    
    # add intercept
    b0 = np.ones(n).reshape(-1,1)
    X_new = np.concatenate((b0, X), axis=1)
    
    # calculate Beta hat
    Xt = X_new.T
    XtX = np.matmul(Xt, X_new)
    XtX_inv = np.linalg.inv(XtX)
    beta = np.matmul(np.matmul(XtX_inv, Xt), y)
    
    # calculate y predict value (y hat)
    y_pred = np.matmul(X_new, beta)
    
    # calculate se(beta)
    SSE = sum((y - y_pred)**2)
    MSE = SSE / (n-p-1)
    se_beta = np.sqrt(MSE * np.diag(XtX_inv))
    
    # calculate t-value, p-value
    t_value = beta / se_beta
    p_value = (1 - stats.t.cdf(np.abs(t_value), n-p-1)) * 2
    
    # print result
    name = np.append("Const", varname)
    print("-" * 55)
    print("{:^10} {:>10} {:>10} {:>10}  {:^10}".format("Variable", "coef", "se", "t", "Pr>|t|"))
    for i in range(0,14):
        print("{:^10} {:>10.4f} {:>10.4f} {:>10.4f}  {:^10.4f}".format(name[i], beta[i], se_beta[i], t_value[i], p_value[i]))
    print("-" * 55)
    
    return 0

## Do not change!
# load data
data=load_boston()
X=data.data
y=data.target

ftest(X,y)
ttest(X,y,varname=data.feature_names)

