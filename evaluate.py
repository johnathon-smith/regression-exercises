import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

#plot_residuals(X,y,yhat)
def plot_residuals(X,y,yhat):
    #Calculate residuals
    residuals = y - yhat
    
    #plot residuals
    plt.scatter(X, residuals)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel('X')
    plt.ylabel('Residuals')
    plt.title('OLS model residuals')
    plt.show()

#regression errors: SSE, ESS, TSS, MSE, RMSE
def regression_errors(y, yhat):
    #Calculate SSE: sum of squared residuals
    residuals = y - yhat
    SSE = (residuals**2).sum()
    
    #Calculate ESS: difference between mean and yhat
    ESS = ( (yhat - y.mean()) ** 2).sum()
    
    #Calculate TSS: SSE + ESS
    TSS = SSE + ESS
    
    #Calculate MSE: SSE / observations
    MSE = SSE / len(y)
    
    #Calculate RMSE: sqrt of MSE
    RMSE = sqrt(MSE)
    
    return SSE, ESS, TSS, MSE, RMSE

#Baseline mean errors: SSE, MSE, RMSE
def baseline_mean_errors(y):
    #Calculate SSE: sum of squared residuals
    SSE = ( (y - y.mean()) ** 2).sum()
    
    #Calculate MSE: SSE / observations
    MSE = SSE / len(y)
    
    #Calculate RMSE: sqrt of MSE
    RMSE = sqrt(MSE)
    
    return SSE, MSE, RMSE

#Better than baseline
def better_than_baseline(y, yhat):
    #Get baseline mean errors
    baseline_SSE, baseline_MSE, baseline_RMSE = baseline_mean_errors(y)
    
    #Get model regression errors
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    
    #I will judge based on RMSE since that is most common method
    if RMSE < baseline_RMSE:
        return True
    else:
        return False