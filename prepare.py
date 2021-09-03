import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

#The following function will plot the original and scaled distributions of the zillow data
def compare_dists(train, cols_to_scale, cols_scaled):
    plt.figure(figsize=(13,6))

    for i, col in enumerate(cols_to_scale):
        i += 1
        plt.subplot(2,4,i)
        train[col].plot.hist()
        plt.title(col)

    for i, col in enumerate(cols_scaled):
        i += 5
        plt.subplot(2,4,i)
        train[col].plot.hist()
        plt.title(col)

    plt.tight_layout()
    plt.show()

#The following function will take in train, validate, and test data sets and return the scaled version of each
def scale_zillow_data(train, validate, test):
    #Create lists to hold column names
    cols_to_scale = ['bedroom_count', 'bathroom_count', 'area (sq-ft)', 'tax_amount']
    cols_scaled = ['bedroom_count_scaled', 'bathroom_count_scaled', 'area (sq-ft) scaled', 'tax_amount_scaled']

    #Instantiate the scaler
    robust_scaler = RobustScaler()

    #Fit the scaler on train
    robust_scaler.fit(train[cols_to_scale])

    #Transform the data
    train[cols_scaled] = robust_scaler.transform(train[cols_to_scale])
    validate[cols_scaled] = robust_scaler.transform(validate[cols_to_scale])
    test[cols_scaled] = robust_scaler.transform(test[cols_to_scale])

    #Visualize and compare the scaled and non-scaled distributions
    compare_dists(train, cols_to_scale, cols_scaled)

    #Drop the non-scaled columns
    train.drop(columns = cols_to_scale, inplace = True)
    validate.drop(columns = cols_to_scale, inplace = True)
    test.drop(columns = cols_to_scale, inplace = True)

    return train, validate, test