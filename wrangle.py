import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import env
from sklearn.model_selection import train_test_split

#The following function will remove outliers from a litst of columns of a df and return the new dataframe without the outliers
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#The following function will be used in the prep functions to return the train, validate, and test splits
def train_validate_test_split(df, target, seed = 123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test


#The following function will take in the Zillow df and return a cleaned df
def wrangle_zillow():
    #First, acquire the data
    zillow_query = """
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261;
    """

    zillow_url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'

    zillow = pd.read_sql(zillow_query, zillow_url)

    #Now prepare the data
    #Begin by removing the null values
    zillow.dropna(inplace = True)

    #Now remove things that don't make sense and/or are impossible/illegal.
    #If something didn't sound like the average 'single family residential' property, I dropped it.
    zillow = zillow[(zillow.bedroomcnt > 0) & (zillow.bathroomcnt > 0)]
    zillow = zillow[zillow.bedroomcnt <= 5]
    zillow = zillow[zillow.bathroomcnt <= 3]
    zillow = zillow[zillow.calculatedfinishedsquarefeet <= 5000]
    zillow = zillow[zillow.calculatedfinishedsquarefeet >= (120 * zillow.bedroomcnt)]
    zillow = zillow[zillow.taxamount <= 20_000]

    #Now convert bedroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, fips, and yearbuilt to ints
    zillow.bedroomcnt = zillow.bedroomcnt.astype(int)
    zillow.calculatedfinishedsquarefeet = zillow.calculatedfinishedsquarefeet.astype(int)
    zillow.taxvaluedollarcnt = zillow.taxvaluedollarcnt.astype(int)
    zillow.fips = zillow.fips.astype(int)
    zillow.yearbuilt = zillow.yearbuilt.astype(int)

    #Rename calculatedfinishedsquarefeet and taxvaluedollarcnt
    zillow.rename(columns = {'calculatedfinishedsquarefeet':'home_area (sq-ft)', 'taxvaluedollarcnt':'home_value'}, inplace = True)

    #Now remove any outliers and return the prepared dataframe
    zillow = remove_outliers(zillow, 2.5, zillow.columns)

    return zillow
