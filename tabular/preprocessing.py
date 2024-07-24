import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from preprocessing_helper import *
from sklearn.model_selection import train_test_split
from predicting_helpers import hist_gradient_boosting_predict, hist_gradient_boosting_train
from helper import *

# Set the seed
np.random.seed(42)

def preprocess(listings_df, target="PropertyFE", split=True, test_size=0.2, drop_boolean=False, test_X=False, verbose=False):
    # apartments where either Price Net Normalized or Price Gross Normalized is 1, set them both to Nan
    listings_df.loc[listings_df['Price Net Normalized'] == 1, 'Price Gross Normalized'] = np.nan
    listings_df.loc[listings_df['Price Net Normalized'] == 1, 'Price Net Normalized'] = np.nan
    listings_df.loc[listings_df['Price Gross Normalized'] == 1, 'Price Gross Normalized'] = np.nan
    listings_df = listings_df.drop(['Price Net Normalized', 'Jahr'], axis=1)
    print("I'm updating")
    if test_X:
        # drop Prediction column
        listings_df = listings_df.drop(['Prediction'], axis=1)
    # Isolate demand variable if test_X is False
    if not test_X:
        y = listings_df.loc[:, target]
        # Drop demand variable from the dataframe and drop text columns
        print(target)
        listings_df = listings_df.drop([target], axis=1)

    if not test_X:
        listings_df.drop(
                ['Listing Title', 'Listing Description', 'Property Reference Id'], axis=1, inplace=True)

    # Make numerical variables
    listings_df, cantons, subcategories, floors, boolean_columns, zips = make_numerical(listings_df, drop_booleans=drop_boolean)
    if verbose:
        print("Values are now numerical")
    
    # Add crossterm is renovated * Years since renovated and drop the original renovation column
    listings_df['Is Renovated * Years Since Renovation'] = listings_df['Is Renovated'] * listings_df['Years Since Renovation']
    listings_df.drop(['Years Since Renovation'], axis=1, inplace=True)

    if verbose:
        print('Crossterm is renovated * Years since renovated is added')
    
    if split and not test_X:
        # Add a column with demand to the listing dataframe
        listings_df[target] = y.values
        return split_data(listings_df, test_size=test_size)

    if test_X:
        return listings_df
    return listings_df, y
