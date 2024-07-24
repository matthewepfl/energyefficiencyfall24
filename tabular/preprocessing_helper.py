import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

from predicting_helpers import hist_gradient_boosting_predict, hist_gradient_boosting_train
from helper import *

pd.options.mode.chained_assignment = None

# Set the seed
np.random.seed(42)

def make_numerical(df, drop_booleans):
    # Split date created
    df['year_created'] = pd.DatetimeIndex(
        df.loc[:, 'Day of Advertisement Created']).year
    df['month_created'] = pd.DatetimeIndex(
        df.loc[:, 'Day of Advertisement Created']).month
    df['day_created'] = pd.DatetimeIndex(
        df.loc[:, 'Day of Advertisement Created']).day

    df['month_available'] = pd.DatetimeIndex(
        df.loc[:, 'Day of Date Available From']).month
    df['day_available'] = pd.DatetimeIndex(
        df.loc[:, 'Day of Date Available From']).day

    # Split date  available into year, month and day. Calculate the number of days between the two dates and add it to the dataframe
    df['Days between'] = (pd.to_datetime(df['Day of Date Available From']) - pd.to_datetime( df['Day of Advertisement Created'])).dt.days

    # Create binary variable indicating whether a property is renovated or not. A property is considered renovated if the column year last renovated is not nan, 
    # and the year last renovated is not equal to Year Built
    df['Is Renovated'] = df.apply(lambda x: 1 if (not np.isnan(x['Year Lastrenovated']) and (x['Year Lastrenovated'] != x['Year Built'])) else 0, axis=1)

    # Measure number of years between the renovation and the date the advertisement was created, if the property is renovated
    df['Years Since Renovation'] = df.apply(lambda x: x['year_created'] - x['Year Lastrenovated'] if x['Is Renovated'] == 1 else 0, axis=1)

    # Make binary variables for each value in the floor column
    floors=pd.get_dummies(df['Floor'], prefix='Floor')
    df = pd.concat([df, floors], axis=1)

    # Get column names of floors
    floor_columns = floors.columns

    # Make binary variables for each value of canton and add them to the dataframe
    cantons=pd.get_dummies(df['Geo Canton'], prefix='Canton')
    df = pd.concat([df, cantons], axis=1)

    cities = pd.get_dummies(df['Geo City'], prefix='City')
    df = pd.concat([df, cities], axis=1)

    # Get column names of cantons
    canton_columns = cantons.columns
    
    zips = pd.get_dummies(df['Geo Zip'], prefix='Zip')
    df = pd.concat([df, zips], axis=1)
 
    # Get column names of zip codes
    zip_columns = zips.columns

    # Make binary variables for each string value of Subcategory Idx
    subcategories = pd.get_dummies(df['Subcategory En Idx'], prefix='Subcategory')
    df = pd.concat(
        [df, subcategories], axis=1)

    # Get column names of subcategories
    subcategory_columns = subcategories.columns
    
    # Drop Geo Canton, Category Idx, Floor Subcategory Idx, original date columns and zip code
    df.drop(['Geo Canton', 'Floor', 'Geo City', 'Year Lastrenovated', 'Category Idx',
            'Subcategory En Idx', 'Day of Advertisement Created', 'Day of Date Available From', 'Geo Zip'], axis=1, inplace=True)

    # Make binary variables for each boolean column
    boolean_columns = ['Has Balcony', 'Has Cabletv', 'Has Elevator', 'Has Fireplace', 'Has Garage', 'Has Parking',
                    'Is New Construction', 'Is New Construction Potential', 'Is Tenant2Tenant', 'Is Wheelchairaccessible', 'Are Pets Allowed']
    if drop_booleans:
        drop_columns = ['Has Cabletv', 'Has Fireplace', 'Is New Construction', 'Is Tenant2Tenant', 'Is Wheelchairaccessible']
        boolean_columns = [x for x in boolean_columns if x not in drop_columns]
        df.drop(drop_columns, axis=1, inplace=True)

    for column in boolean_columns:
        # Make binary values for the boolean column, keep NaN values
        df[column] = df[column].map({True: 1, False: 0})

    return df, canton_columns, subcategory_columns, floor_columns, boolean_columns, zips

def not_in_train(test_data, train_data):
    # Find zip codes in test data that are not in training data
    zip_codes_train = train_data['Geo Zip'].unique()
    zip_codes_test = test_data['Geo Zip'].unique()
    zip_codes_train_sorted = np.sort(zip_codes_train)
    zip_codes_test_sorted = np.sort(zip_codes_test)
    zip_codes_not_in_train = np.sort(np.setdiff1d(zip_codes_test_sorted, zip_codes_train_sorted))
    # For each zip code that is not in the training data, change the zip code to the closest zip code that is in the training data
    for zip_code in zip_codes_not_in_train:
        test_data.loc[test_data['Geo Zip'] == zip_code, 'Geo Zip'] = zip_codes_train_sorted[np.abs(zip_codes_train_sorted - zip_code).argmin()]
    return test_data

def to_numeric(arr):
    result = np.empty(arr.shape, dtype=float)
    for i, val in enumerate(arr):
        try:
            result[i] = float(val)
        except ValueError:
            result[i] = np.nan
    return result



def not_in_test(test_data, train_data, test_data_prepr, train_data_prepr):
    # Find zip codes in test data that are not in training data
    zip_codes_train = train_data['Geo Zip'].unique()
    zip_codes_test = test_data['Geo Zip'].unique()
    zip_codes_not_in_test = np.setdiff1d(zip_codes_train, zip_codes_test)

    # Create a DataFrame with dummy columns for zip codes not in test data
    zip_code_dummies = pd.DataFrame(0, columns=['Zip_' + str(zip_code) for zip_code in zip_codes_not_in_test], index=test_data_prepr.index)

    # Concatenate the dummy columns to the test data
    test_data_prepr = pd.concat([test_data_prepr, zip_code_dummies], axis=1)

    # Find Floor values in training data that are not in test data
    floors_train = train_data['Floor'].unique()
    floors_test = test_data['Floor'].unique()
    floors_not_in_test = np.setdiff1d(floors_train, floors_test)
    floors_not_in_test = floors_not_in_test[~np.isnan(floors_not_in_test)]

    # Create a DataFrame with dummy columns for floors not in test data
    floor_dummies = pd.DataFrame(0, columns=['Floor_' + str(floor) for floor in floors_not_in_test], index=test_data_prepr.index)

    # Concatenate the dummy columns to the test data
    test_data_prepr = pd.concat([test_data_prepr, floor_dummies], axis=1)

    # Find subcategory values in training data that are not in test data
    subcategories_train = train_data['Subcategory En Idx'].unique()
    subcategories_test = test_data['Subcategory En Idx'].unique()
    subcategories_not_in_test = np.setdiff1d(subcategories_train, subcategories_test)

    #subcategories_not_in_test = to_numeric(subcategories_not_in_test)


    subcategories_not_in_test = subcategories_not_in_test[~pd.isna(subcategories_not_in_test)]

    # Create a DataFrame with dummy columns for subcategories not in test data
    subcategory_dummies = pd.DataFrame(0, columns=['Subcategory_' + str(subcategory) for subcategory in subcategories_not_in_test], index=test_data_prepr.index)

    # Concatenate the dummy columns to the test data
    test_data_prepr = pd.concat([test_data_prepr, subcategory_dummies], axis=1)

    # Find city values in training data that are not in test data
    cities_train = train_data['Geo City'].unique()
    cities_test = test_data['Geo City'].unique()
    cities_not_in_test = np.setdiff1d(cities_train, cities_test)
    #cities_not_in_test = to_numeric(cities_not_in_test)
    cities_not_in_test = cities_not_in_test[~pd.isna(cities_not_in_test)]

    # Create a DataFrame with dummy columns for cities not in test data
    city_dummies = pd.DataFrame(0, columns=['City_' + str(city) for city in cities_not_in_test], index=test_data_prepr.index)

    # Concatenate the dummy columns to the test data
    test_data_prepr = pd.concat([test_data_prepr, city_dummies], axis=1)

    # Find Canton values in training data that are not in test data
    cantons_train = train_data['Geo Canton'].unique()
    cantons_test = test_data['Geo Canton'].unique()
    cantons_not_in_test = np.setdiff1d(cantons_train, cantons_test)
    #cantons_not_in_test = to_numeric(cantons_not_in_test)

    cantons_not_in_test = cantons_not_in_test[~pd.isna(cantons_not_in_test)]

    # Create a DataFrame with dummy columns for cantons not in test data
    canton_dummies = pd.DataFrame(0, columns=['Canton_' + str(canton) for canton in cantons_not_in_test], index=test_data_prepr.index)

    # Concatenate the dummy columns to the test data
    test_data_prepr = pd.concat([test_data_prepr, canton_dummies], axis=1)
    test_data_prepr.to_csv('test_data_prepr.csv', index=False)
    # Align the ordering of the columns in the test data with the training data
    test_data_prepr = test_data_prepr[train_data_prepr.columns]
    
    return test_data_prepr




    
