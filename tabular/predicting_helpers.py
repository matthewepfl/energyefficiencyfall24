import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor

# Set the seed
np.random.seed(42)

def linear_regression_train(X_train, y_train, ylog=False):
    """
    Linear regression using sklearn
    """
    lin_reg = LinearRegression()
    if ylog:
        epsilon=0.1
        y_train = np.log(y_train+epsilon)
    lin_reg.fit(X_train, y_train)
    # return coefficients
    return lin_reg

def linear_regression_predict(X_test, lin_reg, ylog=False, train=True, y_test=None):
    """
    Linear regression prediction using sklearn
    """
    y_pred = lin_reg.predict(X_test)
    if ylog:
        y_pred = np.exp(y_pred)

    # round to nearest integer
    y_pred = np.round(y_pred)

    # Remove negative values
    y_pred[y_pred <= 0] = 1e-30

    if train:
        # Print the method title and the poisson deviance value
        print("Linear regression yields poisson deviance: ", compute_poisson_deviance(y_test, y_pred))
    return y_pred

def neural_network_train(X_train, y_train, ylog=False, hidden_layer_sizes=(100, 100, 100), max_iter=1000):
    """
    Neural network using sklearn
    """
    if ylog:
        epsilon=1e-4
        y_train = np.log(y_train+epsilon)

    nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    nn.fit(X_train, y_train)
    return nn

def neural_network_predict(X_test, nn, ylog=False, train=True, y_test=None):
    """
    Neural network prediction using sklearn
    """
    y_pred = nn.predict(X_test)
    if ylog:
        y_pred = np.exp(y_pred)
    
    # Round to nearest integer
    y_pred = np.round(y_pred)

    # Remove negative values
    y_pred[y_pred <= 0] = 1e-10

    if train:
        # Print the method title and the poisson deviance value
        print("Neural network yields poisson deviance: ", compute_poisson_deviance(y_test, y_pred))
    return y_pred

def random_forest_train(X_train, y_train, ylog=False, n_estimators=100, max_depth=None, max_features=None, min_samples_split=2, random_state=1):
    """
    Random forest using sklearn
    """
    if ylog:
        epsilon=0.0001
        y_train = np.log(y_train+epsilon)
    rf = RandomForestRegressor(n_estimators=n_estimators, criterion='poisson', max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf

def random_forest_predict(X_test, rf, ylog=False, train=True, y_test=None):
    """
    Random forest prediction using sklearn
    """
    y_pred = rf.predict(X_test)
    if ylog:
        y_pred = np.exp(y_pred)
    
    # Round to nearest integer
    y_pred = np.round(y_pred)

    # Remove negative values
    y_pred[y_pred <= 0] = 1e-10

    if train:
        # Print the method title and the poisson deviance value
        print("Random forest yields poisson deviance: ", compute_poisson_deviance(y_test, y_pred))
    return y_pred

def ridge_train(X_train, y_train):
    """Ridge using sklearn. """

    ridge = Ridge()
    ridge_grid = GridSearchCV(ridge, param_grid={'alpha': [0.1, 1, 10, 100, 1000]}, cv=5)
    ridge_grid.fit(X_train, y_train)
    return ridge_grid


def ridge_predict(X_test, ridge_grid):
    """
    Ridge prediction using sklearn
    """
    y_pred_ridge = ridge_grid.predict(X_test)
    y_pred_ridge = np.round(y_pred_ridge)
    y_pred_ridge[y_pred_ridge <= 0] = 1e-10
    return y_pred_ridge

def hist_gradient_boosting_train(X_train, y_train, max_iter=100, max_leaf_nodes=31, learning_rate=0.1, 
                                max_depth=None, min_samples_leaf=20, l2_regularization=0 ,feature_mask=None, ylog=False):
    """
    Histogram gradient boosting using sklearn
    """
    if ylog:
        epsilon=0.0001
        y_train = np.log(y_train+epsilon)
    hgb = HistGradientBoostingRegressor(loss='poisson', categorical_features=feature_mask, max_iter=max_iter, min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization, learning_rate=learning_rate, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=1)
    print(X_train.shape, y_train.shape)
    hgb.fit(X_train, y_train)
    return hgb

def hist_gradient_boosting_predict(X_test, hgb, ylog=False, train=True, y_test=None, predict_price=False):
    """
    Histogram gradient boosting prediction using sklearn
    """
    y_pred = hgb.predict(X_test)
    if ylog:
        y_pred = np.exp(y_pred)

    # Remove negative values
    y_pred[y_pred <= 0] = 1e-10

    if train:
        if not predict_price:
            # Print the method title and the poisson deviance value
            print("Histogram gradient boosting yields poisson deviance: ", compute_poisson_deviance(y_test, y_pred))

    if predict_price:
        # Print the method title and the poisson deviance value
        print("Poisson deviance for predicting price: ", compute_poisson_deviance(y_test, y_pred))
        
    return y_pred

def compute_poisson_deviance(y, y_pred):
    """
    Compute the Poisson deviance using sklearn
    """
    deviance = mean_poisson_deviance(y, y_pred)
    return deviance

def lightgbm_train(X_train, y_train, ylog=False, n_estimators=100, num_leaves=31, max_depth=3, learning_rate=0.1, n_jobs=-1):
    """
    LightGBM using sklearn
    """
    if ylog:
        epsilon=0.0001
        y_train = np.log(y_train+epsilon)
    lgbm = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, num_leaves=num_leaves, n_jobs=n_jobs)
    lgbm.fit(X_train, y_train)
    return lgbm

def lightgbm_predict(X_test, lgbm, ylog=False, train=True, y_test=None):
    """
    LightGBM prediction using sklearn
    """
    y_pred = lgbm.predict(X_test)
    if ylog:
        y_pred = np.exp(y_pred)

    # Round to nearest integer
    y_pred = np.round(y_pred)

    # Remove negative values
    y_pred[y_pred <= 0] = 1e-10

    if train:
        # Print the method title and the poisson deviance value
        print("LightGBM yields poisson deviance: ", compute_poisson_deviance(y_test, y_pred))
    return y_pred

def pred_default(listings_df):
    listings_df_red = listings_df[(pd.DatetimeIndex(listings_df['Day of Advertisement Created']).year < 2017) & (pd.DatetimeIndex(listings_df['Day of Advertisement Created']).month < 12)]
    # Get all indices where year_created is 2016 and month_created is below 12
    indices_def = listings_df_red.index
    # Get the other indices
    indices_not_def = listings_df.index.difference(indices_def)
    # Create a new dataframe with the other indices
    listings_df = listings_df.loc[indices_not_def]
    # print number of rows with default values
    print("Number of rows with default 0 prediction: ", len(listings_df_red))
    return listings_df, indices_not_def

def split_zips(total_df, n=5):
    # Find the 20 zipcodes with the highest average Demand
    top_zips = total_df.groupby('Geo Zip')['Demand'].mean().sort_values(ascending=False).head(n).index
    # Create separate dataframe for the listings in the top 20 zipcodes
    top_df = total_df[total_df['Geo Zip'].isin(top_zips)]
    # Create separate dataframe for the listings not in the top 20 zipcodes
    not_top_df = total_df[~total_df['Geo Zip'].isin(top_zips)]
    return top_df, not_top_df

def split_cities(total_df):
    # Create separate dataframe for listings with Geo City Zürich or Schlieren
    high_df = total_df[total_df['Geo Canton'].isin(['ZH'])]
    # Get the indices of the listings in Zürich or Schlieren
    indices_high = high_df.index
    # Create separate dataframe for listings not in Zürich or Schlieren
    low_df = total_df[~total_df['Geo Canton'].isin(['ZH'])]
    # Get the indices of the listings not in Zürich or Schlieren
    indices_low = low_df.index
    return high_df, low_df, indices_high, indices_low