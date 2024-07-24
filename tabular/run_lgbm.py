import os
import pandas as pd
import numpy as np
import pickle
from pyaxis import pyaxis
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, train_test_split
from lightgbm import LGBMRegressor
import multiprocessing

# Import custom helper functions
from preprocessing_helper import *
from preprocessing import *
from predicting_helpers import *

# Set constants
DATA_PATH = "/scratch/izar/mmorvan/EnergyEfficiencyPrediction/data/"
SAVE_PATH = "/scratch/izar/mmorvan/EnergyEfficiencyPrediction/tabular/"
TARGET = "PropertyFE"
N_SPLITS = 5
TEST_SIZE = 0.15
NUM_CPUS = 20

# LGBM hyperparameters (you can add as many as you want)
learning_rate_opts = [0.01]
max_leaf_nodes_opts = [20]
max_iter_opts = [100]
max_depth_opts = [None]

def preprocess_tab(train_datatab, test_datatab, target="PropertyFE", max_iter=100, max_leaf_nodes=31, learning_rate=0.1,
                      max_depth=None, drop_correlated=False, number_of_apartments_df=None,
                      use_num_appt=False, drop_boolean=False, verbose=True):
    if verbose:
        print('Starting preprocessing of tabular data......')

    # Ensure no overlapping Property Reference IDs between train and test sets
    train_property_ids = set(train_datatab['Property Reference Id'])
    test_property_ids = set(test_datatab['Property Reference Id'])
    y_test = test_datatab[target]

    overlapping_ids = train_property_ids.intersection(test_property_ids)

    if overlapping_ids:
        if verbose:
            print(f'Removing {len(overlapping_ids)} overlapping Property Reference Ids from the test set.')
        test_datatab = test_datatab[~test_datatab['Property Reference Id'].isin(overlapping_ids)]

    if drop_correlated:
        train_datatab = train_datatab.drop(['Number Of Rooms Cleaned', 'Livingspace'], axis=1)
        test_datatab = test_datatab.drop(['Number Of Rooms Cleaned', 'Livingspace'], axis=1)

    if use_num_appt:
        total_train_datatab = merge_numbappt(train_datatab, number_of_apartments_df)
        total_test_datatab = merge_numbappt(test_datatab, number_of_apartments_df)
        total_train_datatab['Number of apartments'] = total_train_datatab['Number of apartments'].astype(float)
        total_test_datatab['Number of apartments'] = total_test_datatab['Number of apartments'].astype(float)
    else:
        total_train_datatab = train_datatab.copy()
        total_test_datatab = test_datatab.copy()

    total_train_datatab.rename(columns={'Leerwohnungsziffer': 'Vacancy rate provided'}, inplace=True)
    total_test_datatab.rename(columns={'Leerwohnungsziffer': 'Vacancy rate provided'}, inplace=True)
    total_train_datatab['Vacancy rate provided'] = total_train_datatab['Vacancy rate provided'].astype(float)
    total_test_datatab['Vacancy rate provided'] = total_test_datatab['Vacancy rate provided'].astype(float)
    total_test_datatab = not_in_train(total_test_datatab, total_train_datatab)

    train_datatab_prepr, y_train = preprocessed(total_train_datatab, target, drop_boolean=drop_boolean, split=False)
    test_datatab_prepr = preprocessed(total_test_datatab, target, drop_boolean=drop_boolean, split=False, test_X=True)
    test_datatab_prepr = not_in_test(total_test_datatab, total_train_datatab, test_datatab_prepr, train_datatab_prepr)

    if target == "PropertyFE" and 'Demand' in train_datatab_prepr.columns:
        train_datatab_prepr = train_datatab_prepr.drop(['Demand'], axis=1)
        test_datatab_prepr = test_datatab_prepr.drop(['Demand'], axis=1)

    if verbose:
        print('Finished preprocessing of tabular data, ready to start predictions......')

    return train_datatab_prepr, y_train, test_datatab_prepr, y_test

def preprocessed(listings_df, target="PropertyFE", split=True, test_size=0.2, drop_boolean=False, test_X=False, verbose=False):
    listings_df.loc[listings_df['Price Net Normalized'] == 1, 'Price Gross Normalized'] = np.nan
    listings_df.loc[listings_df['Price Net Normalized'] == 1, 'Price Net Normalized'] = np.nan
    listings_df.loc[listings_df['Price Gross Normalized'] == 1, 'Price Gross Normalized'] = np.nan
    listings_df = listings_df.drop(['Jahr'], axis=1)

    if test_X & ('Prediction' in listings_df.columns):
        listings_df = listings_df.drop(['Prediction'], axis=1)
    if not test_X:
        y = listings_df.loc[:, target]
        listings_df = listings_df.drop([target], axis=1)

    if not test_X:
        listings_df.drop(['Listing Title', 'Listing Description', 'Property Reference Id'], axis=1, inplace=True)

    listings_df, cantons, subcategories, floors, boolean_columns, zips = make_numerical(listings_df, drop_booleans=drop_boolean)

    listings_df['Is Renovated * Years Since Renovation'] = listings_df['Is Renovated'] * listings_df['Years Since Renovation']
    listings_df.drop(['Years Since Renovation'], axis=1, inplace=True)

    if split and not test_X:
        listings_df[target] = y.values
        return split_data(listings_df, test_size=test_size)

    if test_X:
        return listings_df
    return listings_df, y

def predict_tab(train_datatab_prepr, y_train, test_datatab_prepr, test_datatab, max_iter=100, max_leaf_nodes=31, learning_rate=0.1,
                max_depth=None, drop_correlated=False, number_of_apartments_df=None,
                use_num_appt=False, drop_boolean=False, verbose=True, return_train=False):
    y_train = y_train.fillna(y_train.mean())
    hist = hist_gradient_boosting_train(train_datatab_prepr, y_train, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, learning_rate=learning_rate,
                                        max_depth=max_depth, ylog=False)
    hist_pred = hist_gradient_boosting_predict(test_datatab_prepr, hist, ylog=False, train=False)
    if return_train:
        hist_pred_train = hist_gradient_boosting_predict(train_datatab_prepr, hist, ylog=False, train=False)
        train_datatab['Prediction'] = hist_pred_train

    test_datatab['Prediction'] = hist_pred

    if verbose:
        print('Finished and returned predictions')
    if return_train:
        return test_datatab, test_datatab_prepr, hist, train_datatab
    else:
        return test_datatab, test_datatab_prepr, hist

def preprocess_and_split(df, FE_listings=None, target='PropertyFE', split_strategy="GroupKFold", n_splits=2, test_size=0.15, random_state=42):
    if target == 'PropertyFE':
        np.random.seed(random_state)
        """
        # Ensure FE_listings has unique Advertisement ID and Jahr pairs
        FE_listings_target = FE_listings.groupby(['Advertisement Id', 'Jahr']).agg({'PropertyFE': 'mean'}).reset_index()

        # Ensure unique Advertisement ID and Jahr pairs in df by taking the first instance (if duplicates exist)
        df = df.drop_duplicates(subset=['Advertisement Id', 'Jahr'])

        # Merge listings and FE datasets on Advertisement ID and Jahr
        df_merged = pd.merge(df, FE_listings_target, on=['Advertisement Id', 'Jahr'], how='left')
        print(df_merged.columns, 'columns___________')

        """

        #df_merged = df_merged.dropna(subset=[target])
        #df = df_merged
        df = df.dropna(subset=[target])        
        print(df)
    else:
        np.random.seed(random_state)
        df = df.dropna(subset=[target])

    # Extract building IDs
    df['Building ID'] = df['Property Reference Id'].str[:4]

    if split_strategy == "GroupKFold":
        if n_splits == 1:
            gkf = GroupKFold(n_splits=2)
            splits = list(gkf.split(df, groups=df['Building ID']))
            splits = [splits[0]]  # Keep only the first fold
        else:
            gkf = GroupKFold(n_splits=n_splits)
            splits = list(gkf.split(df, groups=df['Building ID']))
    elif split_strategy == "Simple":
        train_idx, test_idx = train_test_split(df.index, test_size=test_size, random_state=random_state, stratify=df['Building ID'])
        splits = [(train_idx, test_idx)]
    else:
        raise ValueError("Unknown split_strategy. Choose 'GroupKFold' or 'Simple'.")

    processed_splits = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        train_datatab = df.iloc[train_idx].copy()
        test_datatab = df.iloc[test_idx].copy()
        test_datatab['Prediction'] = np.nan

        train_datatab_prepr, y_train, test_datatab_prepr, y_test = preprocess_tab(
            train_datatab, test_datatab, target,
            number_of_apartments_df=None,
            drop_boolean=True, drop_correlated=True, use_num_appt=False,
            max_iter=max_iter_opts, learning_rate=learning_rate_opts,
            max_depth=max_depth_opts, max_leaf_nodes=max_leaf_nodes_opts
        )

        processed_splits.append((fold, train_datatab_prepr, y_train, test_datatab_prepr, y_test))

    all_features = set()
    for _, train_datatab_prepr, _, test_datatab_prepr, _ in processed_splits:
        all_features.update(train_datatab_prepr.columns)
        all_features.update(test_datatab_prepr.columns)

    all_features = list(all_features)

    for i, (fold, train_datatab_prepr, y_train, test_datatab_prepr, y_test) in enumerate(processed_splits):
        missing_train_features = set(all_features) - set(train_datatab_prepr.columns)
        missing_test_features = set(all_features) - set(test_datatab_prepr.columns)

        for feature in missing_train_features:
            train_datatab_prepr[feature] = np.nan
        for feature in missing_test_features:
            test_datatab_prepr[feature] = np.nan

        processed_splits[i] = (fold, train_datatab_prepr[all_features], y_train, test_datatab_prepr[all_features], y_test)

    return processed_splits

def main():
    # Load data
    listings_df = pd.read_csv(os.path.join(DATA_PATH, 'Listings_FE.csv'))
    properties_test = pd.read_csv(os.path.join(DATA_PATH, 'test_data_properties2.csv'))['Property Reference Id'].unique().tolist()
    # Filter the properties to get text_train and text_test
    text_train = listings_df[~listings_df['Property Reference Id'].isin(properties_test)]
    text_test = listings_df[listings_df['Property Reference Id'].isin(properties_test)]

    # Delete the rows where the text is empty
    text_train = text_train.dropna(subset=['Listing Description', 'PropertyFE'])
    text_test = text_test.dropna(subset=['Listing Description', 'PropertyFE'])

    number_of_apartments_df = pyaxis.parse(uri=os.path.join(DATA_PATH, 'vacancy_rate.px'), encoding='utf-8')
    print(number_of_apartments_df, "number_of_apartments_df")
    # Preprocess and split data
    FE_listings = pd.read_csv(os.path.join(DATA_PATH, 'Listings_FE.csv'))
    cross_validation_splits = preprocess_and_split(text_train, FE_listings=FE_listings, split_strategy="GroupKFold", n_splits=N_SPLITS)

    for i, (fold, train_datatab_prepr, y_train, test_datatab_prepr, y_test) in enumerate(cross_validation_splits):
        columns_to_drop_1 = [col for col in train_datatab_prepr.columns if col.startswith('Zip')]
        columns_to_drop_2 = [col for col in test_datatab_prepr.columns if col.startswith('Zip')]

        train_datatab_prepr = train_datatab_prepr.drop(columns=columns_to_drop_1)
        test_datatab_prepr = test_datatab_prepr.drop(columns=columns_to_drop_2)

        cross_validation_splits[i] = (fold, train_datatab_prepr, y_train, test_datatab_prepr, y_test)

    mse_scores = {}

    for learning_rate_opt in learning_rate_opts:
        for max_leaf_nodes_opt in max_leaf_nodes_opts:
            for max_iter_opt in max_iter_opts:
                for max_depth_opt in max_depth_opts:
                    fold_mse_scores = []
                    cross_val = cross_validation_splits.copy()
                    for fold, train_datatab_prepr, y_train, test_datatab_prepr, y_test in cross_val:
                        print(f"Fold {fold}, learning_rate: {learning_rate_opt}, max_leaf_nodes: {max_leaf_nodes_opt}, max_iter: {max_iter_opt}, max_depth: {max_depth_opt}")

                        if 'Prediction' in test_datatab_prepr.columns:
                            test_datatab_prepr = test_datatab_prepr.drop(columns=['Prediction'])

                        test_datatab_fit, test_datatab_prepr, trained_hist_model = predict_tab(
                            train_datatab_prepr, y_train, test_datatab_prepr, test_datatab_prepr,
                            number_of_apartments_df=None, drop_boolean=True,
                            drop_correlated=True, use_num_appt=False, max_iter=max_iter_opt,
                            learning_rate=learning_rate_opt, max_depth=max_depth_opt, max_leaf_nodes=max_leaf_nodes_opt
                        )

                        mse = mean_squared_error(test_datatab_fit['Prediction'], y_test)
                        fold_mse_scores.append(mse)
                        print(f"MSE on fold {fold} test set: {mse}")

                    avg_mse = np.mean(fold_mse_scores)
                    hyperparams = (learning_rate_opt, max_leaf_nodes_opt, max_iter_opt, max_depth_opt)
                    mse_scores[hyperparams] = avg_mse

                    print(f"Average MSE for learning_rate: {learning_rate_opt}, max_leaf_nodes: {max_leaf_nodes_opt}, max_iter: {max_iter_opt}, max_depth: {max_depth_opt}: {avg_mse}")

    with open(os.path.join(SAVE_PATH, 'prediction_tabular.pkl'), 'wb') as f:
        pickle.dump(mse_scores, f)

    heatmap_data = []
    for (learning_rate_opt, max_leaf_nodes_opt, max_iter_opt, max_depth_opt), mse in mse_scores.items():
        heatmap_data.append([learning_rate_opt, max_leaf_nodes_opt, max_iter_opt, mse])

    df_heatmap = pd.DataFrame(heatmap_data, columns=['learning_rate', 'max_leaf_nodes', 'max_iter', 'mse'])
    df_heatmap.to_csv(os.path.join(SAVE_PATH, 'mse_scores_5_splits.csv'), index=False)

if __name__ == "__main__":
    main()
