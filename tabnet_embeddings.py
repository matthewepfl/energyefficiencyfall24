from pytorch_tabnet.tab_model import TabNetRegressor
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import os

# Load the data
data_path = '/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/'
X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
Y_train = pd.read_csv(os.path.join(data_path, 'Y_train.csv'))
X_valid = pd.read_csv(os.path.join(data_path, 'X_valid.csv'))
y_valid = pd.read_csv(os.path.join(data_path, 'y_valid.csv'))
X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))

# Convert DataFrames to numpy arrays
X_train_np = X_train.values
Y_train_np = Y_train.values
X_valid_np = X_valid.values
y_valid_np = y_valid.values
X_test_np = X_test.values
y_test_np = y_test.values

# Load best parameters from previous Optuna study (assume they are saved)
best_params = {
    'lr': 0.005,
    'n_steps': 5,
    'gamma': 1.3,
    'n_independent': 2,
    'n_shared': 2,
    'momentum': 0.03,
    'weight_decay': 0.0001,
    'batch_size': 128,
    'virtual_batch_size': 64
}

# Train final model with best hyperparameters
clf = TabNetRegressor(
    optimizer_params=dict(lr=best_params['lr'], weight_decay=best_params['weight_decay']),
    n_steps=best_params['n_steps'],
    gamma=best_params['gamma'],
    n_independent=best_params['n_independent'],
    n_shared=best_params['n_shared'],
    momentum=best_params['momentum']
)
clf.fit(
    X_train=X_train_np, y_train=Y_train_np,
    eval_set=[(X_valid_np, y_valid_np)],
    max_epochs=2,
    patience=20,
    batch_size=best_params['batch_size'],
    virtual_batch_size=best_params['virtual_batch_size'],
    num_workers=20,
    drop_last=False,
    eval_metric=['rmse']
)

# Extract embeddings
train_embeddings = clf.predict(X_train_np)
valid_embeddings = clf.predict(X_valid_np)
test_embeddings = clf.predict(X_test_np)

# Save embeddings to disk
np.save('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/tabnet_train_embeddings.npy', train_embeddings)
np.save('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/tabnet_valid_embeddings.npy', valid_embeddings)
np.save('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/tabnet_test_embeddings.npy', test_embeddings)
