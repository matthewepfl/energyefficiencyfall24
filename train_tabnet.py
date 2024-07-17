import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.metrics import Metric
import optuna
import wandb

# Initialize wandb with the API key
wandb.login(key="4d425439d88047b5cd269aef0271ab70453f2402")
wandb.init(project="energyefficiency")

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


class MSE(Metric):
    def __init__(self):
        self._name = "mse"
        self._maximize = False

    def __call__(self, y_true, y_score):
        mse = mean_squared_error(y_true, y_score)
        return mse


def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 0.001, 0.01)  # Reduced range
    n_steps = trial.suggest_int('n_steps', 3, 10)  # Reduced range
    gamma = trial.suggest_uniform('gamma', 1.0, 1.5)  # Reduced range
    n_independent = trial.suggest_int('n_independent', 1, 3)  # Reduced range
    n_shared = trial.suggest_int('n_shared', 1, 3)  # Reduced range
    momentum = trial.suggest_uniform('momentum', 0.01, 0.1)  # Reduced range
    weight_decay = trial.suggest_loguniform('weight_decay', 0.00001, 0.001)  # Reduced range
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])  # Reduced and categorical
    virtual_batch_size = trial.suggest_categorical('virtual_batch_size', [32, 64])  # Reduced and categorical

    # Initialize and train the TabNet model
    clf = TabNetRegressor(
        optimizer_params=dict(lr=lr, weight_decay=weight_decay),
        n_steps=n_steps,
        gamma=gamma,
        n_independent=n_independent,
        n_shared=n_shared,
        momentum=momentum
    )
    clf.fit(
        X_train=X_train_np, y_train=Y_train_np,
        eval_set=[(X_valid_np, y_valid_np)],
        max_epochs=200,  # Reduced epochs for quicker trials
        patience=20,  # Reduced patience
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=20,  # Use 20 CPU cores
        drop_last=False,
        eval_metric=['rmse']
    )
    
    # Predict on validation set
    y_pred = clf.predict(X_valid_np)
    
    # Calculate RMSE
    rmse = mean_squared_error(y_valid_np, y_pred, squared=False)
    
    # Log to wandb
    wandb.log({'validation_rmse': rmse, 'trial_number': trial.number})
    
    return rmse

# Create study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # Reduced number of trials

# Get best hyperparameters
best_params = study.best_params
print("Best hyperparameters: ", best_params)

# Prepare to log metrics to CSV
results = []

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
    max_epochs=200,
    patience=20,
    batch_size=best_params['batch_size'],
    virtual_batch_size=best_params['virtual_batch_size'],
    num_workers=20,
    drop_last=False,
    eval_metric=['rmse']
)

# Predict on test set
train_pred = clf.predict(X_train_np)
valid_pred = clf.predict(X_valid_np)
test_pred = clf.predict(X_test_np)

train_mse = mean_squared_error(Y_train_np, train_pred)
valid_mse = mean_squared_error(y_valid_np, valid_pred)
test_mse = mean_squared_error(y_test_np, test_pred)

train_loss = clf.history['loss']
valid_loss = clf.history['val_0_rmse']

print(f"Train MSE: {train_mse}")
print(f"Validation MSE: {valid_mse}")
print(f"Test MSE: {test_mse}")

# Log final results to wandb
wandb.log({'train_mse': train_mse, 'valid_mse': valid_mse, 'test_mse': test_mse})

# Save results to CSV
results.append({
    'trial_number': 'final',
    'train_mse': train_mse,
    'valid_mse': valid_mse,
    'test_mse': test_mse,
    'train_loss': train_loss,
    'valid_loss': valid_loss
})

results_df = pd.DataFrame(results)
results_df.to_csv('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/results.csv', index=False)

# Function to run the training with Optuna
def run_study():
    study.optimize(objective, n_trials=20)
    
    # Get best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters: ", best_params)
    
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
        max_epochs=200,
        patience=20,
        batch_size=best_params['batch_size'],
        virtual_batch_size=best_params['virtual_batch_size'],
        num_workers=20,
        drop_last=False,
        eval_metric=['rmse']
    )
    
    # Predict on test set
    train_pred = clf.predict(X_train_np)
    valid_pred = clf.predict(X_valid_np)
    test_pred = clf.predict(X_test_np)
    
    train_mse = mean_squared_error(Y_train_np, train_pred)
    valid_mse = mean_squared_error(y_valid_np, valid_pred)
    test_mse = mean_squared_error(y_test_np, test_pred)
    
    train_loss = clf.history['loss']
    valid_loss = clf.history['val_0_rmse']
    
    print(f"Train MSE: {train_mse}")
    print(f"Validation MSE: {valid_mse}")
    print(f"Test MSE: {test_mse}")
    
    # Log final results to wandb
    wandb.log({'train_mse': train_mse, 'valid_mse': valid_mse, 'test_mse': test_mse})
    
    # Save results to CSV
    results.append({
        'trial_number': 'final',
        'train_mse': train_mse,
        'valid_mse': valid_mse,
        'test_mse': test_mse,
        'train_loss': train_loss,
        'valid_loss': valid_loss
    })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/results.csv', index=False)

# Run the study
run_study()
