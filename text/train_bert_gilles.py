import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
import wandb
from regression_FE import ListingsDataset, BERTRegressor, train

# WandB setup
wandb.login(key="4d425439d88047b5cd269aef0271ab70453f2402")
wandb.init(project="energyefficiency")

# Define paths
data_path = '/scratch/izar/mmorvan/EnergyEfficiencyPrediction/data/'
script_path = '/scratch/izar/mmorvan/EnergyEfficiencyPrediction/text/'

# Parameters
seed = 42
test_size = 0.2
MAX_LEN = 500
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 2e-5
weight_decay = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
listings_df = pd.read_csv(os.path.join(data_path, 'Listings_FE.csv'))
properties_train = pd.read_csv(os.path.join(data_path, 'train_data_properties2.csv'))['Property Reference Id'].unique().tolist()
properties_test = pd.read_csv(os.path.join(data_path, 'test_data_properties2.csv'))['Property Reference Id'].unique().tolist()

# Filter and preprocess data
text_train = listings_df[listings_df['Property Reference Id'].isin(properties_train)]
text_test = listings_df[listings_df['Property Reference Id'].isin(properties_test)]
text_train = text_train.loc[text_train.groupby('Property Reference Id')['Advertisement Version Id'].idxmax()]
text_test = text_test.loc[text_test.groupby('Property Reference Id')['Advertisement Version Id'].idxmax()]
text_train = text_train.dropna(subset=['Listing Description'])
text_train['Property Reference Id'] = pd.Categorical(text_train['Property Reference Id'], categories=properties_train, ordered=True)
text_train = text_train.sort_values('Property Reference Id')
text_test['Property Reference Id'] = pd.Categorical(text_test['Property Reference Id'], categories=properties_test, ordered=True)
text_test = text_test.sort_values('Property Reference Id')

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BERTRegressor('bert-base-multilingual-cased').to(device)

# Create datasets and dataloaders
train_dataset = ListingsDataset(text_train, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_dataset = ListingsDataset(text_test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
criterion = nn.MSELoss()

# Train and save the model
train(model, train_loader, test_loader, criterion, optimizer, device, EPOCHS, save_path=os.path.join(script_path, "text_model_full_gilles.pth"))

# Load saved model and make predictions
saved_model = BERTRegressor('bert-base-multilingual-cased')
saved_model.load_state_dict(torch.load(os.path.join(script_path, 'text_model_full_gilles.pth')))
saved_model.to(device)

predictions = []
labels = []
saved_model.eval()
with torch.no_grad():
    for d in test_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = saved_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions.extend(outputs.cpu().numpy().flatten().tolist())
        labels.extend(targets.cpu().numpy().flatten().tolist())

# Calculate metrics
mse = np.mean((np.array(predictions) - np.array(labels)) ** 2)
rmse = np.sqrt(mse)
nmbe = np.mean(np.array(predictions) - np.array(labels)) / np.mean(np.array(labels))

# Save metrics and predictions to csv
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'NMBE'],
    'Value': [mse, rmse, nmbe]
})
metrics_df.to_csv(os.path.join(script_path, "metrics.csv"), index=False)

predictions_df = pd.DataFrame({
    'Property Reference Id': properties_test,
    'Predicted FE': predictions,
    'Actual FE': labels
})
predictions_df.to_csv(os.path.join(script_path, "text_predictions_gilles.csv"), index=False)

print(predictions_df)
print(metrics_df)
