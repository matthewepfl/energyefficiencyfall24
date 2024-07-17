import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
import wandb
import os

# Load embeddings
bert_train_embeddings = np.load('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/train_embeddings.npy')
bert_test_embeddings = np.load('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/test_embeddings.npy')
bert_train_keyword_counts = np.load('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/train_keyword_counts.npy')
bert_test_keyword_counts = np.load('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/test_keyword_counts.npy')

tabnet_train_embeddings = np.load('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/tabnet_train_embeddings.npy')
tabnet_test_embeddings = np.load('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/tabnet_test_embeddings.npy')

# Combine embeddings and keyword counts
train_features = np.hstack((bert_train_embeddings, bert_train_keyword_counts.reshape(-1, 1), tabnet_train_embeddings))
test_features = np.hstack((bert_test_embeddings, bert_test_keyword_counts.reshape(-1, 1), tabnet_test_embeddings))

# Assuming target values are the same for both embeddings
train_targets = Y_train_np
test_targets = y_test_np

# Create dataset class
class CombinedDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.tensor(self.features[index], dtype=torch.float32)
        y = torch.tensor(self.targets[index], dtype=torch.float32)
        return x, y

train_dataset = CombinedDataset(train_features, train_targets)
test_dataset = CombinedDataset(test_features, test_targets)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the final neural network
class FinalRegressor(nn.Module):
    def __init__(self, input_dim):
        super(FinalRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.regressor(x)

# Initialize model, optimizer, and criterion
input_dim = train_features.shape[1]
model = FinalRegressor(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

# Training function
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    losses = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)

# Evaluation function
def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    predictions = []
    actuals = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            losses.append(loss.item())
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(y.cpu().numpy())

    mse = mean_squared_error(actuals, predictions)
    return np.mean(losses), mse

# Training loop
epochs = 50
best_test_loss = float('inf')

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_mse = eval_model(model, test_loader, criterion, device)
    
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test MSE: {test_mse}')
    
    # Log metrics to wandb
    wandb.log({"train_loss": train_loss, "test_loss": test_loss, "test_mse": test_mse})
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), "/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/final_model.pth")
        print(f'Model saved at epoch {epoch + 1} with test loss {test_loss}')
