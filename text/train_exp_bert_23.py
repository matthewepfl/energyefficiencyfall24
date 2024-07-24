import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import gc
import wandb
from sklearn.metrics import mean_squared_error

"""
Combination 2: Baseline
Learning Rate: 1e-5
Dropout Rate: 0.3
Weight Decay: 1e-2
Batch Size: 32
Epochs: 50
"""

# Initialize wandb with the API key
wandb.login(key="4d425439d88047b5cd269aef0271ab70453f2402")
wandb.init(project="energyefficiency", name=f"23_lr-1e-5_dropout-0.3_weightdecay-1e-2_batchsize-32_epochs-50")

# Load the data
data_path = '/scratch/izar/mmorvan/EnergyEfficiencyPrediction/data/'
listings_df = pd.read_csv(os.path.join(data_path, 'Listings_FE.csv'))

# Load property data
properties_train = pd.read_csv(os.path.join(data_path, 'cross_val/train_data_properties23.csv'))['Property Reference Id'].unique().tolist()
properties_val = pd.read_csv(os.path.join(data_path, 'cross_val/val_data_properties23.csv'))['Property Reference Id'].unique().tolist()
properties_test = pd.read_csv(os.path.join(data_path, 'cross_val/test_data_properties23.csv'))['Property Reference Id'].unique().tolist()

# Filter the properties to get text_train, text_val, and text_test
text_train = listings_df[listings_df['Property Reference Id'].isin(properties_train)]
text_val = listings_df[listings_df['Property Reference Id'].isin(properties_val)]
text_test = listings_df[listings_df['Property Reference Id'].isin(properties_test)]

# Delete the rows where the text is empty
text_train = text_train.dropna(subset=['Listing Description', 'PropertyFE'])
text_val = text_val.dropna(subset=['Listing Description', 'PropertyFE'])
text_test = text_test.dropna(subset=['Listing Description', 'PropertyFE'])

expanded_keywords = pd.read_csv(os.path.join(data_path, 'expanded_keywords.csv')).iloc[:, 0].tolist()

# Function to count keywords
def count_keywords(description, keywords):
    if isinstance(description, float) and pd.isna(description):
        return 0
    count = 0
    description = str(description).lower()
    for keyword in keywords:
        count += description.count(keyword)
    return count

# Integrate expanded keywords in the dataset
text_train['keyword_count'] = text_train['Listing Description'].apply(lambda x: count_keywords(x, expanded_keywords))
text_val['keyword_count'] = text_val['Listing Description'].apply(lambda x: count_keywords(x, expanded_keywords))
text_test['keyword_count'] = text_test['Listing Description'].apply(lambda x: count_keywords(x, expanded_keywords))

# Dataset class
class ListingsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        description = str(self.data.iloc[index]['Listing Description'])
        target = self.data.iloc[index]['PropertyFE']
        keyword_count = self.data.iloc[index]['keyword_count']

        encoding = self.tokenizer(
            description,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'keyword_count': torch.tensor(keyword_count, dtype=torch.float),
            'targets': torch.tensor(target, dtype=torch.float)
        }

# Define the BERTRegressor model
class BERTRegressor(nn.Module):
    def __init__(self, bert_model, dropout_rate=0.3):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 1, 128),  # +1 for the keyword count
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, keyword_count):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        combined_output = torch.cat((pooled_output, keyword_count.unsqueeze(1)), dim=1)
        return self.regressor(combined_output)

# Initialize model, optimizer, and criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTRegressor('bert-base-multilingual-cased', dropout_rate=0.3).to(device)
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
criterion = nn.MSELoss()

# Initialize GradScaler
scaler = torch.cuda.amp.GradScaler()

# Define a learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training function with mixed precision, gradient clipping, and logging
def train_epoch(model, data_loader, criterion, optimizer, scaler, device):
    model = model.train()
    losses = []
    all_targets = []
    all_predictions = []
    for _, d in enumerate(tqdm(data_loader, desc="Training")):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        keyword_count = d["keyword_count"].to(device)
        targets = d["targets"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, keyword_count=keyword_count)
            loss = criterion(outputs.squeeze(1), targets.float())

        scaler.scale(loss).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(outputs.squeeze(1).detach().cpu().numpy())

        # Manually trigger garbage collection and clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

    mse = mean_squared_error(all_targets, all_predictions)
    return np.mean(losses), mse

# Evaluation function with mixed precision
def eval_model(model, data_loader, criterion, device):
    model = model.eval()
    losses = []
    predictions = []
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            keyword_count = d["keyword_count"].to(device)
            targets = d["targets"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, keyword_count=keyword_count)
                loss = criterion(outputs, targets.unsqueeze(1))
                predictions.extend(outputs.cpu().detach().numpy())

            losses.append(loss.item())
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.squeeze(1).detach().cpu().numpy())

            # Manually trigger garbage collection and clear CUDA cache
            gc.collect()
            torch.cuda.empty_cache()

    mse = mean_squared_error(all_targets, all_predictions)
    return np.mean(losses), mse

# Training loop function
def train(model, train_loader, val_loader, test_loader, criterion, optimizer, scaler, scheduler, device, epochs, save_path=None):
    best_test_loss = float('inf')
    for epoch in range(epochs):
        train_loss, train_mse = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train loss: {train_loss}')
        print(f'Train MSE: {train_mse}')

        # Update the learning rate
        scheduler.step()

        val_loss, val_mse = eval_model(model, val_loader, criterion, device)
        print(f'Val loss: {val_loss}')
        print(f'Val MSE: {val_mse}')

        test_loss, test_mse = eval_model(model, test_loader, criterion, device)
        print(f'Test loss: {test_loss}')
        print(f'Test MSE: {test_mse}')

        # Log metrics to wandb
        wandb.log({"train_loss": train_loss, "train_mse": train_mse, "val_loss": val_loss, "val_mse": val_mse, "test_loss": test_loss, "test_mse": test_mse})

        if save_path and test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved at epoch {epoch + 1} with test loss {test_loss}') 

# Define Parameters
MAX_LEN = 500
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 50  # Adjust number of epochs if needed

# Load tokenizer and pretrained model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Create the datasets and dataloaders
num_workers = 20
train_dataset = ListingsDataset(text_train, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=num_workers)

val_dataset = ListingsDataset(text_val, tokenizer, MAX_LEN)
val_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=num_workers)

test_dataset = ListingsDataset(text_test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=num_workers)

# Train and save the model
train(model, train_loader, val_loader, test_loader, criterion, optimizer, scaler, scheduler, device, EPOCHS, save_path="model/text_model_full_cross_val_1_properties23.pth")

# Function to save predictions to CSV
def save_predictions(model, data_loader, device, file_path):
    model = model.eval()
    predictions = []
    property_ids = []
    all_targets = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            keyword_count = d["keyword_count"].to(device)
            targets = d["targets"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, keyword_count=keyword_count)
                predictions.extend(outputs.cpu().detach().numpy())
                property_ids.extend(d['Property Reference Id'])
                all_targets.extend(targets.cpu().numpy())

    df = pd.DataFrame({
        'Property Reference Id': property_ids,
        'Predicted FE': np.squeeze(predictions),
        'Actual FE': all_targets
    })
    df.to_csv(file_path, index=False)

# Save predictions for train, validation, and test sets
save_predictions(model, train_loader, device, "train_predictions_properties23.csv")
save_predictions(model, val_loader, device, "val_predictions_properties23.csv")
save_predictions(model, test_loader, device, "test_predictions_properties23.csv")

# Calculate final RMSE and NMBE
def calculate_final_metrics(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    nmbe = np.mean(np.array(predictions) - np.array(labels)) / np.mean(np.array(labels))
    return rmse, nmbe

train_predictions_df = pd.read_csv("train_predictions_properties23.csv")
val_predictions_df = pd.read_csv("val_predictions_properties23.csv")
test_predictions_df = pd.read_csv("test_predictions_properties23.csv")

train_rmse, train_nmbe = calculate_final_metrics(train_predictions_df['Predicted FE'], train_predictions_df['Actual FE'])
val_rmse, val_nmbe = calculate_final_metrics(val_predictions_df['Predicted FE'], val_predictions_df['Actual FE'])
test_rmse, test_nmbe = calculate_final_metrics(test_predictions_df['Predicted FE'], test_predictions_df['Actual FE'])

final_metrics = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [train_rmse, val_rmse, test_rmse],
    'NMBE': [train_nmbe, val_nmbe, test_nmbe]
})

final_metrics.to_csv("final_metrics_properties23.csv", index=False)

print(final_metrics)
