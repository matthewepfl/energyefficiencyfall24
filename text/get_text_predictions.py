import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import gc
from sklearn.metrics import mean_squared_error




# Define the ListingsDataset class
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
            'targets': torch.tensor(target, dtype=torch.float),
            'property_id': self.data.iloc[index]['Property Reference Id']
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
            property_id = d["property_id"]

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, keyword_count=keyword_count)
                predictions.extend(outputs.cpu().detach().numpy())
                property_ids.extend(property_id)
                all_targets.extend(targets.cpu().numpy())

    df = pd.DataFrame({
        'Property Reference Id': property_ids,
        'Predicted FE': np.squeeze(predictions),
        'Actual FE': all_targets
    })
    df.to_csv(file_path, index=False)

# Load the data
data_path = '/scratch/izar/mmorvan/EnergyEfficiencyPrediction/data/'
results_path = '/scratch/izar/mmorvan/EnergyEfficiencyPrediction/text/results/'

listings_df = pd.read_csv(os.path.join(data_path, 'Listings_FE.csv'))
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
listings_df['keyword_count'] = listings_df['Listing Description'].apply(lambda x: count_keywords(x, expanded_keywords))

# Loop over the properties datasets
for i in range(21, 26):
    properties_train = pd.read_csv(os.path.join(data_path, f'cross_val/train_data_properties{i}.csv'))['Property Reference Id'].unique().tolist()
    properties_val = pd.read_csv(os.path.join(data_path, f'cross_val/val_data_properties{i}.csv'))['Property Reference Id'].unique().tolist()
    properties_test = pd.read_csv(os.path.join(data_path, f'cross_val/test_data_properties{i}.csv'))['Property Reference Id'].unique().tolist()

    # Filter the properties to get text_train, text_val, and text_test
    text_train = listings_df[listings_df['Property Reference Id'].isin(properties_train)]
    text_val = listings_df[listings_df['Property Reference Id'].isin(properties_val)]
    text_test = listings_df[listings_df['Property Reference Id'].isin(properties_test)]

    # Delete the rows where the text is empty
    text_train = text_train.dropna(subset=['Listing Description', 'PropertyFE'])
    text_val = text_val.dropna(subset=['Listing Description', 'PropertyFE'])
    text_test = text_test.dropna(subset=['Listing Description', 'PropertyFE'])

    # Define Parameters
    MAX_LEN = 500
    BATCH_SIZE = 32

    # Load tokenizer and pretrained model
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Create the datasets and dataloaders
    num_workers = 4  # Adjust this if you encounter issues or want to use more workers
    train_dataset = ListingsDataset(text_train, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    val_dataset = ListingsDataset(text_val, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    test_dataset = ListingsDataset(text_test, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # Initialize model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTRegressor('bert-base-multilingual-cased', dropout_rate=0.3).to(device)
    model_path = f"/scratch/izar/mmorvan/EnergyEfficiencyPrediction/text/model/text_model_full_cross_val_1_properties{i}.pth"
    model.load_state_dict(torch.load(model_path))

    # Save predictions for train, validation, and test sets
    save_predictions(model, train_loader, device, os.path.join(results_path, f"train_predictions_properties{i}.csv"))
    save_predictions(model, val_loader, device, os.path.join(results_path, f"val_predictions_properties{i}.csv"))
    save_predictions(model, test_loader, device, os.path.join(results_path, f"test_predictions_properties{i}.csv"))
