import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import torch.nn as nn
import gc
import wandb
import os

# Initialize wandb with the API key
wandb.login(key="4d425439d88047b5cd269aef0271ab70453f2402")
wandb.init(project="energyefficiency")

# Load the data
listings_df = pd.read_csv('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/Listings_FE.csv')
properties_test = pd.read_csv('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/test_data_properties2.csv')['Property Reference Id'].unique().tolist()

# Filter the properties to get text_train and text_test
text_train = listings_df[~listings_df['Property Reference Id'].isin(properties_test)]  # Keep all except test data
text_test = listings_df[listings_df['Property Reference Id'].isin(properties_test)]

# Delete the rows where the text is empty
text_train = text_train.dropna(subset=['Listing Description', 'PropertyFE'])
text_test = text_test.dropna(subset=['Listing Description', 'PropertyFE'])

expanded_keywords = pd.read_csv('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/expanded_keywords.csv').iloc[:, 0].tolist()

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
            'keyword_count': torch.tensor(keyword_count, dtype=torch.float)
        }

# Load tokenizer and pretrained model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Create the dataset
MAX_LEN = 500
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
num_workers = 20

train_dataset = ListingsDataset(text_train, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=num_workers)

test_dataset = ListingsDataset(text_test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=num_workers)

# Define the BERT model for embeddings extraction
class BERTModelForEmbeddings(nn.Module):
    def __init__(self, bert_model):
        super(BERTModelForEmbeddings, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return pooled_output

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTModelForEmbeddings('bert-base-multilingual-cased').to(device)

# Function to extract embeddings
def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    keyword_counts = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            keyword_count = d["keyword_count"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(output.cpu().numpy())
            keyword_counts.append(keyword_count.cpu().numpy())

            # Manually trigger garbage collection and clear CUDA cache
            gc.collect()
            torch.cuda.empty_cache()

    return np.vstack(embeddings), np.hstack(keyword_counts)

# Extract embeddings for train and test data
train_embeddings, train_keyword_counts = extract_embeddings(model, train_loader, device)
test_embeddings, test_keyword_counts = extract_embeddings(model, test_loader, device)

# Save embeddings to disk
np.save('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/train_embeddings.npy', train_embeddings)
np.save('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/train_keyword_counts.npy', train_keyword_counts)
np.save('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/test_embeddings.npy', test_embeddings)
np.save('/scratch/izar/mmorvan/EnergyEfficiencyPredictionMatthew/data/test_keyword_counts.npy', test_keyword_counts)
