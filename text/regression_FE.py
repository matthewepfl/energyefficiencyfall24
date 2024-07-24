import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt



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
            'targets': torch.tensor(target, dtype=torch.float)
        }


# Model with regression head
class BERTRegressor(nn.Module):
    def __init__(self, bert_model):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.regressor(pooled_output)
    
# Training function
def train_epoch(model, data_loader, criterion, optimizer, device):
    model = model.train()
    losses = []
    for _, d in enumerate(tqdm(data_loader, desc="Training")):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        optimizer.zero_grad()  

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.squeeze(1), targets.float())

        loss.backward()         
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


# Evaluation function
def eval_model(model, data_loader, criterion, device):
    model = model.eval()
    losses = []
    predictions = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets.unsqueeze(1))
            predictions.extend(outputs.cpu().detach().numpy())

            losses.append(loss.item())
    return np.mean(losses)

# Training loop function
def train(model, train_loader, test_loader, criterion, optimizer, device, epochs, save_path=None):
    # Freeze the BERT parameters
    for param in model.bert.parameters():
        param.requires_grad = False

    best_test_loss = float('inf')  

    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train loss: {train_loss}')

        test_loss = eval_model(
            model,
            test_loader,
            criterion,
            device
        )

        print(f'Test loss: {test_loss}')

        # Save the model if the test loss is the best we've seen so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f'Model saved at epoch {epoch + 1} with test loss {test_loss}')