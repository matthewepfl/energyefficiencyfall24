import torch
import os
import wandb
import pickle
from torch.optim.lr_scheduler import LambdaLR


def parse_list_of_floats(string):
    return [float(item.strip()) for item in string.strip('[]').split(',')]

def parse_nested_list_of_ints(string):
    return string.strip("[]").split(',') # input is like: "512-256-124,512-256"

def mask_branch_type(string):
    return [int(item.strip()) for item in string.strip('[]').split(',')]


def my_train_model(model, train_data, val_data, lr, weight_decay, num_epochs, seed, run_name, checkpoint_dir='checkpoints'):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    def linear_lr_lambda(epoch):
        return lr + (lr * 1e-1 - lr) * (epoch / (num_epochs - 1))

    scheduler = LambdaLR(optimizer, lr_lambda=linear_lr_lambda)
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{run_name}_best.pth')

    model.train()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Log training loss
            wandb.log({'train_loss': loss.item()})
            wandb.log({'learning_rate': scheduler.get_last_lr()[0]})

            scheduler.step()
            
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}')

        # Evaluate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                loss = criterion(outputs, batch['labels'])
                val_loss += loss.item()

                # Log validation loss
                wandb.log({'val_loss': loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}')

        # Save the best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Best model saved at epoch {epoch+1} with validation loss {best_val_loss}')

    print(f'Best Validation Loss: {best_val_loss} - Model saved at {checkpoint_path}')
    return model

def my_evaluate_model(model, train_loader, val_loader, test_loader, do_train, checkpoint_path):

    if not do_train and checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path + '/pytorch_model.bin'))
        print(f'Model loaded from checkpoint {checkpoint_path} for evaluation.')

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch)
            predictions.append(outputs)
            labels.append(batch['labels'])
            eval_loss = torch.nn.MSELoss()(outputs, batch['labels'])
            wandb.log({'test_loss': eval_loss.item()})

    # save predictions
    predictions_path = os.path.join(checkpoint_path, 'predictions.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump({'predictions': predictions, 'labels': labels}, f)
        
    # do the same for train and val
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in train_loader:
            outputs = model(batch)
            predictions.append(outputs)
            labels.append(batch['labels'])
            eval_loss = torch.nn.MSELoss()(outputs, batch['labels'])
            wandb.log({'train_loss': eval_loss.item()})
    predictions_path = os.path.join(checkpoint_path, 'predictions_train.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump({'predictions': predictions, 'labels': labels}, f)

    predictions = []
    labels = []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            predictions.append(outputs)
            labels.append(batch['labels'])
            eval_loss = torch.nn.MSELoss()(outputs, batch['labels'])
            wandb.log({'val_loss': eval_loss.item()})
    predictions_path = os.path.join(checkpoint_path, 'predictions_val.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump({'predictions': predictions, 'labels': labels}, f)
