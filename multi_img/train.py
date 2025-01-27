'''
Grid search for demand calculation using joint image encoders. 
'''

import os
import wandb
import argparse
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup

from transformers import EarlyStoppingCallback
import itertools
from typing import Optional, List
from torch.optim import lr_scheduler

from models import *
from data import *
from helpers import *

workingOn = 'server'
# ---------------------------------------- GLOBAL VARIABLES ---------------------------------------- #

# Global configurations
if workingOn == 'server':
    BASE_DIR = '/work/FAC/HEC/DEEP/shoude/ml_green_building/'
else:
    BASE_DIR = '/Users/silviaromanato/Desktop/'
    
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
BATCH_SIZE = 64
print("the batch size is: ", BATCH_SIZE)

# ---------------------------------------- TRAINING FUNCTIONS ---------------------------------------- #

class MultimodalTrainer(Trainer):
    '''
    Custom trainer for multimodal models.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        Computes MSE loss.
        '''
        outputs = model(**inputs)
        predictions = outputs['prediction']
        labels = inputs['labels'].to(predictions.device)
        loss = nn.MSELoss()(predictions, labels)
        return (loss, outputs) if return_outputs else loss

def create_trainer(model, 
                   train_data, 
                   val_data, 
                   output_dir,
                   run_name, 
                   epochs=10, 
                   lr=1e-5, 
                   batch_size=32, 
                   weight_decay=1e-2, 
                   seed=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data) * epochs)

    training_args = TrainingArguments(

        # Training
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.999,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        dataloader_num_workers=0, 
        seed=seed,

        # Evaluation & checkpointing
        run_name=run_name,
        report_to='wandb',
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        save_total_limit=3,
        logging_dir='./logs',
        logging_first_step=True,
        logging_steps=1,
        logging_strategy='epoch',
        load_best_model_at_end=True,
    )

    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=val_data.collate_fn,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
        optimizers=(optimizer, scheduler)
    )

    return trainer

def freeze_vision_encoder_layers(model, vision: Optional[str]):
    if vision:
        for param in model.vision_encoder.model_0.parameters():
            param.requires_grad = False
        for param in model.vision_encoder.model_1.parameters():
            param.requires_grad = False
        for param in model.vision_encoder.model_2.parameters():
            param.requires_grad = False
        for param in model.vision_encoder.model_3.parameters():
            param.requires_grad = False
        for param in model.vision_encoder.model_4.parameters():
            param.requires_grad = False
        for param in model.vision_encoder.model_5.parameters():
            param.requires_grad = False

    print(model)

def train_model(model, train_data, val_data, lr, weight_decay, num_epochs, seed, run_name):
    print('Training: Starting training')
    trainer = create_trainer(model, train_data, val_data, CHECKPOINTS_DIR, run_name=run_name,
                            epochs=num_epochs, lr=lr, batch_size=BATCH_SIZE, 
                            weight_decay=weight_decay, seed=seed)
    trainer.train()

    model_path = os.path.join(CHECKPOINTS_DIR, f'final_model_{run_name}.bin')
    torch.save(model.state_dict(), model_path)

def evaluate_model(model, train_data, val_data, test_data, lr, weight_decay, num_epochs, seed, do_train, checkpoint_path, run_name):
    if not do_train and checkpoint_path:

        model.load_state_dict(torch.load(checkpoint_path + '/pytorch_model.bin'))
        print(f'Model loaded from checkpoint {checkpoint_path} for evaluation.')

    trainer = create_trainer(model, train_data, val_data, CHECKPOINTS_DIR, run_name=run_name,
                             epochs=num_epochs, lr=lr, batch_size=BATCH_SIZE, 
                             weight_decay=weight_decay, seed=seed)

    # Evaluation
    print('Evaluation:\tStarting evaluation')
    eval_results = trainer.evaluate(eval_dataset=test_data)
    wandb.log(eval_results)
    print('Evaluation:\tResults\n', eval_results)

    # Predictions: Test
    test_predictions = trainer.predict(test_data)
    test_labels = test_data.get_labels()
    test_adv_id = test_data.get_adv_id()

    # Train
    train_predictions = trainer.predict(train_data)
    train_labels = train_data.get_labels()
    train_adv_id = train_data.get_adv_id()

    # Validation
    eval_predictions = trainer.predict(val_data)
    eval_labels = val_data.get_labels()
    eval_adv_id = val_data.get_adv_id()

    # convert to list 
    test_predictions, test_labels, test_adv_id  = test_predictions.predictions.tolist(), test_labels.tolist(), test_adv_id.tolist()
    train_predictions, train_labels, train_adv_id = train_predictions.predictions.tolist(), train_labels.tolist(), train_adv_id.tolist()
    eval_predictions, eval_labels, eval_adv_id = eval_predictions.predictions.tolist(), eval_labels.tolist(), eval_adv_id.tolist()

    # save predictions
    predictions_path = os.path.join(CHECKPOINTS_DIR, f'predictions_{run_name}.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump({'train': (train_predictions, train_labels, train_adv_id), 'val': (eval_predictions, eval_labels, eval_adv_id), 'test': (test_predictions, test_labels, test_adv_id)}, f)

def grid_search(train_data, val_data, test_data,
            vision: List[str] = ['resnet50'],
            hidden_dims: List[int] = ['512-256'],
            dropout_prob: List[float] = 0.0,
            batch_norm: List[bool] = False,
            lr: List[float] = 0.001,
            weight_decay: List[float] = 0.0,
            num_epochs: int = 20,
            seed: int = 0,
            do_train: bool = True,
            do_eval: bool = False, 
            checkpoint_path: Optional[str] = None,
            mask_branch: List[int] = [],
            ):
    '''
    Grid search for radiology diagnosis using joint image encoders. 
    '''
    print("Training:\t", do_train, "\nEvaluation:\t", do_eval, "\nCheckpoint path:\t", checkpoint_path, "\nMask branch:\t", mask_branch)
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Convert single value parameters to lists if they are not already
    vision = [args.vision] if isinstance(args.vision, str) else args.vision
    hidden_dims = args.hidden_dims  # This should be a list of lists if parsed correctly
    dropout_prob = [args.dropout_prob] if isinstance(args.dropout_prob, float) else args.dropout_prob
    batch_norm = [args.batch_norm]  # Boolean, should be in a list if there's only one value
    lr = [args.lr] if isinstance(args.lr, float) else args.lr
    weight_decay = [args.weight_decay] if isinstance(args.weight_decay, float) else args.weight_decay

    print('Grid search:\tStarting grid search')
    for vision, hidden_dims, dropout_prob, batch_norm, lr, weight_decay in itertools.product(vision, hidden_dims, dropout_prob, batch_norm, lr, weight_decay):
        
        print(f'Vision: {vision}, Hidden dims: {hidden_dims}, Dropout: {dropout_prob}, Batch norm: {batch_norm}, LR: {lr}, Weight decay: {weight_decay}')
        run_name = f'{vision}_{lr}_{weight_decay}_{num_epochs}_{hidden_dims}_{dropout_prob}_{batch_norm}'
        config = {'vision': vision, 'hidden_dims': hidden_dims, 'dropout_prob': dropout_prob, 'batch_norm': batch_norm, 'lr': lr, 'weight_decay': weight_decay, 'num_epochs': num_epochs, 'seed': seed}

        model = JointEncoder(vision=vision, hidden_dims=hidden_dims, dropout_prob=dropout_prob, batch_norm=batch_norm, mask_branch=mask_branch)

        freeze_vision_encoder_layers(model, vision)
        
        if do_train:
            print(f'W&B initialization:\trun {run_name}')
            wandb.init(project='energyefficiency', entity='silvy-romanato', name=f'{run_name}', config=config)
            wandb.config.update({'vision': vision, 'hidden_dims': hidden_dims, 'dropout_prob': dropout_prob, 'batch_norm': batch_norm, 'lr': lr, 'weight_decay': weight_decay, 'num_epochs': num_epochs, 'seed': seed})

            train_model(model, 
                        train_data, 
                        val_data, 
                        lr, 
                        weight_decay, 
                        num_epochs, 
                        seed, 
                        run_name)
            
            wandb.finish()

        # Evaluate model
        if do_eval:
            evaluate_model(model, 
                        train_data, 
                        val_data, 
                        test_data, 
                        lr, 
                        weight_decay, 
                        num_epochs, 
                        seed, 
                        do_train, 
                        checkpoint_path, 
                        run_name,
                        #property_data
                        )
        
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision', type=str, default='efficientnet_b0')
    parser.add_argument('--hidden_dims', type=parse_nested_list_of_ints, default=[[256]], help='Hidden dimensions for the MLP.') # input like: '512-256-124,512-256'   # try only one layer
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--batch_norm', action='store_true', default=False) # 0.3, 0.4 # try 0.2
    parser.add_argument('--lr', type=parse_list_of_floats, default=[1e-4])  # 1e-3, 1e-4, 1e-5, 1e-6
    parser.add_argument('--weight_decay', type=float, default=1e-2) # 1e-4 # try 1e-2
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--do_eval', action='store_true', help="Enable evaluation mode")
    parser.add_argument('--no_eval', action='store_false', dest='do_eval', help="Disable evaluation mode")
    parser.add_argument('--do_train', action='store_true', help="Enable training mode")
    parser.add_argument('--no_train', action='store_false', dest='do_train', help="Disable training mode")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--mask_branch', type=mask_branch_type, default=[], help='Which layers to turn off in the vision encoder.')
    parser.add_argument('--reduce_dataset', action='store_true', help='Reduce the dataset size for testing purposes.')
    parser.add_argument('--cv', type=str, default='1', help='Number of cross-validation folds.')

    args = parser.parse_args()

    print(f'Cuda is available: {torch.cuda.is_available()}')

    image_data = prepare_data(args.reduce_dataset, cv = args.cv)
    train_data, val_data, test_data = load_data(image_data, vision=args.vision)

    del args.reduce_dataset
    del args.cv
    grid_search(train_data, val_data, test_data, **vars(args))
    
    