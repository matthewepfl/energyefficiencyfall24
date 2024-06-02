'''
Grid search for demand calculation using joint image encoders. 
'''

import os
import argparse
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import EarlyStoppingCallback
from models import *
from data import *

import logging
logging.basicConfig(level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)  # Replace UserWarning with the specific category if known

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


workingOn = 'server'
# ---------------------------------------- GLOBAL VARIABLES ---------------------------------------- #

# Global configurations
if workingOn == 'server':
    BASE_DIR = '/work/FAC/HEC/DEEP/shoude/ml_green_building/'
else:
    BASE_DIR = '/Users/silviaromanato/Desktop/EPFL/MA4/EnergyEfficiencyPrediction/multi_img/'
    
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

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

def create_trainer(model, train_data, val_data, output_dir, epochs=10, lr=1e-5, batch_size=16, weight_decay=1e-2, seed=0):
    '''
    Trains a Joint Encoder model. 
    W&B logging is enabled by default.

    Arguments:
        - model (JointEncoder): Joint encoder model
        - train_data (torch.utils.data.Dataset): Training data
        - val_data (torch.utils.data.Dataset): Validation data
        - output_dir (str): Path to directory where checkpoints will be saved
        - epochs (int): Number of epochs
        - lr (float): Learning rate
        - batch_size (int): Batch size
        - weight_decay (float): Weight decay
        - seed (int): Random seed
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    params = [{'params': model.regression.parameters(), 
                'lr': 0.001, 'weight_decay': 0.01}]
    if model.vision:
        params.append({'params': model.vision_encoder.parameters()}) 
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=len(train_data)*epochs)

    training_args = TrainingArguments(

        # Training
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.999,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0, 
        seed=seed,

        # Evaluation & checkpointing
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
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
        data_collator=train_data.collate_fn,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
        optimizers=(optimizer, scheduler)
    )
    return trainer

def grid_search(vision=None, 
                hidden_dims=[256, 512],
                dropout_prob=0.0,
                batch_norm=False,
                lr=0.001, 
                weight_decay=0.01,
                num_epochs=10,
                seed=0,
                eval=False
                ):
    '''
    Grid search for radiology diagnosis using joint image encoders. 
    '''
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = JointEncoder(
        vision=vision, 
    )

    # Freeze layers of vision encoder
    if vision:
       for param in model.vision_encoder.parameters():
           param.requires_grad = False

    # Load data
    print('Data:\tLoading data')
    image_data = prepare_data() # image data don't have all the clusters
    train_data, val_data, test_data = load_data(image_data, vision=vision)

    # Train model
    trainer = create_trainer(model, train_data, val_data, CHECKPOINTS_DIR, 
                             epochs=num_epochs, lr=lr, batch_size = 8, 
                            #  hidden_dims=hidden_dims, dropout_prob=dropout_prob,
                            # batch_norm=batch_norm,
                             weight_decay=weight_decay, seed=seed)
    print('Training:\tStarting training')
    trainer.train()

    # Evaluate model
    if eval:
        eval_results = trainer.evaluate(eval_dataset=test_data)

        print('Evaluation:\tResults')
        print(eval_results)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision', type=str, default='resnet50')
    parser.add_argument('--hidden_dims', type=str, default=[256, 512])
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()

    if args.hidden_dims and type(args.hidden_dims) == str:
        args.hidden_dims = [int(x) for x in args.hidden_dims.split('-')]

    print(f'Cuda is available: {torch.cuda.is_available()}')

    grid_search(**vars(args))
    
    