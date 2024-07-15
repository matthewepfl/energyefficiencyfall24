'''
Data loading and preprocessing for MIMIC-CXR and MIMIC-IV. 
Multimodal data loader and dataset classes. 
Train/val/test splitting. 
'''

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose
import albumentations as A

from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

IMAGE_SIZE = 384                        # All images are resized to 384 x 384
NORM_MEAN = [0.4734, 0.4734, 0.4734]    # MIMIC-CXR mean (based on 2GB of images) # compute it
NORM_STD = [0.3006, 0.3006, 0.3006]     # MIMIC-CXR std (based on 2GB of images) # compute it

"""
loss_selected = nn.MSELoss()
num_images = 3
epochs = 45
batch_size = 16
initial_learning_rate = 0.01
show_img = True
IMAGE_EMBEDDING_DIM = 512
selected_model = 'resnet'
dropout_rate = 0.4
"""

workingOn = 'erver' # 'server' or 'laptop
minim_amount_classes = 2
# ---------------------------------------- GLOBAL VARIABLES ---------------------------------------- #

# Global configurations
if workingOn == 'server':
    BASE_DIR = '/work/FAC/HEC/DEEP/shoude/ml_green_building/'
else:
    BASE_DIR = '/Users/silviaromanato/Desktop/'
IMAGES_PATH = os.path.join(BASE_DIR, 'images_full_data/MediaSyncFolder/')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'modelS/images_models')
PATH_NUM_APPT = os.path.join(DATA_DIR, 'vacancy_rate.px')
IMAGES_DF_PATH = os.path.join(DATA_DIR, 'images_df.csv')
LISTINGS_PATH = os.path.join(DATA_DIR, 'Listings_FE.pkl')
INQUIRIES_PATH = os.path.join(DATA_DIR, 'inquiries_full.pkl')
ENERGY_PATH = os.path.join(DATA_DIR, 'Listings_FE.csv')
LABELS_TRAIN_PATH = os.path.join(DATA_DIR, f'train_data_properties{minim_amount_classes}.csv')
LABELS_VAL_PATH = os.path.join(DATA_DIR, f'val_data_properties{minim_amount_classes}.csv')
LABELS_TEST_PATH = os.path.join(DATA_DIR, f'test_data_properties{minim_amount_classes}.csv')
CLUSTERED_PATH = os.path.join(DATA_DIR, f'Clusters_images/clean_clustered_images_greater{minim_amount_classes}.csv')

# ---------------------------------------- HELPER FUNCTIONS ---------------------------------------- #

def Efficiency(efficiency):

    efficiency = efficiency.groupby(["Advertisement Id"]).last().reset_index()
    efficiency = efficiency.groupby(["Property Reference Id", "PropertyFE"]).size().reset_index(name='counts').drop(columns="counts") # change this too # use more
    print('The efficiency: ', efficiency.shape)

    return efficiency

def list_images(base_path):
    '''
    Recursively lists all image files starting from the base path.
    Assumes that images have extensions typical for image files (e.g., .jpg, .jpeg, .png).
    '''
    image_files = []
    for subdir, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(subdir, file))
    return image_files

def load_efficiency():
    # Load the dataframes
    images_df = pd.read_csv(IMAGES_DF_PATH)                               # The pathnames and the Property Reference Ids of the images
    efficiency = pd.read_csv(ENERGY_PATH)                                 # The energy data to create the Efficiency
    
    # Calculate the Efficiency per listing
    efficiency = Efficiency(efficiency)

    # Merge Demand in listings and the images_df on the Property Reference Id
    images_df = images_df[['pathname', 'Property Reference Id']].merge(efficiency[['Property Reference Id', 'PropertyFE']], on = 'Property Reference Id', how = 'inner')
    images_df = images_df[images_df['PropertyFE'].notna()]
    images_df.drop(columns = ['pathname'], inplace = True)

    return images_df

def load_images_data(cluster_data):
    '''
    Load image data: labels, image files, image metadata
    '''
    print(IMAGES_PATH)
    if not os.path.exists(IMAGES_PATH):
        raise ValueError(f'Images folder not found in {IMAGES_PATH}.')
    
    labels_data = load_efficiency()
    image_files = list_images(IMAGES_PATH)
    
    properties = cluster_data['Property Reference Id'].unique()
    image_files = [image_file for image_file in image_files if image_file.split(os.sep)[-1][:-5] in properties]

    labels_data = labels_data[labels_data['Property Reference Id'].isin([image_file.split(os.sep)[-1][:-5] for image_file in image_files])]
    labels_data = labels_data.merge(cluster_data[['Property Reference Id', 'cluster', 'pathname']], on = 'Property Reference Id', how = 'inner')
    labels_data = labels_data.drop_duplicates(subset = ['Property Reference Id', 'cluster', 'PropertyFE', 'pathname'])
    image_files = [image_file for image_file in image_files if image_file.split(os.sep)[-1][:-5] in labels_data['Property Reference Id'].unique()]

    print(f'Number of samples:\tLabels: {len(labels_data)}\tImage: {len(image_files)}')

    # CHECK 
    images_df = labels_data.groupby(['Property Reference Id', 'cluster']).first().reset_index()
    image_counts = images_df.groupby('Property Reference Id').size()
    print("There are clusters for :", image_counts.groupby(image_counts).size())

    if image_files == []:
        raise ValueError(f'No image files found in {IMAGES_PATH}.')
    
    return labels_data

def create_image_labels_mapping(labels_data):
    '''
    Create a mapping from image files to their corresponding labels and view positions.

    Parameters:
    - labels_data: DataFrame containing label information

    Returns:
    A dictionary with image file paths as keys and dicts with labels and ViewPosition as values.
    '''
    image_labels_mapping = {}
    for property in tqdm(labels_data['Property Reference Id'].unique()):
        labels = labels_data[labels_data['Property Reference Id'] == property]
        propertyFE = labels['PropertyFE'].values[0]
        for classes in [0, 1, 2, 3, 4, 5]:
            if classes not in labels['cluster'].values:
                path = BASE_DIR + 'images_full_data/black.png'
                labels_out = {'Property Reference Id': property, 'PropertyFE': propertyFE, 'cluster': classes, 'pathname': path}
                unique_index = (property , classes)
                image_labels_mapping[unique_index] = labels_out
            else:
                labels_row = labels[labels['cluster'] == classes]
                labels_out = labels_row.iloc[0].to_dict()
                unique_index = (property , classes)
                image_labels_mapping[unique_index] = labels_out

    return image_labels_mapping
        
def join_multi(labels_data):
    '''
    Join the image data.
    Returns: 
        dict_img (dict): keys = image file paths and values = dicts with labels and ViewPosition
    '''
    print('Join multi input data')

    # Image data
    image_labels_mapping = create_image_labels_mapping(labels_data) 
    df_img = pd.DataFrame.from_dict(image_labels_mapping, orient='index')
    df_img['Property Reference Id'] = df_img['Property Reference Id'].astype(str)
    df_img['cluster'] = df_img['cluster'].astype(str)

    # Create a dictionary
    dict_img = df_img.T.to_dict()

    return dict_img
    
# ---------------------------------------- PREPROCESSING ---------------------------------------- #

# correct
def split(labels, val_size=0.15, test_size=0.20, seed=42):
    '''
    Split tabular data and labels into train, val, and test sets.
    '''
    paths = [LABELS_TRAIN_PATH, LABELS_VAL_PATH, LABELS_TEST_PATH]
    
    if False:#all([os.path.exists(path) for path in paths]):
        print('Splitting:\tLOADING pre-processed train, val, and test sets.')
        labels_train = pd.read_csv(LABELS_TRAIN_PATH)
        labels_val = pd.read_csv(LABELS_VAL_PATH)
        labels_test = pd.read_csv(LABELS_TEST_PATH)

    else:
        print('Splitting:\tTabular data and labels into train, val, and test sets.')

        # Split the study_ids into train, val, and test sets
        property__ref_id = labels['Property Reference Id'].unique()
        property_id = [int(i.split('.')[0]) for i in property__ref_id]
        property_id = list(set(property_id))
        np.random.seed(seed)
        property_id = np.random.permutation(property_id)
        num_property_ids = len(property_id)
        num_val = int(num_property_ids * val_size)
        num_test = int(num_property_ids * test_size)
        study_ids_train = property_id[num_val + num_test:]
        study_ids_val = property_id[:num_val]
        study_ids_test = property_id[num_val:num_val + num_test]

        # Get the tabular data and labels for the train, val, and test sets
        labels_train = labels[labels['Property Reference Id'].str.split('.').str[0].astype(int).isin(study_ids_train)]
        labels_val = labels[labels['Property Reference Id'].str.split('.').str[0].astype(int).isin(study_ids_val)]
        labels_test = labels[labels['Property Reference Id'].str.split('.').str[0].astype(int).isin(study_ids_test)]

        print('Splitting:\tSaving train, val, and test sets.')
        labels_train.to_csv(LABELS_TRAIN_PATH, index=False)
        labels_val.to_csv(LABELS_VAL_PATH, index=False)
        labels_test.to_csv(LABELS_TEST_PATH, index=False)

        # Check proportions of total, train, val, and test sets
        total_len = len(labels_train) + len(labels_val) + len(labels_test)
        print('Total set: ', total_len)
        print('Percent train: ', len(labels_train) / total_len)
        print('Percent val: ', len(labels_val) / total_len)
        print('Percent test: ', len(labels_test) / total_len)

    return labels_train, labels_val, labels_test

# ---------------------------------------- DATA LOADING ---------------------------------------- #

def transform_image(image_size, vision=None, augment=True): 
    '''
    Defines the image transformation pipeline. 
    1. Augmentation (flips, rotations) (only for training)
    2. Crop a square from the (non-square) image
    3. Resize to IMAGE_SIZE x IMAGE_SIZE
    4. Convert to tensor
    5. Normalize (with ImageNet mean and std)
    '''
    transforms = []
    size = min(image_size) # Get minimum of image height and width to crop to square

    # Augmentation (flips, rotations)
    if augment:
        transforms.append(RandomHorizontalFlip())
        transforms.append(RandomVerticalFlip())
        transforms.append(RandomRotation(degrees=10))

    transforms.append(CenterCrop((size, size)))
    transforms.append(Resize((IMAGE_SIZE, IMAGE_SIZE)))

    if vision == 'vit':
        processor = ViTImageProcessor.from_pretrained(
            'google/vit-large-patch32-384', 
            do_normalize=False, 
            image_mean=NORM_MEAN, 
            image_std=NORM_STD, 
            return_tensors='pt',
        )
        transforms.append(lambda x: processor(x).pixel_values[0])
    else: 
        transforms.append(Resize((IMAGE_SIZE, IMAGE_SIZE)))
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=NORM_MEAN, std=NORM_STD))
    return Compose(transforms)

class MultimodalDataset(Dataset):

    def __init__(self, vision, data_dict, augment=True):
        self.vision = vision
        self.data_dict = data_dict

        if vision is not None: 
            self.transform = lambda img_size: transform_image(img_size, vision=vision, augment=augment)
        
        self.properties = [x[0] for x in list(self.data_dict.keys())]

    def __len__(self):
        return len(self.data_dict) // 6

    def _load_and_process_image(self, path):
        if path:
            image = Image.open(path).convert('RGB')
        else:
            image = Image.new('RGB', (self.size, self.size))
        image = self.transform(image.size)(image)
        return image

    def __getitem__(self, idx):

        if idx >= len(self.data_dict):
            raise IndexError(f"Index {idx} out of range. Dataset has {len(self.data_dict)} samples.")
        
        # Get the subject_id and study_id for this index
        property_cluster_pair = list(self.data_dict.keys())[idx]
        property, cluster = property_cluster_pair

        # Get the paths for the PA and Lateral images
        path_0 = self.data_dict[(property, 0)]['pathname']
        path_1 = self.data_dict[(property, 1)]['pathname']
        path_2 = self.data_dict[(property, 2)]['pathname']
        path_3 = self.data_dict[(property, 3)]['pathname']
        path_4 = self.data_dict[(property, 4)]['pathname']
        path_5 = self.data_dict[(property, 5)]['pathname']

        # Get the labels
        labels = self.data_dict[(property, cluster)]['PropertyFE']
        label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
        
        inputs = {'labels': label_tensor}
        
        # Load and process PA and Lateral images
        if self.vision is not None:
            image_0 = self._load_and_process_image(path_0) if path_0 else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            image_1 = self._load_and_process_image(path_1) if path_1 else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            image_2 = self._load_and_process_image(path_2) if path_2 else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            image_3 = self._load_and_process_image(path_3) if path_3 else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            image_4 = self._load_and_process_image(path_4) if path_4 else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            image_5 = self._load_and_process_image(path_5) if path_5 else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            
            image_0 = torch.tensor(image_0) if not isinstance(image_0, torch.Tensor) else image_0
            image_1 = torch.tensor(image_1) if not isinstance(image_1, torch.Tensor) else image_1
            image_2 = torch.tensor(image_2) if not isinstance(image_2, torch.Tensor) else image_2
            image_3 = torch.tensor(image_3) if not isinstance(image_3, torch.Tensor) else image_3
            image_4 = torch.tensor(image_4) if not isinstance(image_4, torch.Tensor) else image_4
            image_5 = torch.tensor(image_5) if not isinstance(image_5, torch.Tensor) else image_5

            inputs['x_0'] = image_0
            inputs['x_1'] = image_1
            inputs['x_2'] = image_2
            inputs['x_3'] = image_3
            inputs['x_4'] = image_4
            inputs['x_5'] = image_5

        return inputs
    
    def get_labels(self):
        labels = []
        for idx in range(len(self)):
            labels.append(self.__getitem__(idx)['labels'])
        return torch.stack(labels)
    
    def count_black_images(self):
        mean = 0
        for idx in range(len(self)):
            count = 0
            inputs = self.__getitem__(idx)
            if inputs['x_0'].sum() == 0:
                count += 1
            if inputs['x_1'].sum() == 0:
                count += 1
            if inputs['x_2'].sum() == 0:
                count += 1
            if inputs['x_3'].sum() == 0:
                count += 1
            if inputs['x_4'].sum() == 0:
                count += 1
            if inputs['x_5'].sum() == 0:
                count += 1
            
            mean += count

        return mean / len(self)

    def collate_fn(self, batch):
        inputs = {}
        
        if 'labels' in batch[0]:
            inputs['labels'] = torch.stack([x['labels'] for x in batch if 'labels' in x])

        if self.vision is not None:
            if 'x_0' in batch[0]:
                inputs['x_0'] = torch.stack([x['x_0'] for x in batch if 'x_0' in x])

            if 'x_1' in batch[0]:
                inputs['x_1'] = torch.stack([x['x_1'] for x in batch if 'x_1' in x])

            if 'x_2' in batch[0]:
                inputs['x_2'] = torch.stack([x['x_2'] for x in batch if 'x_2' in x])

            if 'x_3' in batch[0]:
                inputs['x_3'] = torch.stack([x['x_3'] for x in batch if 'x_3' in x])

            if 'x_4' in batch[0]:
                inputs['x_4'] = torch.stack([x['x_4'] for x in batch if 'x_4' in x])

            if 'x_5' in batch[0]:
                inputs['x_5'] = torch.stack([x['x_5'] for x in batch if 'x_5' in x])

            
        return inputs

# ---------------------------------------- MAIN FUNCTIONS ---------------------------------------- #
    
def prepare_data(reduce): 
    '''
    Load and pre-process tabular data and labels.
    Split into train/val/test sets.
    Filter images based on tabular data.
    '''
    print(f'\tPREPARING DATA\n')
    
    # Load image labels, files and metadata
    cluster_data = pd.read_csv(CLUSTERED_PATH)
    cluster_data = reduce_dataset(cluster_data) if reduce else cluster_data

    data = load_images_data(cluster_data)

    # Split labels into train/val/test sets
    lab_train, lab_val, lab_test = split(data, val_size=0.1, test_size=0.15, seed=42)
    print(f'Split data into train/val/test sets:\nTrain: {len(lab_train)}\nValidation: {len(lab_val)}\nTest: {len(lab_test)}')

    image_data_test = join_multi(lab_test)
    image_data_val = join_multi(lab_val)
    image_data_train = join_multi(lab_train) 
    
    image_data = {'train': image_data_train, 'val': image_data_val, 'test': image_data_test}
    print(f'Loaded image data:\nTrain: {len(image_data_train)}\nValidation: {len(image_data_val)}\nTest: {len(image_data_test)}')
        
    return image_data

def reduce_dataset(data):
    print('*' * 20, 'Reducing dataset size', '*' * 20)
    print("the length of the cluster_data is: ", len(data))
    properties = data.groupby('Property Reference Id').first().reset_index()['Property Reference Id']
    properties = properties.sample(frac=0.01, random_state=42)
    data = data[data['Property Reference Id'].isin(properties)]
    print("the length of the cluster_data is: ", len(data))
    print('Reduced dataset size')
    return data
    
def load_data(image_data, vision=None):

    print(f'LOADING DATA (vision: {vision})')

    train_data = MultimodalDataset(vision, image_data['train'], augment=True)
    val_data = MultimodalDataset(vision, image_data['val'], augment=False)
    test_data = MultimodalDataset(vision, image_data['test'], augment=False)
    print(f'Created datasets:\tTrain: {len(train_data)}\tValidation: {len(val_data)}\tTest: {len(test_data)} samples.')

    train_black_images = train_data.count_black_images()
    val_black_images = val_data.count_black_images()
    test_black_images = test_data.count_black_images()

    print(f'Black images in train: {train_black_images}\nBlack images in val: {val_black_images}\nBlack images in test: {test_black_images}')

    return train_data, val_data, test_data

if __name__ == '__main__': 

    image_data = prepare_data(False)
    train_data, val_data, test_data = load_data(image_data, vision='vit')




