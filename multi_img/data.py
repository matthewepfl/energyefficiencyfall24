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

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
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

workingOn = 'laptop' # 'server' or 'laptop
# ---------------------------------------- GLOBAL VARIABLES ---------------------------------------- #

# Global configurations
if workingOn == 'server':
    BASE_DIR = '/work/FAC/HEC/DEEP/shoude/ml_green_building/'
else:
    BASE_DIR = '/Users/silviaromanato/Desktop/EPFL/MA4/EnergyEfficiencyPrediction/multi_img/'
IMAGES_PATH = os.path.join(BASE_DIR, 'images_full_data/MediaSyncFolder/')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'modelS/images_models')
PATH_NUM_APPT = os.path.join(DATA_DIR, 'vacancy_rate.px')
IMAGES_DF_PATH = os.path.join(DATA_DIR, 'images_df.csv')
LISTINGS_PATH = os.path.join(DATA_DIR, 'listings_full.pkl')
INQUIRIES_PATH = os.path.join(DATA_DIR, 'inquiries_full.pkl')
LABELS_TRAIN_PATH = os.path.join(DATA_DIR, 'labels_train.csv')
LABELS_VAL_PATH = os.path.join(DATA_DIR, 'labels_val.csv')
LABELS_TEST_PATH = os.path.join(DATA_DIR, 'labels_test.csv')
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed_data')
CLUSTERED_PATH = os.path.join(DATA_DIR, 'clustered_images_with6classes.csv')

# ---------------------------------------- HELPER FUNCTIONS ---------------------------------------- #

def Demand(listings, inquiries):
    """
    Calculate the demand for each listing
        input: listings, inquiries
        output: listings with demand
    """

    # Filter the listings with only one advertisement version
    listings = listings[listings["Advertisement Version Id"] == 1]

    # Create a Column for the Time Left and Time Passed
    inquiries.loc[:,"Time Left"] = inquiries.loc[:,"Created"].max() - inquiries.loc[:,"Advertisement Created"]
    inquiries.loc[:, "Time Passed"] = inquiries.loc[:,"Created"] - inquiries.loc[:,"Advertisement Created"]

    # Filter inquiries
    inquiries = inquiries[(inquiries.loc[:,"Time Left"] >= pd.Timedelta("30 days")) &
                        (inquiries.loc[:,"Time Passed"] <= pd.Timedelta("30 days"))]

    # Calculate the demand from the inquiries
    Demand = inquiries.groupby(["Advertisement Id"]).size()
    Demand.name = "Demand"

    # Merge the demand with the listings
    listings = listings.merge(Demand, left_on=["Advertisement Id"],
                            right_index=True, how="left",
                            indicator="_merge")

    # Fill the missing values with 0
    listings.loc[listings._merge == "left_only", "Demand"] = 0

    # Drop the columns which are not needed
    listings = listings.drop(columns=["Form Lead Unique", "_merge"])
    return listings

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

def load_demand():
    # Load the dataframes
    images_df = pd.read_csv(IMAGES_DF_PATH)                               # The pathnames and the Property Reference Ids of the images
    listings = pd.read_pickle(LISTINGS_PATH)                              # The listings to create the Demand
    inquiries = pd.read_pickle(INQUIRIES_PATH)                            # The inquiries to create the Demand
    
    # Calculate the Demand per listing
    listings_demand = Demand(listings, inquiries)
    listings_demand = listings_demand[~listings_demand['Property Reference Id'].duplicated(keep='last')]

    # Merge Demand in listings and the images_df on the Property Reference Id
    images_df = images_df[['pathname', 'Property Reference Id']].merge(listings_demand[['Property Reference Id', 'Demand']], on = 'Property Reference Id', how = 'inner')
    images_df = images_df[images_df['Demand'].notna()]
    images_df.drop(columns = ['pathname'], inplace = True)

    return images_df

def load_images_data(cluster_data):
    '''
    Load image data: labels, image files, image metadata
    '''
    if not os.path.exists(IMAGES_PATH):
        raise ValueError(f'Images folder not found in {IMAGES_PATH}.')
    
    labels_data = load_demand()
    image_files = list_images(IMAGES_PATH)
    
    properties = cluster_data['Property Reference Id'].unique()
    image_files = [image_file for image_file in image_files if image_file.split(os.sep)[-1][:-5] in properties]

    labels_data = labels_data[labels_data['Property Reference Id'].isin([image_file.split(os.sep)[-1][:-5] for image_file in image_files])]
    labels_data = labels_data.merge(cluster_data[['Property Reference Id', 'cluster']], on = 'Property Reference Id', how = 'inner')

    print(f'Number of samples:\tLabels: {len(labels_data)}\tImage: {len(image_files)}')
    print("the labels look like this: ", labels_data.head(40))
    print("the image files look like this: ", image_files[:10])

    if image_files == []:
        raise ValueError(f'No image files found in {IMAGES_PATH}.')
    
    return labels_data, image_files

def create_image_labels_mapping(image_files, labels_data):
    '''
    Create a mapping from image files to their corresponding labels and view positions.

    Parameters:
    - image_files: List of image file paths
    - labels_data: DataFrame containing label information

    Returns:
    A dictionary with image file paths as keys and dicts with labels and ViewPosition as values.
    '''
    image_labels_mapping = {}

    for image_path in tqdm(image_files):
        # Extract subject_id, study_id, and dicom_id from the file path
        parts = image_path.split(os.sep)
        property_id = parts[-1][:-5]
        number = parts[-1][-5]

        # Find the corresponding row in the labels CSV
        labels_row = labels_data[(labels_data['Property Reference Id'] == str(property_id))]
        
        if not labels_row.empty:
            labels = labels_row.iloc[0].to_dict()
            labels['number'] = number
            image_labels_mapping[image_path] = labels  

    return image_labels_mapping
        
def join_multi(labels_data, image_files):
    '''
    Join the image data.
    Returns: 
        dict_img (dict): keys = image file paths and values = dicts with labels and ViewPosition
    '''
    print('Join multi input data')

    # Image data
    image_labels_mapping = create_image_labels_mapping(image_files, labels_data)
    df_img = pd.DataFrame.from_dict(image_labels_mapping, orient='index').reset_index()
    df_img['Property Reference Id'] = df_img['Property Reference Id'].astype(str)
    df_img['number'] = df_img['number'].astype(str)

    # Keep only PA and LATERAL images
    df_img = df_img[df_img['number'].isin(['a', 'b'])]

    # Group by study_id and subject_id and ViewPosition and keep the first row
    df_img = df_img.groupby(['Property Reference Id', 'number']).first().reset_index()

    # Function to check if both PA and Lateral images are present
    def has_both_views(group):
        return 'a' in group['number'].values and 'b' in group['number'].values

    # Filter the DataFrame
    df_img = df_img.groupby(['Property Reference Id']).filter(has_both_views)

    print(f'Number of samples:\tImage: {len(df_img)}')

    # Return the image data to a dictionary
    dict_img = df_img.set_index('index').T.to_dict()

    return dict_img
    
# ---------------------------------------- PREPROCESSING ---------------------------------------- #

def split(labels, val_size=0.1, test_size=0.15, seed=42):
    '''
    Split tabular data and labels into train, val, and test sets.
    '''
    paths = [LABELS_TRAIN_PATH, LABELS_VAL_PATH, LABELS_TEST_PATH]
    
    if all([os.path.exists(path) for path in paths]):
        print('Splitting:\tLoading pre-processed train, val, and test sets.')
        labels_train = pd.read_csv(LABELS_TRAIN_PATH)
        labels_val = pd.read_csv(LABELS_VAL_PATH)
        labels_test = pd.read_csv(LABELS_TEST_PATH)

    else:
        print('Splitting:\tTabular data and labels into train, val, and test sets.')
        # Split the study_ids into train, val, and test sets
        property_id = labels['Property Reference Id'].unique()
        np.random.seed(seed)
        np.random.shuffle(property_id)
        num_property_ids = len(property_id)
        num_val = int(num_property_ids * val_size)
        num_test = int(num_property_ids * test_size)
        study_ids_train = property_id[num_val + num_test:]
        study_ids_val = property_id[:num_val]
        study_ids_test = property_id[num_val:num_val + num_test]

        # Get the tabular data and labels for the train, val, and test sets
        labels_train = labels[labels['Property Reference Id'].isin(study_ids_train)]
        labels_val = labels[labels['Property Reference Id'].isin(study_ids_val)]
        labels_test = labels[labels['Property Reference Id'].isin(study_ids_test)]

        # Save the train, val, and test sets
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
        transforms.append(RandomRotation(10))
        transforms.append(RandomVerticalFlip())
        transforms.append(RandomHorizontalFlip())

    transforms.append(CenterCrop((size, size)))

    if vision == 'vit':
        processor = ViTImageProcessor.from_pretrained(
            'google/vit-large-patch32-384', 
            do_normalize=False, 
            image_mean=NORM_MEAN, 
            image_std=NORM_STD, 
            return_tensors='pt')
        transforms.append(lambda x: processor(x).pixel_values[0])
    else: 
        transforms.append(Resize((IMAGE_SIZE, IMAGE_SIZE)))
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=NORM_MEAN, std=NORM_STD))
    return Compose(transforms)

class MultimodalDataset(Dataset):
    '''
    Dataset class for MIMIC-CXR and MIMIC-IV.
    Handles both tabular data and images.
    '''
    def __init__(self, vision, data_dict, augment=True):
        self.vision = vision
        self.data_dict = data_dict

        if vision is not None: 
            self.transform = lambda img_size: transform_image(img_size, vision=vision, augment=augment)
        
        # Organize paths by subject_id and study_id
        self.organized_paths = self._organize_paths()

        # Filter out pairs where both images are None
        self.organized_paths = {k: v for k, v in self.organized_paths.items() if v['a'] is not None and v['b'] is not None}

    def _organize_paths(self):
        organized = {}
        for path in self.data_dict.keys():
            parts = path.split(os.sep)
            property_id = parts[-1][:-5]
            number = parts[-1][-5]
            key = (property_id)
            if key not in organized:
                organized[property_id] = {'a': None, 'b': None}
            if number in ['a', 'b']:
                organized[property_id][number] = path

        print('The shape of the organized paths:', organized)
        return organized

    def __len__(self):
        return len(self.organized_paths)

    def _load_and_process_image(self, path):
        if path:
            image = Image.open(path).convert('RGB')
        else:
            image = Image.new('RGB', (self.size, self.size))
        image = self.transform(image.size)(image)
        return image

    def __getitem__(self, idx):

        if idx >= len(self.organized_paths):
            raise IndexError(f"Index {idx} out of range. Dataset has {len(self.organized_paths)} samples.")
        
        # Get the subject_id and study_id for this index
        property_number_pair = list(self.organized_paths.keys())[idx]

        # Get the paths for the PA and Lateral images
        a_path = self.organized_paths[property_number_pair]['a']
        b_path = self.organized_paths[property_number_pair]['b']

        # Get labels from the image data
        labels_path = a_path if a_path else b_path
        if not labels_path:
            raise ValueError(f'No labels path found for {property_number_pair}.')
        labels = self.data_dict[labels_path]['Demand']
        label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
        if torch.any(label_tensor < 0):
            print(f'Negative label values for {property_number_pair}: {label_tensor}')
        
        inputs = {'labels': label_tensor}
        
        # Load and process PA and Lateral images
        if self.vision is not None:
            a_image = self._load_and_process_image(a_path) \
                if a_path else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            b_image = self._load_and_process_image(b_path) \
                if b_path else torch.zeros((3, self.size, self.size), dtype=torch.float32)
            if not isinstance(a_image, torch.Tensor):
                a_image = torch.tensor(a_image)
            if not isinstance(b_image, torch.Tensor):
                b_image = torch.tensor(b_image)
            inputs['x_a'] = a_image
            inputs['x_b'] = b_image
        return inputs

    def collate_fn(self, batch):
        inputs = {}
        
        if 'labels' in batch[0]:
            inputs['labels'] = torch.stack([x['labels'] for x in batch if 'labels' in x])

        if self.vision is not None:
            if 'x_a' in batch[0]:
                inputs['x_a'] = torch.stack([x['x_a'] for x in batch if 'x_a' in x])

            if 'x_b' in batch[0]:
                inputs['x_b'] = torch.stack([x['x_b'] for x in batch if 'x_b' in x])
        return inputs


# ---------------------------------------- MAIN FUNCTIONS ---------------------------------------- #
    
def prepare_data(): 
    '''
    Load and pre-process tabular data and labels.
    Split into train/val/test sets.
    Filter images based on tabular data.
    '''
    print(f'PREPARING DATA')
    
    # Load image labels, files and metadata
    print('Loading:\tImage data (labels, files).')
    cluster_data = pd.read_csv(CLUSTERED_PATH)
    print('Cluster data shape:', cluster_data.shape, 'Columns:', cluster_data.columns)
    labels_data, image_files = load_images_data(cluster_data)

    # Split labels into train/val/test sets
    print('Splitting:\tLabels into train/val/test sets.')
    lab_train, lab_val, lab_test = split(labels_data, val_size=0.1, test_size=0.15, seed=42)

    print('Joining:\tIntersection of tabular and image data.')
    image_data_train = join_multi(lab_train, image_files)
    image_data_val = join_multi(lab_val, image_files)
    image_data_test = join_multi(lab_test, image_files)
    image_data = {'train': image_data_train, 'val': image_data_val, 'test': image_data_test}
        
    return image_data

def load_data(image_data, vision=None):
    '''
    Create datasets for each split.
    Arguments: 
        image_data (dict): Dictionary with keys = 'train', 'val', 'test' and values = image data
        vision (str): Type of vision encoder 'resnet50', 'densenet121' or 'vit' (Default: None --> No images)
    '''
    print(f'LOADING DATA (vision: {vision})')
    print(f'Loaded image data:\tTrain: {len(image_data["train"])}\tValidation: {len(image_data["val"])}\tTest: {len(image_data["test"])} samples.')
    train_data = MultimodalDataset(vision, image_data['train'], augment=True)
    val_data = MultimodalDataset(vision, image_data['val'], augment=False)
    test_data = MultimodalDataset(vision, image_data['test'], augment=False)
    print(f'Created datasets:\tTrain: {len(train_data)}\tValidation: {len(val_data)}\tTest: {len(test_data)} samples.')
    return train_data, val_data, test_data

if __name__ == '__main__': 

    image_data = prepare_data()

    image_data_train, image_data_val, image_data_test = image_data['train'], image_data['val'], image_data['test']

    # Print the shapes of the dataframes
    print(f'Image data\nTrain: {len(image_data_train)}\nVal: {len(image_data_val)}\nTest: {len(image_data_test)}')

    # Save the dictionaries
    # np.save(os.path.join(PROCESSED_PATH, 'image_data_train.npy'), image_data_train)
    # np.save(os.path.join(PROCESSED_PATH, 'image_data_val.npy'), image_data_val)
    # np.save(os.path.join(PROCESSED_PATH, 'image_data_test.npy'), image_data_test)

    # Delete not matched images
    all_images = set(list(image_data_train.keys()) + list(image_data_val.keys()) + list(image_data_test.keys()))
    _, image_files = load_images_data()
    for image_file in image_files:
        if image_file not in all_images:
            os.remove(image_file)
    
    print('Finished!')

