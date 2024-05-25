import numpy as np
import pandas as pd

images_df = pd.read_csv('/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/Clusters_images/clustered_images.csv')
images_df = images_df.groupby(['Property Reference Id', 'cluster']).first().reset_index().drop(columns = ['Unnamed: 0', 'letter'])
print(images_df.head())

image_counts = images_df.groupby('Property Reference Id').size()

# Choose only the ones with 6 clusters.
number = 6
valid_property_ids = image_counts[image_counts >= number].index
images_df = images_df[images_df['Property Reference Id'].isin(valid_property_ids)]

# Save.
images_df.to_csv(f'/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/Clusters_images/clustered_images_with{number}classes.csv')
print(f'The images were saved, from 211082 we result having {images_df.shape[0]} properly classified images. \n')