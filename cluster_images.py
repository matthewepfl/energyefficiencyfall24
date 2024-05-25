import numpy as np
import pandas as pd

images_df = pd.read_csv('/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/Clusters_images/clean_clustered_images_5_6.csv')
images_df = images_df.groupby(['Property Reference Id', 'cluster']).first().reset_index().drop(columns = ['Unnamed: 0', 'letter'])
print(images_df.head())

image_counts = images_df.groupby('Property Reference Id').size()

# Choose only the ones with 6 clusters.
valid_property_ids = image_counts[image_counts >= 5].index
images_df = images_df[images_df['Property Reference Id'].isin(valid_property_ids)]

# Save.
images_df.to_csv('/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/Clusters_images/clustered_images.csv')
print(f'The images were saved, from 211082 we result having {images_df.shape[0]} properly classified images. \n')