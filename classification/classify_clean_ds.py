import numpy as np
import pandas as pd

SAVE_PATH = '/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/'
images_df = pd.read_csv(SAVE_PATH + 'Clusters_images/clustered_images.csv')

# show some statistics
print(images_df.head(50))

# Group by images and clusteres and select randomly the first one.
images_df = images_df.groupby(['Property Reference Id', 'cluster']).first().reset_index()

# Count how many images that belong to each cluster each PRI has.
image_counts = images_df.groupby('Property Reference Id').size()

print(image_counts.groupby(image_counts).size())


# for i in [3,4,5,6]:
#     # Choose only the ones with 6 clusters.
#     valid_property_ids = image_counts[image_counts == i].index
#     images_df_only = images_df[images_df['Property Reference Id'].isin(valid_property_ids)]
#     valid_property_ids = image_counts[image_counts >= i].index
#     images_df_greater = images_df[images_df['Property Reference Id'].isin(valid_property_ids)]

#     # Save.
#     images_df_only.to_csv(SAVE_PATH + f'Clusters_images/clean_clustered_images_only{i}.csv')
#     images_df_greater.to_csv(SAVE_PATH + f'Clusters_images/clean_clustered_images_greater{i}.csv')
#     print(f'The images were saved for {i} cluster only this category has: {images_df_only.shape[0]} properly classified images. \n')
#     print(f'The images were saved for {i} cluster more at least this category has {images_df_greater.shape[0]} properly classified images. \n')