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
