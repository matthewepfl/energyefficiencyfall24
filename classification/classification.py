# -*- coding: utf-8 -*-
'''VGG16-places365 model for Keras

# Reference:
- [Places: A 10 million Image Database for Scene Recognition](http://places2.csail.mit.edu/PAMI_places.pdf)

This file performs the clustering of the images using the VGG16-places365 model and using kmeans unsupervised learning algorithm.
'''

from __future__ import division, print_function
import os
import numpy as np
from PIL import Image
import os
import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd

from classification_helper import *

WEIGHTS_PATH = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'
SAVE_PATH = '/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/'
SCRATCH_PATH = '/scratch/sromanat/'

def load_and_preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

from tqdm import tqdm  # Import tqdm

def extract_features(img_paths, model):
    features = []
    for img_path in tqdm(img_paths, desc='Extracting features'):
        processed_img = load_and_preprocess_img(img_path)
        feature = model.predict(processed_img, verbose=0)
        features.append(feature)
    return np.array(features)

def load_images():
    if not os.path.exists(SAVE_PATH + 'images_df.csv'):
        images_df = get_the_images_dataset(path_separator= '/')
        images_df.to_csv(SAVE_PATH + 'images_df.csv', index=False)
    else:
        images_df = pd.read_csv(SAVE_PATH + 'images_df.csv')
    img_path = images_df['pathname']
    return img_path, images_df

def plot_images_from_cluster(cluster_label, images_df, num_images=24, SAVE_PATH=''):
    filtered_df = images_df[images_df['cluster'] == cluster_label]
    sample_images = filtered_df['pathname'].sample(n=num_images)
    plt.figure(figsize=(24, 9))  # Adjust size as needed
    for index, img_path in enumerate(sample_images):
        plt.subplot(3, 8, index + 1)  # 3 rows, 8 columns
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()  # Adjust layout to not overlap images
    plt.savefig('/work/FAC/HEC/DEEP/shoude/ml_green_building/Data/Clusters_images/cluster_' + str(cluster_label) + '.png')  # Save the plot

def incremental_PCA(n_batches = 100, best_number_of_components = 125, variance_explained = 0):

    while variance_explained < 0.95:
        print(f'Trying the number of components: {best_number_of_components}')

        ipca = IncrementalPCA(n_components=best_number_of_components, batch_size=None)
        now = time.time()
        for i, X_batch in enumerate(np.array_split(features_flattened, n_batches)):
            ipca.partial_fit(X_batch)
        variance_explained = ipca.explained_variance_ratio_.sum()
        print(f'The best number of components explain {variance_explained:.2f} of the variance')

        best_number_of_components += 25

    features_reduced = ipca.transform(features_flattened)
    return features_reduced

def optimal_K(features, brute_choice = True, plot = False):
    distortions = []
    K = range(1, 11)

    if brute_choice:
        return 6
    
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(features)
        distortions.append(kmeanModel.inertia_)

    if plot:
        plt.figure(figsize=(16, 8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.savefig(SCRATCH_PATH + 'elbow_method.png')

    return distortions.index(min(distortions)) + 1

if __name__ == '__main__':

    # percentage of dataset to use
    PERCENTAGE = 1.0

    print("This file performs the clustering of the images using the VGG16 model and using kmeans unsupervised learning algorithm.")
    print("The output of this file is a csv file with the image path and the cluster label.")
    print("The output file is saved in the following path: ", SAVE_PATH, " and the name of the file is: image_cluster.csv")
    
    img_path, images_df = load_images()

    img_path = img_path[:int(len(img_path)*PERCENTAGE)]
    images_df = images_df[:int(len(images_df)*PERCENTAGE)]
    prediction_classification_task = []

    # Load the pre-trained VGG16 model
    model = VGG16(weights='imagenet', include_top=False)

    # Extracting features
    name = f'features_{PERCENTAGE}.npy'
    if not os.path.exists(SAVE_PATH + name):
        print("Extracting features because", os.path.exists(SAVE_PATH + name), "does not exist")
        features = extract_features(img_path, model)
        np.save(SAVE_PATH + name, features)
    else:
        print("Loading features from", SAVE_PATH + name)
        features = np.load(SAVE_PATH + name)

    # Flatten the features
    features_flattened = features.reshape(features.shape[0], -1)
    print("The shape of the features is: ", features_flattened.shape)

    # Fit
    best_number_of_clusters = optimal_K(features_flattened, brute_choice = False, plot = False)
    print('The number of clusters chosen is: ', best_number_of_clusters, 'but forced to 6')
    best_number_of_clusters = 6
    kmeans = KMeans(n_clusters=best_number_of_clusters, random_state=22, n_init='auto')

    prediction_classification_task = kmeans.fit_predict(features_flattened)
    images_df['cluster'] = prediction_classification_task
    images_df.to_csv(SAVE_PATH + 'Clusters_images/clustered_images.csv', index=False)

    for i in range(best_number_of_clusters):
        plot_images_from_cluster(i, images_df)
        print(f'Cluster {i} plotted and saved')

    print("The number of images in the dataset is: ", len(images_df), "and the file looks like this: \n", images_df.head(50))

    # Group by images and clusteres and select randomly the first one.
    images_df = images_df.groupby(['Property Reference Id', 'cluster']).first().reset_index()
    image_counts = images_df.groupby('Property Reference Id').size()

    print(image_counts.groupby(image_counts).size())
