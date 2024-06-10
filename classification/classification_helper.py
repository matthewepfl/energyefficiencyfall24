from __future__ import division, print_function
import os
import warnings
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.utils import get_file, get_source_inputs
from keras.utils import get_file, get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape


WEIGHTS_PATH = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16_Places365(include_top=True, weights='places',
                    input_tensor=None, input_shape=None,
                    pooling=None,
                    classes=365):
    """Instantiates the VGG16-places365 architecture.

    Optionally loads weights pre-trained
    on Places. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
                 'places' (pre-training on Places),
                 or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape
        """
    if not (weights in {'places', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `places` '
                         '(pre-training on Places), '
                         'or the path to the weights file to be loaded.')

    if weights == 'places' and include_top and classes != 365:
        raise ValueError('If using `weights` as places with `include_top`'
                         ' as true, `classes` should be 365')


    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten =include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv1')(img_input)

    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool", padding='valid')(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv1')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool", padding='valid')(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv1')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv2')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool", padding='valid')(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool", padding='valid')(x)

    # Block 5
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool", padding='valid')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='drop_fc2')(x)
        
        x = Dense(365, activation='softmax', name="predictions")(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16-places365')

    # load weights
    if weights == 'places':
        if include_top:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        model.load_weights(weights_path)

        if K.backend() == 'theano':
            # Perform kernel format conversion for Theano
            for layer in model.layers:
                if hasattr(layer, 'kernel_initializer') and hasattr(layer.kernel_initializer, 'backend'):
                    original_backend = layer.kernel_initializer.backend
                    if original_backend == 'tensorflow':
                        layer.kernel_initializer.backend = 'theano'
                        layer.build(layer.input_shape)

        if K.image_data_format() == 'channels_first':
            # Convert weights for channels_first data format
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                dense.kernel = tf.transpose(dense.kernel, (1, 0))

    elif weights is not None:
        model.load_weights(weights)

    return model

def extract_features(ig, model):
    image = Image.open(str(ig))
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.uint8)
    image = np.expand_dims(image, 0)
    if image.shape == (1, 224, 224, 3):
        features = model.predict(image)
        features = features.flatten()
        return features
    else:
        print(f'image at path: {ig} has a shape of {image.shape} which is not allowed')

def kmeans_clustering(images_features, images_df):
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42,}
    # A list holds the SSE values for each k
    sse = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(images_features)
        sse.append(kmeans.inertia_)
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 11), sse)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    # save the plot
    print('saving the elbow plot...')
    plt.savefig('elbow_plot.png')

    # print the best value of k
    kl = KneeLocator(range(2, 11), sse, curve="convex", direction="decreasing")
    print(f"The best value of k is: {kl.elbow}")
    if kl.elbow == None:
        k_best = 6
    else:
        k_best = kl.elbow
    print('performing kmeans clustering...')
    # perform kmeans with the best value of k
    kmeans = KMeans(n_clusters=k_best)
    kmeans.fit(images_features)
    print('Done KMeans clustering...')
    print('Starting saving the results...')
    cluster_labels = kmeans.predict(images_features)
    # evaluate the clusteR calculate silhouette score  
    silhouette_avg = silhouette_score(images_features, cluster_labels)
    print("For n_clusters =", k_best, "The average silhouette_score is :", silhouette_avg)

    # Print the number of images in each cluster
    for i in range(k_best):
        num_images = np.sum(cluster_labels == i)
        print(f"Cluster {i}: {num_images} images")

    print('Done saving the results...')
    # creating the DataFrame of the images and the label and saving it as a pickle file
    data_dict = {"image_path": images_df['pathname'], "cluster_label": cluster_labels}
    df = pd.DataFrame(data_dict)
    print('The shape of the image_clusters.pkl is: ', df.shape)
    df.to_pickle("image_clusters.pkl")
    return df, k_best

def print_random_clusters(clustering, k_best):
    cluster_dict = {}
    for i, row in clustering.iterrows():
        label = row['cluster_label']
        image_path = row['image_path']
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(image_path)

    num_clusters = len(cluster_dict)
    num_cols = 4
    num_rows = (num_clusters // num_cols) + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    for cluster_num in range(k_best):
        # Choose a cluster to display
        cluster_to_display = cluster_num

        # Create a list of image file paths from the chosen cluster
        cluster_files = cluster_dict[cluster_to_display]

        # Set up the figure and subplots
        num_cols = 3
        num_rows = 3
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

        # pick 9 random images from the cluster_files list
        cluster_files = np.random.choice(cluster_files, 9, replace=False)
        # Iterate over the images in the cluster and display them
        n = 0
        for i, file in enumerate(cluster_files):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col]

            # Load the image and display it
            img = Image.open(file)
            ax.imshow(img)
            ax.set_title(f"Image {i+1}")

            # Remove the axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            n += 1
            if n == 9:
                break

        # Remove any unused subplots
        n = 0
        for i in range(len(cluster_files), num_rows*num_cols):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col]
            ax.axis("off")
            n += 1
            if n == 9:
                break

        # Display the figure
        plt.tight_layout()
        plt.show()
        
        # Save the figure
        plt.savefig(f"clusters_{cluster_num}.png")

def get_the_images_dataset(path_separator):
    """This function has as input the path of the images and the path separator type that it's being used. 
    It returns a dataframe with the images and the number of the image and the Property Reference Id."""

    PATH_IMAGES = '/work/FAC/HEC/DEEP/shoude/ml_green_building/images_full_data/MediaSyncFolder/*'

    images = []
    count = 0
    for folder in glob.glob(PATH_IMAGES):
        path = str(folder) + f"{path_separator}*.jpg"
        for img in glob.glob(path):
            images_data = {
                'image_array': cv2.imread(img, cv2.IMREAD_COLOR),
                'pathname': img, 
                'number':img.split(f'{path_separator}')[-1].split('.')[0], 
                'Property Reference Id': img.split(f'{path_separator}')[-2]
                }
            images.append(images_data)
        count += 1
    images_df = pd.DataFrame(images)
    return images_df
