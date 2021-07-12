# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:48:52 2021

@author: Saeid
"""

#!pip install git+https://github.com/tensorflow/docs

# Importing Required Modules
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MAE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import elu, relu, softmax
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import regularizers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import os
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pathlib
import shutil
import tempfile
import numpy as np
import os
import PIL
import PIL.Image

#pip install tensorflow_datasets
import tensorflow_datasets as tfds

print(tf.__version__)


# Downloading the flowers dataset
# Classes are :# flowers_photos : daisy/dandelion/roses/sunflowers/tulips

my_dir1 = "C:/Users/Saeid/Desktop/Github/Tensorflow/Image_preprocessing/Dataset"
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True,
                                   cache_subdir=my_dir1)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print("There are {} images in this dataset" .format(image_count))

#####################################
# Load using `tf.keras.preprocessing`
#####################################

img_height, img_width, batch_size = 180,180,32
# Train-Validation Split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# You can find the class names in the `class_names` attribute on these datasets.
class_names = train_ds.class_names
print(class_names)

# Visualize the data
# Here are the first 9 images from the training dataset.
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Manually iterate over the dataset and retrieve batches of images:
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

######################
# Standardize the data
######################

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) # or `Rescaling(1./127.5, offset=-1)

# Adding normalization layer with map function
# Or, it is possible to include the layer inside our model definition to simplify deployment.
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))


# Configure the dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Build, compile and train a Conv2d model

num_classes = 5
model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# Note: You will only train for a few epochs so this tutorial runs quickly. 

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

###################################
# Using `tf.data` for finer control
###################################

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

# The name of classes which are the name of folders in the data_dir
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

# Split the dataset into train and validation:
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# the length of each dataset as follows:
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

# A function that converts a file path to an `(img, label)` pair:
def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Use `Dataset.map` to create a dataset of `image, label` pairs:

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())


# Configure dataset for performance
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

# Visualize the data

image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")


# Continue training the model

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

###########################
# Using TensorFlow Datasets
###########################

# Download the flowers [dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) using TensorFlow Datasets.

my_dir2 = "C:/Users/Saeid/Desktop/Github/Tensorflow/Image_preprocessing/Dataset_TF"
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
    data_dir = my_dir2)

# The flowers dataset has five classes.
num_classes = metadata.features['label'].num_classes
print(num_classes)

# Retrieve an image from the dataset.

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
plt.figure()
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))


# As before, remember to batch, shuffle, and configure each dataset for performance.
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)
