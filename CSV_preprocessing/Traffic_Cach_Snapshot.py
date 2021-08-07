# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:48:52 2021

@author: Saeid
"""

# Importing Required Modules
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MAE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import elu, relu, softmax, sigmoid
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import os
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pydot
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Modules options for better presentation
np.set_printoptions(precision=3, suppress=True)
pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
pd.options.display.width = 400
plt.style.use('seaborn')

# Preventing showing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(3)
warnings.filterwarnings(action='ignore')

# Chose a proper PWD
Pycharm_path = "C:/Users/Saeid/PycharmProjects/Tensorflow"
My_own_path = "C:/Users/Saeid/Desktop/New_TF_Basics/Py_Scripts/4-Save and Load Model"
os.chdir(My_own_path)

#####################
# Reading the dataset
#####################

traffic_volume_csv_gz = tf.keras.utils.get_file(
    'Metro_Interstate_Traffic_Volume.csv.gz',
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
    cache_dir='.', cache_subdir='traffic')

# Set the `compression_type` argument to read directly from the compressed file:
traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(
    traffic_volume_csv_gz,
    batch_size=256,
    label_name='traffic_volume',
    num_epochs=1,
    compression_type="GZIP")

# A preview of  `Dataset.from_tensor_slices`:
for batch, label in traffic_volume_csv_gz_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value[:5]}")
  print()
  print(f"{'label':20s}: {label[:5]}")


# Caching : `Dataset.cache`  stores the data form the first epoch and replays it in order
# Note: any shuffles earlier in the pipeline. Below the `.shuffle` is added back in after `.cache`.
caching = traffic_volume_csv_gz_ds.cache().shuffle(1000)
for i, (batch, label) in enumerate(caching.shuffle(1000).repeat(20)):
    if i % 40 == 0:
        print('.', end='')
        print()

# Snaoshot: `snapshot` files are meant for *temporary* storage of a dataset while in use. This is *not* a format\
# for long term storage. The file format is considered an internal detail, and not guaranteed between TensorFlow\
# versions.
snapshot = tf.data.experimental.snapshot('titanic.tfsnap')
snapshotting = traffic_volume_csv_gz_ds.apply(snapshot).shuffle(1000)
for i, (batch, label) in enumerate(snapshotting.shuffle(1000).repeat(20)):
    if i % 40 == 0:
        print('.', end='')
    print()

