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
from tensorflow.keras.activations import elu, relu, softmax
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
os.chdir(Pycharm_path)

################
# In memory data
################

# Reading the Abalone Dataset which contains a set of measurements of different types of sea snail.
abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()

# Basic preprocessing
# Creating a Normalization Layer
normalize = preprocessing.Normalization()

# Adapting the preprocessing layer to our features
normalize.adapt(abalone_features)

# The nominal task for this dataset is to predict the age from the other measurements.
# Feature/Label splitting:
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

# For this dataset you will treat all features identically. Pack the features into a single NumPy array.:

abalone_features = np.array(abalone_features)
abalone_features

# Building, compiling and training a Regressive NN for predicting the Age of each Abalone
abalone_model = tf.keras.Sequential([
  layers.Dense(512, activation=relu),
  layers.Dense(32, activation=relu),
  layers.Dense(1)
])

abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

abalone_model.fit(abalone_features, abalone_labels, epochs=10)

# Evaluating the model
Age_pred = abalone_model.predict(abalone_features)
max_age = abalone_labels.max()
min_age = abalone_labels.min()

plt.figure()
plt.plot([min_age,max_age],[min_age,max_age])
plt.scatter(Age_pred, abalone_labels)
plt.xlabel("Predicted Age")
plt.xlabel("True Age")
