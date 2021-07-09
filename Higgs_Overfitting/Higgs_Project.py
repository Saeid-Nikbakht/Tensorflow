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

# Downloading the Higgs dataset
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28

# Reading the csv records directly from a gzip file with no intermediate decompression step
ds = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES+1), compression_type="GZIP")

# Repacking the list of scalars for each record into a (feature_vector, label) pair
def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

# Applying the `pack_row` function to each batch, and then splits the batches back up into individual records
packed_ds = ds.batch(10000).map(pack_row).unbatch()

# Lets use just the first 1000 samples for validation, and the next 10 000 for training:
N_VALIDATION = 1000
N_TRAIN = 10000
BUFFER_SIZE = 10000
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

# The `Dataset.skip` and `Dataset.take` methods make this easy.
# At the same time, use the `Dataset.cache` method to ensure that the loader doesn't \
# need to re-read the data from the file on each epoch:
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

# These datasets return individual examples. Use the `.batch` method to create batches \
# of an appropriate size for training. Before batching also remember to `.shuffle` and `.repeat` the training set.
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

#########################
# Demonstrate Overfitting
#########################

# Learning rate Scheduling:
# hyperbolically decrease the learning rate to 1/2 of the base rate at 1000 epochs, 1/3 at 2000 epochs and so on.
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001,
                                                             decay_steps=STEPS_PER_EPOCH*1000,
                                                             decay_rate=1,
                                                             staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

# Plotting the learning rate over steps
step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')


# Callbacks:
# 1) tfdocs.modeling.EpochDots : To reduce the logging noise
# 2) tf.keras.callbacks.EarlyStopping
# 3) tf.keras.callbacks.TensorBoard : to generate TensorBoard logs for the training

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]


# A function for Compile and Fit all models:
def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

############
# Tiny model
############

# Building the model:
tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

# Checking the model Performance:
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

#############
# Small model
#############

# Building the model:
small_model = tf.keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

##############
# Medium model
##############

# Building the model:
medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])
size_histories['Medium'] = compile_and_fit(medium_model, "sizes/Medium")

#############
# Large model
#############

# Building the model:
large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])
size_histories['large'] = compile_and_fit(large_model, "sizes/large")

#########################################
# Plot the training and validation losses
#########################################

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")

# Define a dictionary to input the results of different models containing regularizers
regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']

###########################
# Add weight regularization
###########################

# Building the model:
l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")

# Checking the influence of Weight Regularization
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

#############################
# Adding Dropout to our Model
#############################

dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

####################################################
# Combining L2 Regularization and Dropout Techniques
####################################################

combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])