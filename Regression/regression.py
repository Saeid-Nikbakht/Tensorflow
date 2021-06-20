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

# Make numpy printouts easier to read.

# Downloading the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

raw_df = pd.read_csv(url, names=column_names,
                     na_values='?', comment='\t',
                     sep=' ', skipinitialspace=True)

df = raw_df.copy()
print(df.tail())

# clean the data
print("Missing values in this dataset: \n{}".format(df.isna().sum()))

# As there are 6 missing values in the column: Horsepower we drop those rows.
df = df.dropna().reset_index()

# Mapping USA, Europe and Japan to 1, 2 and 3
df['Origin'] = df['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

df = pd.get_dummies(df, columns=['Origin'], prefix='', prefix_sep='')
print(df.tail())

# Split the data into train and test
train = df.sample(frac=0.8, random_state=1234)
test = df.drop(train.index)

sns.pairplot(train[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

print(train.describe().transpose())

# Split features from labels
train_features = train.copy()
test_features = test.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

##################
# Helper Functions
##################

# A function for comparing the a sample of the dataset before and after normalization
def compare_norm(row, normalizer):
    with np.printoptions(precision=2, suppress=True):
        print("Showing a random row of dataset before and after normalization :")
        print('First example:', row)
        print('Normalized:   ', normalizer(row).numpy())

# A function for plotting the history of loss and val-loss over the course of training
def plot_loss(history):
    plt.figure(figsize = [6, 6])
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

# A function for saving the results of all models in a dictionary
def dict_all_results(model, test_features, test_labels, dictionary):

    if dictionary is None:
        dictionary = {}

    dictionary[model.name] = model.evaluate(test_features, test_labels, verbose=0)
    print("The results of Loss so far are:")
    print(dictionary)

    return dictionary

# A function for comparing the predicted and true test samples (only when we have 1 featured data)
def plot_hp_model_performance(model):

    # Since this is a single variable regression it's easy to look at the model's predictions as a function of the input:
    x = tf.linspace(0.0, 250, 251)
    y = model.predict(x)
    plt.figure(figsize = [6, 6])
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.title(model.name)
    plt.legend()

# A function for plotting the error histogram
def plot_error_histogram(test_labels, test_predictions, name):

    fig, [ax1, ax2] = plt.subplots(nrows = 1, ncols = 2, figsize = [12, 6])

    # Error distribution:
    error = test_predictions - test_labels
    ax1.hist(error, bins=25)
    ax1.set_xlabel('Prediction Error [MPG]')
    ax1.set_ylabel('Count')
    ax1.set_title(name)

    # prediction distribution
    ax2.plot([10, 40],[10, 40])
    ax2.scatter(test_labels, test_predictions)
    ax2.set_xlabel('True Values [MPG]')
    ax2.set_ylabel('Predictions [MPG]')

##################################################################################
# Linear regression(All Variables):This model will predict `MPG` from `Horsepower`.
##################################################################################

# The Normalization layer
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# Building the model:
linear_model = tf.keras.Sequential([normalizer, Dense(units=1)], name = 'linear_model')
linear_model.summary()

# When you call this model on a batch of inputs, it produces `units=1` outputs for each example.
#linear_model.predict(train_features[:10])

# When you call the model it's weight matrices will be built. Now you can see that the `kernel` (the $m$ in $y=mx+b$) has a shape of `(9,1)`.
#print("Weights:")
#print(linear_model.layers[1].kernel)

first = np.array(train_features[:1])
compare_norm(first, normalizer)

# Compile and Fit the model:
linear_model.compile(optimizer=Adam(learning_rate=0.1), loss=MAE)
history = linear_model.fit(train_features, train_labels, epochs=100, verbose=0, validation_split=0.2)

# Visualize the model's training progress using the stats stored in the `history` object:
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Plotthe history of loss:
plot_loss(history)

# Collect the results in test_results dictionary:
test_results = dict_all_results(linear_model, test_features, test_labels, dictionary=None)

##################################################################################
# Linear regression(One Variable):This model will predict `MPG` from `Horsepower`.
##################################################################################
hp = np.array(train_features['Horsepower'])

# The Normalization layer
hp_normalizer = preprocessing.Normalization(input_shape=[1, ]) # input shape is needed only if we have a 1D array for normalization
hp_normalizer.adapt(hp)

# Build the sequential model:
hp_model = tf.keras.Sequential([hp_normalizer, layers.Dense(units=1)], name = 'hp_model')
hp_model.summary()

# Compile and Fit the model:
hp_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')
history = hp_model.fit(train_features['Horsepower'], train_labels, epochs=100, verbose=0, validation_split=0.2)

# Visualize the model's training progress using the stats stored in the `history` object:
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Plot the history of loss:
plot_loss(history)

# Collect the results in test_results dictionary:
test_results = dict_all_results(hp_model, test_features['Horsepower'], test_labels, dictionary=test_results)

# Plot the comparison between predicted and true test data
plot_hp_model_performance(hp_model)

##################
# A DNN regression
##################

# A helper function for build and compile DNN models
def build_and_compile_model(norm, name):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ], name=name)

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

#########################################################
# Start with a DNN model for a single input: "Horsepower"
#########################################################

# Build and Compile the model:
dnn_hp_model = build_and_compile_model(hp_normalizer, "dnn_hp_model")
dnn_hp_model.summary()

# Fit the model:
history = dnn_hp_model.fit(train_features['Horsepower'], train_labels, epochs=100, verbose=0, validation_split=0.2)

# Plot the history of loss:
plot_loss(history)

# Plot the comparison between predicted and true test data
plot_hp_model_performance(dnn_hp_model)

# Collect the results in test_results dictionary:
test_results = dict_all_results(dnn_hp_model, test_features['Horsepower'], test_labels, dictionary=test_results)

#################################
# A full DNN model for all inputs
#################################

# Build and Compile the model:
dnn_full_model = build_and_compile_model(normalizer, "dnn_full_model")
dnn_full_model.summary()

# Fit the model:
history = dnn_full_model.fit(train_features, train_labels, epochs=100, verbose=0, validation_split=0.2)

# Plot the history of loss:
plot_loss(history)

# Collect the results in test_results dictionary:
test_results = dict_all_results(dnn_full_model, test_features, test_labels, dictionary=test_results)

#################
# Post Processing
#################

# The Performance of all models:
perfromance_df = pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
print(perfromance_df)

# Make predictions
pred_dnn_full = dnn_full_model.predict(test_features).flatten()
pred_dnn_hp = dnn_hp_model.predict(test_features['Horsepower']).flatten()
pred_linear_full = linear_model.predict(test_features).flatten()
pred_linear_hp = hp_model.predict(test_features['Horsepower']).flatten()


plot_error_histogram(test_labels, pred_dnn_full, "Full DNN Model")
plot_error_histogram(test_labels, pred_dnn_hp, "DNN Model: Horsepower")
plot_error_histogram(test_labels, pred_linear_full, "Full Linear Model")
plot_error_histogram(test_labels, pred_linear_hp, "Linear Model: Horsepower")

# Save the model
dnn_full_model.save('dnn_full_model.h5')

# Reload the model
# reloaded = tf.keras.models.load_model('dnn_model')

