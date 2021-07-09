# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:48:52 2021

@author: Saeid
"""

# Importing Required Modules
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import elu, softmax
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow import keras
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
My_project_path = "C:/Users/Saeid/Desktop/New_TF_Basics/Py_Scripts/4-Save and Load Model"
os.chdir(My_project_path)

# Install and import the Keras Tuner.
#pip install -q -U keras-tuner
import keras_tuner as kt

# Importing the Fashion Mnist Dataset
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# Normalizing the Pixels
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

##################
# Define the model
##################

def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

##############
# Hypertunning
##############

directory = "C:/Users/Saeid/Desktop/Github/Tensorflow/FMnist_Hypertuning/My_dir"
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=directory,
                     project_name='Hypertunning')


# Early stopping Callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Run the hyper-parameter search.
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyper-parameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print("The Hyper-parameter saerch is complete")
print("The optimum number of Neurons in the single layer is {}".format(best_hps.get('units')))
print("The optimum learning rate in the single layer is {}".format(best_hps.get('learning_rate')))


# Build the optimum model according to the hyper-parameter tuning search for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

# Finding the optimum epoch number regarding the val_accuracy
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d'.format(best_epoch))


# Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)


# Evaluating the hypermodel on the test data
eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)
