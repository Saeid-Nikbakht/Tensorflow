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
My_project_path = "C:/Users/Saeid/Desktop/Github/Tensorflow/Mnist_Checkpoint"
os.chdir(My_project_path)

# Import the Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

# Normalizing the training and test datasets
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define a function for building and compiling a model
def create_model():
  model = Sequential([
    Dense(512, activation=elu, input_shape=(784,)),
    Dropout(0.2),
    Dense(10, activation = softmax)
  ])

  model.compile(optimizer='adam',
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

model = create_model()
model.summary()

##################################
# Save checkpoints during training
##################################

# Create a callback that saves weights only during training:
checkpoint_path = "SavedCheckpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,
                              verbose=1)

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# Then load the weights from the checkpoint and re-evaluate:
# Loads the weights (In this case we do not need to save the weights, we can simply just load them from the model)
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

########################################################
# Save checkpoints during training with detailed options
########################################################

# Train a new model, and save uniquely named checkpoints once every five epochs:
checkpoint_path = "SavedDetailedCheckpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32
# Create a callback that saves the model's weights every 5 epochs
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              verbose=1,
                              save_weights_only=True,
                              save_freq=5*batch_size)

# Create a new model instance
model = create_model()

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=50,
          batch_size=batch_size,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

# Now, look at the resulting checkpoints and choose the latest one:
os.listdir(checkpoint_dir)
latest = tf.train.latest_checkpoint(checkpoint_dir)

# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#######################
# Manually save weights
#######################

model.save_weights('SavedManualCheckpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('SavedManualCheckpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

############################################
# Save the entire model / SavedModel format:
############################################

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the Model
model.save('SavedModel/SavedModelFormat')
new_model = tf.keras.models.load_model('SavedModel/SavedModelFormat')

# Check its architecture
new_model.summary()

# The restored model is compiled with the same arguments as the original model. Try running evaluate and predict\
# with the loaded model:

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

######################################
# Save the entire model / HDF5 format:
######################################

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
model.save('SavedModel/HDF5Format.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model = load_model('SavedModel/HDF5Format.h5')

# Show the model architecture
new_model.summary()

# Check its accuracy:
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
