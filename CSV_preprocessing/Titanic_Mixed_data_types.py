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
import pathlib

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

# Importing the Titanic Dataset (label = Survived or Not)
titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()

# Seperating the Features and labels
titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

# Building a preprocessing model by building a set of symbolic `keras.Input` objects, matching\
# the names and data-types of the CSV columns.

inputs = {}

for name, column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

# Concatenating the numeric inputs together and running them through a normalization layer:
numeric_inputs = {name: input for name, input in inputs.items()
                  if input.dtype == tf.float32} # This parameter is a DICTIONARY

x = layers.Concatenate()(list(numeric_inputs.values())) # numeric_inputs.values() and x are TENSORS

# Normalization layer
norm = preprocessing.Normalization()

# Adapting the normalization layer to the numeric inputs NP.ARRAY
norm.adapt(np.array(titanic[numeric_inputs.keys()]))

# Inputting the x TENSOR to the adapted normalization layer
all_numeric_inputs = norm(x)

# Collecting all the symbolic preprocessing results, to concatenate them later.
preprocessed_inputs = [all_numeric_inputs]

# `preprocessing.StringLookup` --> mapping from strings to integer indices
# `preprocessing.CategoryEncoding` --> convert the indexes into `float32` data
for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = preprocessing.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())
    #print(lookup.vocab_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

# Collecting `inputs` and `processed_inputs` --> concatenating all the preprocessed inputs together,\
# and building a model that handles the preprocessing:

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
titanic_preprocessing.summary()


# Feature DS dictionary:
titanic_features_dict = {name: np.array(value)
                         for name, value in titanic_features.items()}

# Slice out the first training example and pass it to this preprocessing model, you see the numeric features\
# and string one-hots all concatenated together:

features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)


# Building the model on top of this:
def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(256, activation=relu),
        layers.Dense(64, activation=relu),
        layers.Dense(1,activation=sigmoid)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam())
    return model

titanic = titanic_model(titanic_preprocessing, inputs)
titanic.summary()

# Training the model
titanic.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

# Evaluating the model
Sur_pred = np.round(titanic.predict(titanic_features_dict))
print("The accuracy of this model after 10 epochs was {0:.2%}".format(accuracy_score(Sur_pred, titanic_labels)))

plt.figure()
sns.heatmap(confusion_matrix(Sur_pred, titanic_labels),vmin = 25, vmax = 300, cmap = 'Reds', annot = True)
plt.xlabel("Predicted_Survival")
plt.ylabel("True_Survival")

####################################################################################################################
# Using tf.data: Having more control over the input data pipeline/Using the data that doesn't easily fit into memory
####################################################################################################################

# `Dataset.from_tensor_slices` constructor returns a `tf.data.Dataset`: A generator which slices the name and values\
# of our dataset
features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)

# A preview of  `Dataset.from_tensor_slices`:
for example in features_ds:
    for name, value in example.items():
        print(f"{name:19s}: {value}")
    break

# The `from_tensor_slices` function can handle any structure of dataset of `(features_dict, labels)` pairs:
titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))

# Shuffling and Batching the data
titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)

# Passing the batched data through the learned model and train it for 5 more epochs
titanic.fit(titanic_batches, epochs=5)

# Evaluating the model
Sur_pred = np.round(titanic.predict(titanic_features_dict))
print("The accuracy of this model after 15 epochs was {0:.2%}".format(accuracy_score(Sur_pred, titanic_labels)))

plt.figure()
sns.heatmap(confusion_matrix(Sur_pred, titanic_labels),vmin = 25, vmax = 300, cmap = 'Reds', annot = True)
plt.xlabel("Predicted_Survival")
plt.ylabel("True_Survival")

####################
# From a single file
####################

titanic_file_path = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

# Reading the CSV data from the file and create a `tf.data.Dataset`.
titanic_csv_ds = tf.data.experimental.make_csv_dataset(
    titanic_file_path,
    batch_size=5, # Artificially small to make examples easier to show.
    label_name='survived',
    num_epochs=1,
    ignore_errors=True)

# A preview of  `tf.data.Dataset`:
for batch, label in titanic_csv_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value}")
  print()
  print(f"{'label':20s}: {label}")

####################
# `tf.io.decode_csv`
####################
text = pathlib.Path(titanic_file_path).read_text()
lines = text.split('\n')[1:-1]

all_strings = [str()]*10
all_strings

features = tf.io.decode_csv(lines, record_defaults=all_strings)

for f in features:
  print(f"type: {f.dtype.name}, shape: {f.shape}")

# To parse them with their actual types, create a list of `record_defaults` of the corresponding types:

print(lines[0])

titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]
titanic_types

features = tf.io.decode_csv(lines, record_defaults=titanic_types)

for f in features:
  print(f"type: {f.dtype.name}, shape: {f.shape}")

###################################
# `tf.data.experimental.CsvDataset`
###################################

# The `tf.data.experimental.CsvDataset` class provides a minimal CSV `Dataset` interface without the convenience\
# features of the `make_csv_dataset` function: column header parsing, column type-inference, automatic shuffling,\
# file interleaving.

# This constructor follows uses `record_defaults` the same way as `io.parse_csv`:

simple_titanic = tf.data.experimental.CsvDataset(titanic_file_path, record_defaults=titanic_types, header=True)

for example in simple_titanic.take(1):
  print([e.numpy() for e in example])

def decode_titanic_line(line):
  return tf.io.decode_csv(line, titanic_types)

manual_titanic = (
    # Load the lines of text
    tf.data.TextLineDataset(titanic_file_path)
    # Skip the header row.
    .skip(1)
    # Decode the line.
    .map(decode_titanic_line)
)

for example in manual_titanic.take(1):
  print([e.numpy() for e in example])

