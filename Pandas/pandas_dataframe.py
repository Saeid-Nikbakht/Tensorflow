# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:48:52 2021

@author: Saeid
"""

import pandas as pd
import tensorflow as tf
import time

# Download the csv file containing the heart dataset.
csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

# Read the csv file using pandas.
df = pd.read_csv(csv_file)

df.head()
df.dtypes


# Convert `thal` column which is an `object` in the dataframe to a discrete numerical value.
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
df.head()

start = time.time()
# Load data using `tf.data.Dataset`
target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# Shuffle and batch the dataset.
train_dataset = dataset.shuffle(len(df)).batch(1)

# Building, Compiling and Training a model
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)

print("The elapsed time related to pd.DataFrame was  {}".format(time.time()-start))

start = time.time()
# Alternative to feature columns : A dictionary, keys = columns names, values = the tensor containing the values of\
# the corresponded column
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}

# Stacking all the values of the aforementioned dictionary together and building the model
x = tf.stack(list(inputs.values()), axis=-1)
x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1)(x)
model_func = tf.keras.Model(inputs=inputs, outputs=output)

# Compiling the model
model_func.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])

# Converting the `pd.DataFrame` to a `dict`, and slice that dictionary.
dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

# Training the model
model_func.fit(dict_slices, epochs=15)

print("The elapsed time related to tensor dictionaries was  {}".format(time.time()-start))