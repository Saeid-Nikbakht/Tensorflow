# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:48:52 2021

@author: Saeid
"""

# Importing Required Modules
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import pandas as pd

# Modules options for better presentation
np.set_printoptions(precision=3, suppress=True)
pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
pd.options.display.width = 400
plt.style.use('seaborn')

#####################
# Reading the Dataset
#####################

fonts_zip = tf.keras.utils.get_file(
    'fonts.zip',  "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
    cache_dir='.', cache_subdir='fonts',
    extract=True)

import pathlib
font_csvs = sorted(str(p) for p in pathlib.Path('fonts').glob("*.csv"))

print("The length of this dataset is {}".format(len(font_csvs)))

fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=10, num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000)

# These csv files have the images flattened out into a single row. The column names are formatted\
# `r{row}c{column}`. Here's the first batch:

for features in fonts_ds.take(1):
  for i, (name, value) in enumerate(features.items()):
    if i>15:
      break
    print(f"{name:20s}: {value}")
print('...')
print(f"[total: {len(features)} features]")


# Packing fields : pack the pixels into an image-tensor

# Here is code that parses the column names to build images for each example:

def make_images(features):
  image = [None]*400
  new_feats = {}

  for name, value in features.items():
    match = re.match('r(\d+)c(\d+)', name)
    if match:
      image[int(match.group(1))*20+int(match.group(2))] = value
    else:
      new_feats[name] = value

  image = tf.stack(image, axis=0)
  image = tf.reshape(image, [20, 20, -1])
  new_feats['image'] = image

  return new_feats


# Apply that function to each batch in the dataset:
fonts_image_ds = fonts_ds.map(make_images)

# An example of an image in this dataset
for features in fonts_image_ds.take(1):
  break

# Plot the resulting images:

plt.figure(figsize=(6,6), dpi=120)

for n in range(9):
  plt.subplot(3,3,n+1)
  plt.imshow(features['image'][..., n])
  plt.title(chr(features['m_label'][n]))
  plt.axis('off')

# Multiple files

# To parse the fonts dataset using `experimental.CsvDataset`, you first need to determine the column types for\
# the `record_defaults`. Start by inspecting the first row of one file:

font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
print(font_line)

# Only the first two fields are strings, the rest are ints or floats, and you can get the total number of\
# features by counting the commas:

num_font_features = font_line.count(',') + 1
font_column_types = [str(), str()] + [float()] * (num_font_features - 2)

# The `CsvDatasaet` constructor can take a list of input files, but reads them sequentially.\
# The first file in the list of CSVs is `AGENCY.csv`:

font_csvs[0]

# So when you pass pass the list of files to `CsvDataaset` the records from `AGENCY.csv` are read first:

simple_font_ds = tf.data.experimental.CsvDataset(
  font_csvs,
  record_defaults=font_column_types,
  header=True)

for row in simple_font_ds.take(10):
  print(row[0].numpy())

# To interleave multiple files, use `Dataset.interleave`.

# Here's an initial dataset that contains the csv file names:

font_files = tf.data.Dataset.list_files("fonts/*.csv")

# This shuffles the file names each epoch:

print('Epoch 1:')
for f in list(font_files)[:5]:
  print("    ", f.numpy())
print('    ...')
print()

print('Epoch 2:')
for f in list(font_files)[:5]:
  print("    ", f.numpy())
print('    ...')


# The `interleave` method takes a `map_func` that creates a child-`Dataset` for each element of the parent-`Dataset`.

# Here, you want to create a `CsvDataset` from each element of the dataset of files:

def make_font_csv_ds(path):
  return tf.data.experimental.CsvDataset(
      path,
      record_defaults=font_column_types,
      header=True)


# The `Dataset` returned by interleave returns elements by cycling over a number of the child-`Dataset`s.\
# Note, below, how the dataset cycles over `cycle_length)=3` three font files:

font_rows = font_files.interleave(make_font_csv_ds,
                                cycle_length=3)

fonts_dict = {'font_name': [], 'character': []}

for row in font_rows.take(10):
  fonts_dict['font_name'].append(row[0].numpy().decode())
  fonts_dict['character'].append(chr(row[2].numpy()))

pd.DataFrame(fonts_dict)

################
# Multiple files
################

# To parse the fonts dataset using `experimental.CsvDataset`, you first need to determine the column types for\
# the `record_defaults`. Start by inspecting the first row of one file:

font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
print(font_line)

# Only the first two fields are strings, the rest are ints or floats, and you can get the total number of\
# features by counting the commas:

num_font_features = font_line.count(',')+1
font_column_types = [str(), str()] + [float()]*(num_font_features-2)


# The `CsvDatasaet` constructor can take a list of input files, but reads them sequentially.\
# The first file in the list of CSVs is `AGENCY.csv`:

font_csvs[0]


# So when you pass pass the list of files to `CsvDataaset` the records from `AGENCY.csv` are read first:

simple_font_ds = tf.data.experimental.CsvDataset(
    font_csvs,
    record_defaults=font_column_types,
    header=True)

for row in simple_font_ds.take(10):
  print(row[0].numpy())


# To interleave multiple files, use `Dataset.interleave`.

# Here's an initial dataset that contains the csv file names:

font_files = tf.data.Dataset.list_files("fonts/*.csv")

# This shuffles the file names each epoch:

print('Epoch 1:')
for f in list(font_files)[:5]:
  print("    ", f.numpy())
print('    ...')
print()

print('Epoch 2:')
for f in list(font_files)[:5]:
  print("    ", f.numpy())
print('    ...')


# The `interleave` method takes a `map_func` that creates a child-`Dataset` for each element of the parent-`Dataset`.

# Here, you want to create a `CsvDataset` from each element of the dataset of files:

def make_font_csv_ds(path):
  return tf.data.experimental.CsvDataset(
    path,
    record_defaults=font_column_types,
    header=True)

# The `Dataset` returned by interleave returns elements by cycling over a number of the child-`Dataset`s.\
# Note, below, how the dataset cycles over `cycle_length)=3` three font files:

font_rows = font_files.interleave(make_font_csv_ds,
                                  cycle_length=3)

fonts_dict = {'font_name':[], 'character':[]}

for row in font_rows.take(10):
  fonts_dict['font_name'].append(row[0].numpy().decode())
  fonts_dict['character'].append(chr(row[2].numpy()))

pd.DataFrame(fonts_dict)

#############
# Performance
#############

BATCH_SIZE=2048
fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=BATCH_SIZE, num_epochs=1,
    num_parallel_reads=100)

for i,batch in enumerate(fonts_ds.take(20)):
    print('.',end='')
print()


# Passing **batches of text lines** to`decode_csv` runs faster:
fonts_files = tf.data.Dataset.list_files("fonts/*.csv")
fonts_lines = fonts_files.interleave(
    lambda fname:tf.data.TextLineDataset(fname).skip(1),
    cycle_length=100).batch(BATCH_SIZE)

fonts_fast = fonts_lines.map(lambda x: tf.io.decode_csv(x, record_defaults=font_column_types))

for i,batch in enumerate(fonts_fast.take(20)):
    print('.',end='')
print()

