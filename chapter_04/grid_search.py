import pandas as pd
import numpy as np
import os
import math
from itertools import product

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# constants

checkpoint_path = '.checkpoints/cp-{epoch:03d}.ckpt'
num_features = 3
num_classes = 3

# parse the labels.csv
labels = pd.read_csv('labels.csv', index_col=0)
labels = labels.sort_values('id')

# grab filenames from the data directory
filenames = os.listdir('data')
filenames.sort()

dataframes = []

# parse and concatenate all csv files into df
for filename in filenames:
  if filename.endswith('.csv'):
    batch = pd.read_csv(os.path.join('data',filename), index_col=0)
    batch['batch'] = int(filename.replace('.csv', ''))
    dataframes.append(batch)

df = pd.concat(dataframes, ignore_index=True)

# clean up original dataframes
del dataframes

# add label column (if it is not already available)
if (not 'label' in df.columns):
  df = df.merge(labels, left_on=["batch"], right_on=["id"])


def time_to_float(inputstr):
  hours, minutes, seconds = map(float, inputstr.split(':'))

  # return hours * 3600 + minutes * 60 + seconds
  # this is sufficient because hours should always be 0
  return minutes * 60 + seconds

if (not df['sensorid'].dtype == 'int'):
  df['sensorid'] = df['sensorid'].astype('int')
if (not df['label'].dtype == 'category'):
  df['label'] = df['label'].astype('category')
if (not df['zeit'].dtype == 'float64'):
  df['zeit'] = df['zeit'].apply(time_to_float)

grouped = df.groupby('batch')

##### GRID SEARCH #####

batch_sizes = [64]
lsmt_units = [128, 256]
sequence_length = 256

# create sequences
sequences = []
sequence_labels = []

for batch, readings in grouped:
  readings = readings.sort_values('zeit')
  for i in range(0, len(readings) - sequence_length, sequence_length):
    sequence = readings.iloc[i:i + sequence_length]
    sequences.append(sequence[['zeit', 'sensorid', 'messwert']].values)
    sequence_labels.append(sequence['label'].values[0])

# convert to numpy arrays
sequences = np.array(sequences)
sequence_labels = np.array(sequence_labels)

# split into train, test and validation data
X_train, X_test, y_train, y_test = train_test_split(sequences, sequence_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# create a result dataframe to store the results
results = []

for batch_size, lsmt_unit_count in product(batch_sizes, lsmt_units):
  n_batches = math.ceil(len(X_train) / batch_size)

  model = Sequential()
  model.add(LSTM(lsmt_unit_count, input_shape=(sequence_length, num_features)))
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # callbacks
  cp_cb = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_freq=8*n_batches)
  stp_cb = EarlyStopping(monitor='val_loss', patience=16, restore_best_weights=True, min_delta=1e-4, start_from_epoch=32, verbose=1)

  model.fit(X_train, y_train, epochs=256, batch_size=batch_size, callbacks=[cp_cb, stp_cb], validation_data=(X_val, y_val))

  model.save(f'models/grid_search/lstm_{lsmt_unit_count}-batch_{batch_size}.keras')

  loss, acc = model.evaluate(X_test, y_test, verbose=2)

  # add results to result dataframe

# write results
df = pd.DataFrame(results)
df.to_csv('grid_search_results.csv')
