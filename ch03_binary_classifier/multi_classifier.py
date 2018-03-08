"""build a classifier to map newwire's articles into 46 mutully exclusive
   topics.

author: Yi Zhang <beingzy@gmail.com>
date: 2018/02/19
"""
import numpy as np
import keras
from keras import models
from keras import layers
from keras.datasets import reuters


# partition data into train and test
num_words = 10000
(train_data, train_labels), (test_data, test_labels) = (reuters
    .load_data(num_words=num_words))

# reverse mapping integer to words
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def vectorize_sequences(sequences, dimension=num_words):
    """
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def to_one_hot(labels, dimension=46):
    """
    """
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# prepare data for algorithm to learn
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# build mutli-class classifier
model = model.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(num_words,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# validation
n_obs_validation = 10000
x_val = x_train[:n_obs_validation]
partial_x_train = x_train[n_obs_validation:]

y_val = one_hot_train_labels[:n_obs_validation]
partial_y_train = one_hot_train_labels[n_obs_validation:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
