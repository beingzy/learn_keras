"""build a binary classifier to categorize IMDB reviews into either
   negative or positive

   author: Yi Zhang <beingzy@gmail.com>
   date: 2018/02/18
"""
import sys
import os
import logging

import numpy as np
import tensorflow as tf
import keras

from keras.datasets import imdb
from keras import models
from kears import layers
from kears import optimizers
from keras import losses
from keras import metrics

word_dict_size = 10000
# prepare data for training and testing
# train/test data: is a sequence of indices of words comprising a review
# train/test labels: 0 - negative, 1 - positive
# within labels: 0 - 50%, 1 - 50%
(train_data, train_labels), (test_data, test_labels) = (imdb
    .load_data(num_words=word_dict_size))


# develop utility function to restore words information
word_index = imdb.get_word_index()
# mapping integer to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def get_word_with_index(index):
    """mapping integer to words.
       index is subtracted by 3 because of that 0, 1, 2 are reserved indices
       for 'padding', 'start of sequence' and 'unknown'
    """
    return reverse_word_index[index-3]

# prepare data for algorithm: vectorize word indices
def vectorize_sequences(sequences, dimension=word_dict_size):
    """convert list of sequneces of word index (integer)
       into a 2D-array of dummy variable, indicating wether a word
       appears.
    """
    results = np.zeros((len(sequences), dimension))
    for ii, sequence in enumerate(sequences):
        results[ii, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# build the neural nets to learn the pattern for mapping
# review into sentiment: positive vs. negative
# nerual nets architect (fully connected layers):
#    input --> (16, relu) --> (16, relu) --> (1, sigmoid) --> output
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(word_dict_size,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

rmsprop_opt = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rmsprop_opt,
              loss=loss.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

              
