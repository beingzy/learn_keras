"""
"""
import keras
from keras import layers
import numpy as np


# prepare training data
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

max_len = 60
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i:i+max_len])
    next_chars.append(text[i+max_len])

print('Number of sequences: ', len(sentences))
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectorization...')
x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentences):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# building the network
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(max_len, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
rmsprop_optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop_optimizer)
