"""
"""
import keras
from keras.layers import Dense, Embedding, SimpleRNN, LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
max_len = 500
batch_size = 32

print('loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(
    num_words = max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)
print('input_train shape', input_train.shape)
print('input_test shape', input_test.shape)

# build RNN model
def build_simple_rnn():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def build_lstm_rnn():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def build_1d_convnet():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(Conv1D(32, 7, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(32, 7, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model

# train model
# model=build_simple_rnn() simple_rnn's best validation acc: 0.8200
# model = build_lstm_rnn()
model = build_1d_convnet()

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath = 'covnet_1d_imdb.h5', #'simple_rnn_imdb.h5',
        monitor = 'val_loss',
        save_best_only = True
    ),
    keras.callbacks.TensorBoard(
        log_dir = 'my_log_dir',
        histogram_freq = 1
    )
]

# view tensorboard by running the below commands
# tensorboard --logdir=my_log_dir
history = model.fit(input_train,
                    y_train,
                    epochs=20,
                    batch_size=128,
                    callbacks=callbacks_list,
                    validation_split=0.2)
