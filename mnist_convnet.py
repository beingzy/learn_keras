"""build a 2-d convnet-based multi-layer representationla model to
   map grey-scaled hand-written-digit image to one of 10 classes

   author: Yi Zhang <beingzy@gmail.com>
   date: 2018/02/26
"""
import keras
from keras import models
from keras import layers
from keras import callbacks
from keras.datasets import mnist
from keras.utils import to_categorical


def build_model():
    """
    """
    model = models.Sequential()
    # convolutional layers
    model.add(layers.Conv2D(32, (3, 3),
                            activation='relu',
                            input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # densely connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def image_data_formatter(data, shape):
    """
    """
    data = data.copy()
    data = data.reshape(shape)
    data = data.astype('float32') / 255 # squash (0, 255) to (0, 1)
    return data


# prepare data
(train_image, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = image_data_formatter(train_image, (60000, 28, 28, 1))
test_images = image_data_formatter(test_images, (10000, 28, 28, 1))

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# build model and train
model = build_model()

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
history = model.fit(train_images,
                    train_labels,
                    epochs=5,
                    batch_size=64,
                    callbacks=callbacks_list,
                    validation_split=0.2)
