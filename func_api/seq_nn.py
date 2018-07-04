from keras.models import Sequential, Model
from keras import layers
from keras import Input


def multi_clf_seq_model():
    """sequentail models
    """
    from keras.layers import Dense

    seq_model = Sequential()
    seq_model.add(Dense(32, activation='relu', input_shape=(64,)))
    seq_model.add(Dense(32, activation='relu'))
    # output layer
    seq_model.add(Dense(10, activation='softmax'))

    return seq_model


def multi_clf_func_model():
    """using functional API
    """
    from keras.layers import Dense

    input_tensor = Input(shape=(64,))
    output_l01 = Dense(32, activation='relu')(input_tensor)
    output_l02 = Dense(32, activation='relu')(output_l01)
    output_tensor = Dense(10, activation='softmax')(output_l02)

    return Model(input_tensor, output_tensor)


if __name__ == "__main__":
    seq_model = multi_clf_seq_model()
    func_model = multi_clf_func_model()

    print("###### PRINT SEQUENTIAL MODEL ##########")
    print(seq_model.summary())

    print("###### PRINT FUNCTIONAL MODEL ##########")
    print(func_model.summary())
