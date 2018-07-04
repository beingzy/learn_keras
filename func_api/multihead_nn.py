"""Multi inputs neural network
"""
from keras.models import Model
from keras import layers
from keras import Input


class AnswerQuestionMachine(object):

    def __init__(self,
                 text_vocabulary_size,
                 question_vocabulary_size,
                 answer_vocabulary_size):
        self.text_vocabulary_size = text_vocabulary_size
        self.question_vocabulary_size = question_vocabulary_size
        self.answer_vocabulary_size = answer_vocabulary_size

    def _build_text_arm(self):
        from keras.layers import Embedding, LSTM

        text_input = Input(
            shape=(None,), dtype='int32', name='text')
        embedded_text = Embedding(
            64, self.text_vocabulary_size, name='text_embedding')(text_input)
        return text_input, LSTM(32)(embedded_text)

    def _build_question_arm(self):
        from keras.layers import Embedding, LSTM

        question_input = Input(
            shape=(None,), dtype='int32', name='question')
        embedded_question = Embedding(
            32, self.question_vocabulary_size, name="question_embedding")(question_input)
        return question_input, LSTM(16)(embedded_question)

    def _add_optimizer(self):
        raise NotImplementedError

    def build(self):
        from keras.layers import Dense

        text_input, encoded_text = self._build_text_arm()
        question_input, encoded_question = self._build_question_arm()
        concatenated = layers.concatenate(
            [encoded_text, encoded_question], axis=-1)

        answer = Dense(self.answer_vocabulary_size,
                       activation='softmax',
                       name='output_layer')(concatenated)

        model = Model([text_input, question_input], answer)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        return model
