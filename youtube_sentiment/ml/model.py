import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Embedding, LSTM, Bidirectional
import os,sys
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.constants import DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN

def train_model(vocab_size):

    try:
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN))
        model.add(Bidirectional(LSTM(units=56,return_sequences=True)))
        model.add(Bidirectional(LSTM(units=28,return_sequences=True)))
        model.add(Bidirectional(LSTM(units=8,return_sequences=False)))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.build(input_shape=(None, DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN))
        logging.info(model.summary())

        return model
    except Exception as e:
        raise YoutubeException(e,sys)