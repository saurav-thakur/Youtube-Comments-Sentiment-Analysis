import sys,os
from pandas import DataFrame
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import tensorflow as tf

from youtube_sentiment.constants import DATA_TRANSFORMATION_PAD_SEQUENCES_PADDING,DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN


class YoutubeSentimentPredictor:

    def __init__(self,preprocessing_object,trained_model):
        self.preprocessing_object = preprocessing_object
        self.trained_model = trained_model
    
    def predict(self,dataframe: DataFrame):

        try:
            tokenized_data = self.preprocessing_object.texts_to_sequences(dataframe)
            transformed_feature = pad_sequences(sequences=tokenized_data,padding=DATA_TRANSFORMATION_PAD_SEQUENCES_PADDING,maxlen=DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN)
            prediction = tf.argmax(self.trained_model.predict(transformed_feature),axis=1)

            return prediction
        
        except Exception as e:
            raise YoutubeException(e,sys)
        
