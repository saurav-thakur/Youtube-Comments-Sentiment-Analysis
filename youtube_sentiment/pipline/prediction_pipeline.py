import sys,os
from pandas import DataFrame
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import tensorflow as tf

from youtube_sentiment.constants import DATA_TRANSFORMATION_PAD_SEQUENCES_PADDING,DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN
from youtube_sentiment.entity.s3_estimator import YoutubeS3SentimentClassification
from youtube_sentiment.entity.config_entity import YoutubeSentimentPredictorConfig, ModelPredictionConfig
from youtube_sentiment.utils.utilities import load_keras_model,load_tokenizer
class PredictionPipeline:

    def __init__(self):
        # self.preprocessing_object = preprocessing_object
        # # self.trained_model = trained_model
        # self.youtube_sentiment_predict_config = YoutubeSentimentPredictorConfig()
        self.model_prediction_config = ModelPredictionConfig()
    
    def predict(self,dataframe: DataFrame):

        try:
            preprocessing_object_path = self.model_prediction_config.model_prediction_tokenizer
            model_path = self.model_prediction_config.model_prediction_final_model

            preprocessing_object = load_tokenizer(preprocessing_object_path)
            model = load_keras_model(model_path)

            tokenized_data = preprocessing_object.texts_to_sequences(dataframe)
            transformed_feature = pad_sequences(sequences=tokenized_data,padding=DATA_TRANSFORMATION_PAD_SEQUENCES_PADDING,maxlen=DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN)

            # model = YoutubeS3SentimentClassification(
            #     bucket_name= self.youtube_sentiment_predict_config.model_bucket_name,
            #     model_path=self.youtube_sentiment_predict_config.model_file_path,
            # )
            prediction = tf.argmax(model.predict(transformed_feature),axis=1)
            print("FINAL PREDICTIONN ISSSSSSSS",prediction.numpy()[0])
            return prediction.numpy()[0]
        
        except Exception as e:
            raise YoutubeException(e,sys)
        
