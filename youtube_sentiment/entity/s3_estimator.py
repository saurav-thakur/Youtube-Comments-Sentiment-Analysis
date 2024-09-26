import sys
from pandas import DataFrame
from youtube_sentiment.cloud_storage.aws_storage import SimpleStorageService
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.pipline.prediction_pipeline import YoutubeSentimentPredictor
from youtube_sentiment.logger import logging


class YoutubeS3SentimentClassification:

    def __init__(self,bucket_name,model_path):
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model = None

    def is_model_present(self,model_path):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name,s3_key=model_path)
        except Exception as e:
            logging.info(e)
            return False
        
    def load_model(self):

        return self.s3.load_model(self.model_path,bucket_name=self.bucket_name)
    
    def save_model(self,from_file,remove:bool=False):

        try:
            self.s3.upload_file(from_filename=from_file,to_filename=self.model_path,bucket_name=self.bucket_name,remove=remove)
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def predict(self,dataframe: DataFrame):

        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise YoutubeException(e,sys)