from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException

import os,sys
import pandas as pd
from sklearn.model_selection import train_test_split

from youtube_sentiment.entity.config_entity import DataIngestionConfig
from youtube_sentiment.entity.artifact_entity import DataIngestionArtifact
from youtube_sentiment.data_access.exporting_data_configuration import YoutubeSentimentData


class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):

        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def export_data_into_feature_store(self)->pd.DataFrame:

        try:
            logging.info("Exporting Data From MongoDB")
            youtube_sentiment_data = YoutubeSentimentData()
            df = youtube_sentiment_data.export_collection_as_dataframe()
            logging.info(f"The exported data is of shape: {df.shape}")

            logging.info("Creating a feature store directory")
            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            # os.makedirs(self.data_ingestion_config.feature_store_file_path,exist_ok=True)
            logging.info("Saving the exported data to feature store.")
            df.to_csv(feature_store_file_path,index=False)
            return df
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def splitting_data_into_train_test(self,df: pd.DataFrame)->None:

        logging.info("Entered splitting_data_into_train_test method of DataIngestion")

        try:
            train,test = train_test_split(df,test_size=0.2,random_state=41)
            logging.info("Creating a ingested directory")
            # os.makedirs(self.data_ingestion_config.training_file_path,exist_ok=True)
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            train.to_csv(self.data_ingestion_config.training_file_path,index=False)
            test.to_csv(self.data_ingestion_config.testing_file_path,index=False)

            logging.info("train and test file saved!!")

        except Exception as e:
            raise YoutubeException(e,sys)
        
    def initiate_data_ingestion(self)->DataIngestionArtifact:

        logging.info("Initiating Data Ingestion")

        try:
            logging.info("Fetching the data from MongoDB")
            df = self.export_data_into_feature_store()
            logging.info("Data Fetched from MongoDB")

            logging.info("Splitting the fetched data.")
            self.splitting_data_into_train_test(df=df)
            logging.info("Data splitted.")

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
                                                            test_file_path=self.data_ingestion_config.testing_file_path)
            
            return data_ingestion_artifact
        except Exception as e:
            raise YoutubeException(e,sys)
        
