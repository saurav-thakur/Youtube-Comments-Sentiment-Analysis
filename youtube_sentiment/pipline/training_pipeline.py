import os,sys
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.components.data_ingestion import DataIngestion
from youtube_sentiment.entity.config_entity import DataIngestionConfig
from youtube_sentiment.entity.artifact_entity import DataIngestionArtifact


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) ->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("data ingestion of training pipeline completed")
            return data_ingestion_artifact
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def run_pipeline(self)->None:
        try:
            logging.info("Data Ingestion Started.")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data Ingestion Completed.")
        except Exception as e:
            raise YoutubeException(e,sys)