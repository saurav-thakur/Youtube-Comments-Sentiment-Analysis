import os
import sys
import pandas as pd
import json

from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from youtube_sentiment.entity.config_entity import DataValidationConfig
from youtube_sentiment.utils.utilities import read_yaml_file,read_csv_data


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schmea_config = read_yaml_file(data_validation_config.schema_file)
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def validate_number_of_columns(self,df:pd.DataFrame)-> bool:

        try:
            df_columns = df.columns
            missing_columns = []

            for column in self.schmea_config["categorical_columns"]:
                if column not in df_columns:
                    missing_columns.append(column)
            
            if len(missing_columns) > 0:
                logging.info(f"Missing columns are: {missing_columns}")
            
            return False if len(missing_columns) > 0 else True
        
        except Exception as e:
            raise  YoutubeException(e,sys)
        

    def initiate_data_validation(self) -> DataValidationArtifact:

        try:
            validation_error_message = ""
            logging.info("Initiating Data Validation")
            
            train_df = read_csv_data(self.data_ingestion_artifact.train_file_path)
            test_df = read_csv_data(self.data_ingestion_artifact.test_file_path)

            status = self.validate_number_of_columns(train_df)

            if status == False:
                validation_error_message += f"Columns are missing in train data"
            
            status = self.validate_number_of_columns(test_df)

            if status == False:
                validation_error_message += f"Columns are missing in train data"
            

            validation_status = len(validation_error_message) == 0

            if validation_status == False:
                logging.info(f"Validaion Error: {validation_error_message}")
            else:
                validation_error_message = "data validated and no problems with dataset."
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message
            )
            self.data_validation_config.data_validation_dir
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            logging.info(f"writing the status to data validation dir")
            os.makedirs(self.data_validation_config.data_validation_dir,exist_ok=True)
            file_path = os.path.join(self.data_validation_config.data_validation_file)

            json_data = {
                "validation_status":data_validation_artifact.validation_status,
                "message":data_validation_artifact.message
            }
            with open(file_path,"w") as json_file:
                json.dump(json_data, json_file, indent=4)
            
            logging.info(f"data validation completed")

            return data_validation_artifact
        except Exception as e:
            raise YoutubeException(e,sys)
        