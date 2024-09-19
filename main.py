from youtube_sentiment.data_access.database_configuration import push_data_config
from youtube_sentiment.logger import logging
from youtube_sentiment.constants import *
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv
from youtube_sentiment.pipline.training_pipeline import TrainingPipeline
load_dotenv()
from youtube_sentiment.utils.utilities import load_keras_model
from youtube_sentiment.entity.config_entity import ModelTrainerConfig,DataTransformationConfig

if __name__ == "__main__":
    # pushing data to mongodb

    # logging.info("pushing the data to mongo db")
    # data_path = "./dataset/final_dataset/sentiment_analysis_dataset.csv"
    # push_data_config(data_path=data_path)
    # logging.info("pushing the data to mongo db")

    # training pipeline
    training = TrainingPipeline()
    training.run_pipeline()

    # config = ModelTrainerConfig()
    # data = DataTransformationConfig()
    # model = load_keras_model(config.model_trainer_trained_model_name)
    
    # X_test = np.load(data.data_transformation_transformed_test_data)
    # y_test = np.load(data.data_transformation_transformed_train_label)

    # X_train = np.load(data.data_transformation_transformed_train_data)
    # y_train = np.load(data.data_transformation_transformed_train_label)
    
    # X_test = np.load(data.data_transformation_transformed_test_data)
    # y_test = np.load(data.data_transformation_transformed_test_label)


    # print(y_test)


