from youtube_sentiment.data_access.database_configuration import push_data_config
from youtube_sentiment.logger import logging

import pandas as pd
import os
from dotenv import load_dotenv
import tensorflow as tf
load_dotenv()

# if __name__ == "__main__":
#     logging.info("pushing the data to mongo db")
#     data_path = "./dataset/final_dataset/sentiment_analysis_dataset.csv"
#     push_data_config(data_path=data_path)
#     logging.info("pushing the data to mongo db")