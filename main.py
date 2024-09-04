from youtube_sentiment.data_access.database_configuration import push_data_config
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    data_path = "./dataset/final_dataset/sentiment_analysis_dataset.csv"
    push_data_config(data_path=data_path)