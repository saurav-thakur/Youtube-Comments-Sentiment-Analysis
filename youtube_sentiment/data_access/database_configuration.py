from dotenv import load_dotenv
import os
from youtube_sentiment.constants import DB_NAME,COLLECTION_NAME
import pymongo
import pandas as pd
from youtube_sentiment.logger import logging
load_dotenv()


class DatabaseConfig:
    def __init__(self,dataset):
        self.dataset = dataset
        self.client = pymongo.MongoClient(os.getenv("MONGO_DB_CONNECTION_URL"))
        self.database = self.client[DB_NAME]
        self.collection = self.database[COLLECTION_NAME]

    def push_data(self):
        if COLLECTION_NAME in self.database.list_collection_names() and self.collection.count_documents({}) > 0:
            logging.info(f"Collection '{COLLECTION_NAME}' already exists and has data. Data push aborted.")
        else:
            self.collection.insert_many(self.dataset)
            logging.info(f"Data pushed to collection '{COLLECTION_NAME}'.")


def push_data_config(data_path):
    dataset = pd.read_csv(data_path)
    data = dataset.to_dict(orient="records")
    db_config = DatabaseConfig(dataset=data)
    db_config.push_data()