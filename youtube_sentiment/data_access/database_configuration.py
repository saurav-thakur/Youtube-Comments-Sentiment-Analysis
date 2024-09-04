from dotenv import load_dotenv
import os
from youtube_sentiment.constants import DB_NAME,COLLECTION_NAME
import pymongo
import pandas as pd

load_dotenv()


class DatabaseConfig:
    def __init__(self,dataset):
        self.dataset = dataset
        self.client = pymongo.MongoClient(os.getenv("MONGO_DB_CONNECTION_URL"))
        self.database = self.client[DB_NAME]
        self.collection = self.database[COLLECTION_NAME]

    def push_data(self):
        records = self.collection.insert_many(self.dataset)


def push_data_config(data_path):
    dataset = pd.read_csv(data_path)
    data = dataset.to_dict(orient="records")
    db_config = DatabaseConfig(dataset=data)
    db_config.push_data()