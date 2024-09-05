import os,sys
import pymongo
import certifi
from dotenv import load_dotenv

from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.constants import DB_NAME,COLLECTION_NAME



load_dotenv()

class MongoDBClient:

    """
    exports data into feature store
    """
    client = None

    def __init__(self,database_name=DB_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("MONGO_DB_CONNECTION_URL")
                if mongo_db_url is None:
                    raise YoutubeException("Environment Key is not setup.",sys)
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.collection = self.database[COLLECTION_NAME]
            self.database_name = database_name
            logging.info("MongoDB connection sucessfull.")

        except Exception as e:
            raise YoutubeException(e,sys)