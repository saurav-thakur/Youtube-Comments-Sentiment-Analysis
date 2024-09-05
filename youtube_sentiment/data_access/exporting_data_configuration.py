from youtube_sentiment.configuration.mongo_db_connection import MongoDBClient
from youtube_sentiment.constants import DB_NAME
from youtube_sentiment.exception import YoutubeException

import pandas as pd
import numpy as np
import sys,os
from typing import Optional

class YoutubeSentimentData:
    def __init__(self):

        try:
            self.mongo_client = MongoDBClient(database_name=DB_NAME)
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def export_collection_as_dataframe(self)->pd.DataFrame:
        try:
            collection = self.mongo_client.collection
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df.drop("_id",axis=1,inplace=True)
            df.dropna(inplace=True)
            return df
        
        except Exception as e:
            raise YoutubeException(e,sys)