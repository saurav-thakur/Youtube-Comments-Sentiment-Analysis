import json
import os,sys
import pandas as pd
import yaml
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException


def read_yaml_file(file_path: str) -> dict:

    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise YoutubeException(e,sys)
    

def write_json_file(file_path: str, json_file: json):
    try:
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(file_path, f, ensure_ascii=False, indent=4)
    except Exception as e:
        raise YoutubeException(e,sys)
    
def read_csv_data(path: str) ->pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise YoutubeException(e,sys)