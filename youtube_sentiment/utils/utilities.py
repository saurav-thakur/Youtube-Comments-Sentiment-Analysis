import json
import os, sys
import pandas as pd
import yaml
import pickle
import tensorflow as tf
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.constants import YOUTUBE_DATASET_COLLECTION


def read_yaml_file(file_path: str) -> dict:

    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise YoutubeException(e, sys)


def write_json_file(file_path: str, json_file: json):
    try:
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(file_path, f, ensure_ascii=False, indent=4)
    except Exception as e:
        raise YoutubeException(e, sys)


def read_csv_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise YoutubeException(e, sys)


def save_preprocessed_object(
    preprocessed_object_path: str, preprocessed_object
) -> None:
    with open(preprocessed_object_path, "wb") as handle:
        pickle.dump(preprocessed_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def save_keras_model(model, path):
    model.save(path)


def load_keras_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def retain_youtube_csv_files(df: pd.DataFrame, labels):
    logging.info("Retaining youtube comments as CSV File")
    try:
        if os.path.isfile(YOUTUBE_DATASET_COLLECTION):
            logging.info("Already have a old youtube CSV File")

            old_df = pd.read_csv(YOUTUBE_DATASET_COLLECTION)
            final_df = pd.concat([old_df, df], axis="rows")
            final_df.drop_duplicates("text", keep="first", ignore_index=True)
            final_df.to_csv(YOUTUBE_DATASET_COLLECTION, index=False)
            logging.info("Combined and saved youtube CSV File")
        else:
            logging.info("No old youtube CSV File. Creating new one")
            dir_name = os.path.dirname(YOUTUBE_DATASET_COLLECTION)
            os.makedirs(dir_name, exist_ok=True)
            df.drop_duplicates("text", keep="first", ignore_index=True)
            df.to_csv(YOUTUBE_DATASET_COLLECTION, index=False)
    except Exception as e:
        raise YoutubeException(e, sys)
