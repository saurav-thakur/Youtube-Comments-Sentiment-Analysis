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


def retain_youtube_csv_files(df: pd.DataFrame, prediction_probab, predictions):
    logging.info("Retaining youtube comments as CSV File")
    try:
        # Remove empty rows from the DataFrame before processing
        df = df.dropna(subset=["text"])
        logging.info(f"Number of rows after dropping empty text rows: {df.shape[0]}")

        if os.path.isfile(YOUTUBE_DATASET_COLLECTION):
            logging.info("Already have an old youtube CSV File")

            # Load old CSV and annotate new labels
            old_df = pd.read_csv(YOUTUBE_DATASET_COLLECTION)
            old_df = old_df.dropna(subset=["text"])  # Ensure old file has no empty rows
            logging.info(f"Number of rows in old CSV: {old_df.shape[0]}")

            labelled_df = annotate_label(prediction_probab, predictions, df)

            # Combine old and new DataFrames
            final_df = pd.concat([old_df, labelled_df], axis=0)

            # Drop duplicates based on the 'text' column while keeping the first occurrence
            final_df = final_df.drop_duplicates(
                subset="text", keep="first", ignore_index=True
            )
            logging.info(
                f"Number of rows after dropping duplicates: {final_df.shape[0]}"
            )

            final_df.to_csv(YOUTUBE_DATASET_COLLECTION, index=False)
            logging.info("Combined and saved youtube CSV File")
        else:
            logging.info("No old youtube CSV File. Creating a new one")
            dir_name = os.path.dirname(YOUTUBE_DATASET_COLLECTION)
            os.makedirs(dir_name, exist_ok=True)

            labelled_df = annotate_label(prediction_probab, predictions, df)
            labelled_df.to_csv(YOUTUBE_DATASET_COLLECTION, index=False)
            logging.info("Annotated and saved the data")
    except Exception as e:
        raise YoutubeException(e, sys)


def annotate_label(prediction_probab, predictions, df):
    try:
        if "label" not in df.columns:
            df["label"] = None

        logging.info(f"Applying labels to {len(predictions)} rows")

        # Iterate through each row to apply labels
        for i in range(len(predictions)):
            if predictions[i] == 0:
                if prediction_probab[i][0] > 0.8:
                    df.loc[i, "label"] = 0
                elif prediction_probab[i][1] > 0.8:
                    df.loc[i, "label"] = 1
            elif predictions[i] == 1 and prediction_probab[i][1] > 0.8:
                df.loc[i, "label"] = 1

        logging.info(f"Number of rows before dropping unlabeled: {df.shape[0]}")

        # Drop rows where 'label' is None (NaN)
        df_dropped = df.dropna(subset=["label"])

        logging.info(f"Number of rows after dropping unlabeled: {df_dropped.shape[0]}")

        # Reset index after dropping rows
        df_reset = df_dropped.reset_index(drop=True)
        return df_reset
    except Exception as e:
        raise YoutubeException(e, sys)
