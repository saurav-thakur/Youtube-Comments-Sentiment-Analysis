import os


# database
DB_NAME = "YOUTUBE_DATASET"
COLLECTION_NAME = "sentiment_data"
SENTIMENT_ANALYSIS_DATASET = os.path.join(
    "dataset", "final_dataset", "sentiment_analysis_dataset.csv"
)

PIPELINE_NAME = "youtube_sentiment"
ARTIFACT_DIR = "artifacts"

DATSET_FILE_NAME = "sentiment_dataset.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TARGET_COLUMN = "label"
YOUTUBE_DATASET_COLLECTION = os.path.join(
    "dataset", "collecting_data", "youtube_data.csv"
)


# data ingestion constants
DATA_INGESTION_COLLECTION_NAME = "sentiment_data"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

# data validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_FILE_NAME: str = "data_validation.json"


# data transformation constants
CONFIG_FILE_PATH: str = "config"
SCHEMA_FILE_NAME: str = "schema.yaml"
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_DATA_TRAIN: str = "transformed_train.npy"
DATA_TRANSFORMATION_TRANSFORMED_LABEL_TRAIN: str = "transformed_label_train.npy"
DATA_TRANSFORMATION_TRANSFORMED_DATA_TEST: str = "transformed_test.npy"
DATA_TRANSFORMATION_TRANSFORMED_LABEL_TEST: str = "transformed_label_test.npy"
DATA_TRANSFORMATION_PREPROCESSED_OBJECT_DATA_DIR: str = "preprocessed_objects"
DATA_TRANSFORMATION_TOKENIZER_OBJECT: str = "tokenizer.pkl"
DATA_TRANSFORMATION_LABEL_ENCODED_OBJECT: str = "label_encoded.pkl"

DATA_TRANSFORMATION_PAD_SEQUENCES_PADDING = "post"
DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN = 50

DATA_TRANSFORMATION_POSITIVE_SENTIMENT_MAP: int = 1
DATA_TRANSFORMATION_NEGATIVE_SENTIMENT_MAP: int = 0

# model constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_PLOTS_DIR_NAME: str = "plots"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.keras"
MODEL_TRAINER_TRAINED_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_CONFIG_FILE_PATH: str = "model.yaml"
MODEL_TRAINER_TRAINING_EPOCHS: int = 3
MODEL_TRAINER_TRAINING_BATCH_SIZE: int = 64
MODEL_TRAINER_ACCURACY_PLOT: str = "accuracy.png"
MODEL_TRAINER_VALIDATION_PLOT: str = "validation.png"

# model evaluation constants
MODEL_EVALUATION_THRESHOLD_SCORE: float = 0.7
MODEL_BUCKET_NAME: str = "sentiment-model-youtube-analytics"
MODEL_PUSHER_S3_KEY: str = "model-registry"
REGION_NAME = "eu-north-1"

# model prediction constants
MODEL_PREDICTION_FINAL_MODEL: str = "model.keras"
MODEL_PREDICTION_TOKENIZER_OBJECT: str = "tokenizer.pkl"

# FAST API PORT and HOST
PORT: int = 8080
HOST: str = "127.0.0.1"
