

# database
DB_NAME = "YOUTUBE_DATASET"
COLLECTION_NAME = "sentiment_data"

PIPELINE_NAME = "youtube_sentiment"
ARTIFACT_DIR = "artifacts"

DATSET_FILE_NAME = "sentiment_dataset.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TARGET_COLUMN = "label"


# data ingestion constants
DATA_INGESTION_COLLECTION_NAME = "sentiment_data"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

# data validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"


# data transformation constants
CONFIG_FILE_PATH: str = "config"
SCHEMA_FILE_NAME: str = "schema.yaml"


# model constants
