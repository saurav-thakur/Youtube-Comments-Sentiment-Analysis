

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
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_DATA_TRAIN: str = "transformed_train.npy"
DATA_TRANSFORMATION_TRANSFORMED_LABEL_TRAIN: str = "transformed__label_train.npy"
DATA_TRANSFORMATION_TRANSFORMED_DATA_TEST: str = "transformed_label_test.npy"
DATA_TRANSFORMATION_TRANSFORMED_LABEL_TEST: str = "transformed_test.npy"
DATA_TRANSFORMATION_PREPROCESSED_OBJECT_DATA_DIR: str = "preprocessed_objects"
DATA_TRANSFORMATION_TOKENIZER_OBJECT: str = "tokenizer.pkl"
DATA_TRANSFORMATION_LABEL_ENCODED_OBJECT: str = "label_encoded.pkl"

DATA_TRANSFORMATION_PAD_SEQUENCES_PADDING = "post"
DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN = 50

DATA_TRANSFORMATION_POSITIVE_SENTIMENT_MAP: str = 1
DATA_TRANSFORMATION_NEGATIVE_SENTIMENT_MAP: str = 0

# model constants
