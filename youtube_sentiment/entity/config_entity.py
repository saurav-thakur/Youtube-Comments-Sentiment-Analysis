import os
from youtube_sentiment.constants import *
from dataclasses import dataclass


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, "training_artifacts")


training_pipline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipline_config.artifact_dir, DATA_INGESTION_DIR_NAME
    )
    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, DATSET_FILE_NAME
    )
    training_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME
    )
    testing_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME
    )
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
    )
    data_validation_file: str = os.path.join(
        training_pipline_config.artifact_dir,
        DATA_VALIDATION_DIR_NAME,
        DATA_VALIDATION_FILE_NAME,
    )
    schema_file: str = os.path.join(CONFIG_FILE_PATH, SCHEMA_FILE_NAME)


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
    )
    data_transformation_transformed_train_data: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_TRAIN,
    )
    data_transformation_transformed_test_data: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_TEST,
    )
    data_transformation_transformed_train_label: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        DATA_TRANSFORMATION_TRANSFORMED_LABEL_TRAIN,
    )
    data_transformation_transformed_test_label: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        DATA_TRANSFORMATION_TRANSFORMED_LABEL_TEST,
    )
    data_transformation_preprocessed_tokenizer: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_PREPROCESSED_OBJECT_DATA_DIR,
        DATA_TRANSFORMATION_TOKENIZER_OBJECT,
    )
    data_transformation_preprocessed_label_encoded: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_PREPROCESSED_OBJECT_DATA_DIR,
        DATA_TRANSFORMATION_LABEL_ENCODED_OBJECT,
    )
    schema_file: str = os.path.join(CONFIG_FILE_PATH, SCHEMA_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    model_trainer_dir_name: str = os.path.join(
        training_pipline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
    )
    model_trainer_trained_model: str = os.path.join(
        model_trainer_dir_name, MODEL_TRAINER_TRAINED_MODEL_DIR
    )
    model_trainer_config_file: str = os.path.join(
        CONFIG_FILE_PATH, MODEL_TRAINER_CONFIG_FILE_PATH
    )
    model_trainer_trained_model_name: str = os.path.join(
        model_trainer_trained_model, MODEL_TRAINER_TRAINED_MODEL_NAME
    )
    model_trainer_expected_score = MODEL_TRAINER_TRAINED_EXPECTED_SCORE
    model_trainer_accuracy_plot = os.path.join(
        training_pipline_config.artifact_dir,
        MODEL_TRAINER_PLOTS_DIR_NAME,
        MODEL_TRAINER_ACCURACY_PLOT,
    )
    model_trainer_validation_plot = os.path.join(
        training_pipline_config.artifact_dir,
        MODEL_TRAINER_PLOTS_DIR_NAME,
        MODEL_TRAINER_VALIDATION_PLOT,
    )


@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_TRAINER_TRAINED_MODEL_NAME


@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_TRAINER_TRAINED_MODEL_NAME


@dataclass
class YoutubeSentimentPredictorConfig:
    model_file_path: str = MODEL_TRAINER_TRAINED_MODEL_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME


@dataclass
class ModelPredictionConfig:
    model_prediction_tokenizer: str = os.path.join(
        training_pipline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR_NAME,
        DATA_TRANSFORMATION_PREPROCESSED_OBJECT_DATA_DIR,
        MODEL_PREDICTION_TOKENIZER_OBJECT,
    )
    model_prediction_final_model: str = os.path.join(
        training_pipline_config.artifact_dir,
        MODEL_TRAINER_DIR_NAME,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_PREDICTION_FINAL_MODEL,
    )
