from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str

@dataclass
class DataTransformationArtifact:
    data_transformation_transformed_train_data: str
    data_transformation_transformed_test_data: str
    data_transformation_transformed_train_label: str
    data_transformation_transformed_test_label: str
    data_transformation_tokenizer: str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    accuracy_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: ClassificationMetricArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    curr_and_prod_model_accuracy_difference: float
    s3_model_path: str
    trained_model_path: str

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str