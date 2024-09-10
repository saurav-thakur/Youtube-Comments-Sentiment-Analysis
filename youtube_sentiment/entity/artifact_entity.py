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
