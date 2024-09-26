import os,sys
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.components.data_ingestion import DataIngestion
from youtube_sentiment.components.data_validation import DataValidation
from youtube_sentiment.components.data_transformation import DataTransformation
from youtube_sentiment.components.model_trainer import ModelTrainer
from youtube_sentiment.components.model_evaluation import ModelEvaluation
from youtube_sentiment.components.model_pusher import ModelPusher

from youtube_sentiment.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig, ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig
from youtube_sentiment.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()


    def start_data_ingestion(self) ->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("data ingestion of training pipeline completed")
            return data_ingestion_artifact
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact: DataIngestionArtifact)->DataValidationArtifact:

        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("data validation of training pipeline completed")
            return data_validation_artifact
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact,data_validation_artifact: DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(data_transformation_config=self.data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact,
                                                     data_ingestion_artifact=data_ingestion_artifact)

            data_transformation_artifact = data_transformation.initiate_transform_data()

            logging.info("data transformation of training pipeline completed")
            return data_transformation_artifact
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def start_model_training(self,data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:

        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_training()

            logging.info("model trainer of training pipeline completed")
            return model_trainer_artifact
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def start_model_evaluation(self,data_transformation_artifact: DataTransformationArtifact,model_trainer_artifact:ModelTrainerArtifact) ->ModelEvaluationArtifact:

        try:
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact  = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            return YoutubeException(e,sys)
        
    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,model_pusher_config=self.model_pusher_config
            )
            
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise YoutubeException(e,sys)

        
    def run_pipeline(self)->None:
        try:
            logging.info("Data Ingestion Started")
            # data_ingestion_artifact = self.start_data_ingestion()
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info("Data Ingestion Completed")

            logging.info("Data Validation Started")
            # data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=True,
                message="NOT IMPLEMENTED"
            )
            logging.info("Data Validation Completed")


            logging.info("Data Transformation Started")
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                          data_validation_artifact=data_validation_artifact)
            logging.info("Data Transformation Completed")

            logging.info("Model Trainer Started")
            model_trainer_artifact = self.start_model_training(data_transformation_artifact=data_transformation_artifact)
            logging.info("Model Trainer Completed")

            logging.info("Model Evaluation Started")
            model_evaluation_artifact = self.start_model_evaluation(data_transformation_artifact=data_transformation_artifact,model_trainer_artifact=model_trainer_artifact)
            logging.info("Model Evaluation Completed")

            logging.info("LOGGING MODEL EVAL ARTIFACT")
            logging.info(model_evaluation_artifact)

            logging.info("Model Pusher Started")
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            logging.info("Model Pusher Completed")


        except Exception as e:
            raise YoutubeException(e,sys)