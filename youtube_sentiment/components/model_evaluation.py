from youtube_sentiment.entity.config_entity import ModelEvaluationConfig
from youtube_sentiment.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, DataTransformationArtifact, EvaluateModelResponse
from sklearn.metrics import f1_score
import tensorflow as tf
from youtube_sentiment.entity.s3_estimator import YoutubeS3SentimentClassification
from dataclasses import dataclass
from youtube_sentiment.exception import YoutubeException
import sys
import pandas as pd
import numpy as np
from youtube_sentiment.constants import TARGET_COLUMN
from youtube_sentiment.components.data_transformation import DataTransformation
from youtube_sentiment.logger import logging


class ModelEvaluation:
    def __init__(self,model_eval_config: ModelEvaluationConfig, 
                 data_transformation_artifact: DataTransformationArtifact,model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def get_best_model(self):

        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            youtube_sentiment_estimator = YoutubeS3SentimentClassification(bucket_name=bucket_name,model_path=model_path)

            if youtube_sentiment_estimator.is_model_present(model_path=model_path):
                return youtube_sentiment_estimator
            return None
            
        except Exception as e:
            raise YoutubeException(e,sys)
    
    def evaluate_model(self)->EvaluateModelResponse:

        try:
            X = np.load(self.data_transformation_artifact.data_transformation_transformed_test_data)
            y = np.load(self.data_transformation_artifact.data_transformation_transformed_test_label)
            
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score = None
            best_model = self.get_best_model()

            if best_model is not None:
                y_hat_best_model = tf.argmax(best_model.predict(X),axis=1)
                best_model_f1_score = f1_score(y,y_hat_best_model)
            
            if best_model_f1_score is None:
                temp_best_model_score = 0

            result =  EvaluateModelResponse(
                trained_model_f1_score = trained_model_f1_score,
                best_model_f1_score = best_model_f1_score,
                is_model_accepted = trained_model_f1_score > temp_best_model_score,
                difference = trained_model_f1_score - temp_best_model_score
            )

            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            raise YoutubeException(e,sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                curr_and_prod_model_accuracy_difference=evaluate_model_response.difference,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise YoutubeException(e,sys)
        
    