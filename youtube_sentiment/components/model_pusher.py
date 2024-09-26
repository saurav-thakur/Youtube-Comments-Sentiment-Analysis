import os
import sys
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from youtube_sentiment.entity.config_entity import ModelPusherConfig
from youtube_sentiment.cloud_storage.aws_storage import SimpleStorageService
from youtube_sentiment.entity.s3_estimator import YoutubeS3SentimentClassification

class ModelPusher:

    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact, model_pusher_config: ModelPusherConfig):
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.youtube_sentiment_estimator = YoutubeS3SentimentClassification(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered the initiate_model_pusher method")

        try:
            # Log all attributes of model_evaluation_artifact
            logging.info(f"ModelEvaluationArtifact attributes: {vars(self.model_evaluation_artifact)}")

            # Check if trained_model_path exists and is not None
            if not hasattr(self.model_evaluation_artifact, 'trained_model_path'):
                raise AttributeError("trained_model_path not found in ModelEvaluationArtifact")
            
            if self.model_evaluation_artifact.trained_model_path is None:
                raise ValueError("trained_model_path is None in ModelEvaluationArtifact")

            # Check if the file exists
            if not os.path.exists(self.model_evaluation_artifact.trained_model_path):
                raise FileNotFoundError(f"Trained model file not found at {self.model_evaluation_artifact.trained_model_path}")

            logging.info(f"Uploading model from {self.model_evaluation_artifact.trained_model_path} to S3 bucket")
            self.youtube_sentiment_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info("Uploaded artifacts folder to S3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except AttributeError as ae:
            raise YoutubeException(f"ModelEvaluationArtifact error: {str(ae)}", sys) from ae
        except ValueError as ve:
            raise YoutubeException(f"Invalid value in ModelEvaluationArtifact: {str(ve)}", sys) from ve
        except FileNotFoundError as fnf:
            raise YoutubeException(f"File error: {str(fnf)}", sys) from fnf
        except Exception as e:
            raise YoutubeException(f"Error in ModelPusher: {str(e)}", sys) from e