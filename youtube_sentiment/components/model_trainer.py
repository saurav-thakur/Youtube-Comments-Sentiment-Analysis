import tensorflow as tf
import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from typing import Tuple
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from youtube_sentiment.entity.config_entity import (
    ModelTrainerConfig,
    DataIngestionConfig,
)
from youtube_sentiment.ml.model import train_model, plot_accuracy_and_loss_graph
from youtube_sentiment.utils.utilities import save_keras_model, load_tokenizer
from youtube_sentiment.constants import (
    MODEL_TRAINER_TRAINING_EPOCHS,
    MODEL_TRAINER_TRAINING_BATCH_SIZE,
)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.data_ingestion_config = DataIngestionConfig()

    def model_training(self) -> Tuple[object, object]:

        X_train = np.load(
            self.data_transformation_artifact.data_transformation_transformed_train_data
        )
        y_train = np.load(
            self.data_transformation_artifact.data_transformation_transformed_train_label
        )

        X_test = np.load(
            self.data_transformation_artifact.data_transformation_transformed_test_data
        )
        y_test = np.load(
            self.data_transformation_artifact.data_transformation_transformed_test_label
        )

        tokenizer = load_tokenizer(
            self.data_transformation_artifact.data_transformation_tokenizer
        )

        vocab_size = len(tokenizer.word_index) + 1
        model = train_model(vocab_size=vocab_size)

        history = model.fit(
            X_train,
            y_train,
            epochs=MODEL_TRAINER_TRAINING_EPOCHS,
            batch_size=MODEL_TRAINER_TRAINING_BATCH_SIZE,
            validation_data=(X_test, y_test),
        )
        logging.info("Plotting the accuracy graph")
        plot_accuracy_and_loss_graph(
            history,
            accuracy_filename=self.model_trainer_config.model_trainer_accuracy_plot,
            loss_filename=self.model_trainer_config.model_trainer_validation_plot,
        )

        predictions = model.predict(X_test)
        predictions = tf.argmax(predictions, axis=1)
        print(predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)

        metric_artifact = ClassificationMetricArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall,
            accuracy_score=accuracy,
        )
        logging.info(
            f"The precision is:{precision}, recall is: {recall}, f1 score is: {f1} and accuracy is: {accuracy}"
        )

        return model, metric_artifact

    def initiate_model_training(self) -> ModelTrainerArtifact:

        try:
            logging.info("model training initiated")
            model, metric_artifact = self.model_training()

            if (
                metric_artifact.accuracy_score
                < self.model_trainer_config.model_trainer_expected_score
            ):
                logging.info(
                    "The current trained model is not better than the expected score."
                )
                raise Exception(
                    "The current trained model is not better than the expected score."
                )

            logging.info(
                f"The new model has accuracy of {metric_artifact.accuracy_score}"
            )
            os.makedirs(
                self.model_trainer_config.model_trainer_trained_model, exist_ok=True
            )
            save_keras_model(
                model, self.model_trainer_config.model_trainer_trained_model_name
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.model_trainer_trained_model_name,
                metric_artifact=metric_artifact,
            )

            return model_trainer_artifact

        except Exception as e:
            raise YoutubeException(e, sys)
