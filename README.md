# YouTube Comments Sentiment Analysis

This repository contains the code and resources for performing sentiment analysis on YouTube comments. The project aims to analyze the sentiment (positive, negative, or neutral) expressed in comments to gain insights into viewers' opinions and reactions to specific content.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-folder-structure)
<!-- - [Dataset](#dataset) -->
- [Installation](#installation)
<!-- - [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results) -->
- [License](#license)


## Introduction

Sentiment analysis is a common task in natural language processing (NLP) that involves determining the emotional tone behind a body of text. This project focuses on applying sentiment analysis to YouTube comments, which can be useful for content creators, marketers, and researchers to understand audience feedback.

The project uses machine learning techniques to classify comments into three sentiment categories: positive, negative, and neutral.


## Project Folder Structure

This structure is organized to facilitate clean and modular code, making it easier to manage various components of the project, such as data ingestion, transformation, model training, and evaluation.

#### Folder Structure Overview

```
youtube_sentiment/
│
├── youtube_sentiment/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/
│   │   └── __init__.py
│   │
│   ├── constants/
│   │   └── __init__.py
│   │
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   │
│   ├── exception/
│   │   └── __init__.py
│   │
│   ├── logger/
│   │   └── __init__.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── utilities.py
│
├── tests/
│   ├── __init__.py
│   └── test_app.py
│
├── config/
│   ├── model.yaml
│   └── schema.yaml
│
├── app.py
├── main.py
├── Dockerfile
└── .dockerignore
```

## Project Directory and Files Description

### 1. `youtube_sentiment/`
   - **`__init__.py`**: Initializes the `youtube_sentiment` package.

### 2. `youtube_sentiment/components/`
   - **`__init__.py`**: Initializes the `components` module.
   - **`data_ingestion.py`**: Contains functions and classes for ingesting raw data from various sources.
   - **`data_validation.py`**: Handles validation of the data to ensure it meets the required standards before processing.
   - **`data_transformation.py`**: Contains code for transforming raw data into a format suitable for model training.
   - **`model_trainer.py`**: Responsible for training machine learning models using the processed data.
   - **`model_evaluation.py`**: Includes methods to evaluate the performance of the trained models.
   - **`model_pusher.py`**: Manages the deployment or saving of the trained model to a production environment.

### 3. `youtube_sentiment/configuration/`
   - **`__init__.py`**: Initializes the `configuration` module, which handles configuration management.

### 4. `youtube_sentiment/constants/`
   - **`__init__.py`**: Initializes the `constants` module, which contains project-wide constants.

### 5. `youtube_sentiment/entity/`
   - **`__init__.py`**: Initializes the `entity` module.
   - **`config_entity.py`**: Defines configuration entities that manage configuration settings.
   - **`artifact_entity.py`**: Contains artifact entities that represent outputs at different stages of the pipeline.

### 6. `youtube_sentiment/exception/`
   - **`__init__.py`**: Initializes the `exception` module for custom exception handling.

### 7. `youtube_sentiment/logger/`
   - **`__init__.py`**: Initializes the `logger` module for logging events, errors, and other significant occurrences.

### 8. `youtube_sentiment/pipeline/`
   - **`__init__.py`**: Initializes the `pipeline` module.
   - **`training_pipeline.py`**: Manages the end-to-end training pipeline, coordinating data ingestion, transformation, model training, and evaluation.
   - **`prediction_pipeline.py`**: Manages the prediction pipeline, which uses the trained model to make predictions on new data.

### 9. `youtube_sentiment/utils/`
   - **`__init__.py`**: Initializes the `utils` module.
   - **`utilities.py`**: Contains utility functions that support various components across the project.

### 10. `tests/`
   - **`__init__.py`**: Initializes the `tests` module.
   - **`test_app.py`**: Contains unit tests to validate the functionality of the application.

### 11. `config/`
   - **`model.yaml`**: Configuration file specifying model parameters and settings.
   - **`schema.yaml`**: Schema definition for the dataset, detailing expected data types and structure.

### 12. Project Root Files
   - **`app.py`**: The main entry point for running the application.
   - **`main.py`**: An auxiliary script, potentially for testing or initiating processes.
   - **`Dockerfile`**: Contains instructions to create a Docker image for the application.
   - **`.dockerignore`**: Specifies files and directories to ignore when creating a Docker image.

---

This folder structure ensures that the project is well-organized, modular, and scalable, making it easier to maintain and extend in the future.

## Dataset

The dataset used for this project is collected from YouTube comments. The data is preprocessed to remove noise and irrelevant information. Each comment is labeled with its corresponding sentiment (positive, negative, or neutral).

**Note:** Due to privacy concerns and YouTube's data policy, the dataset is not included in this repository. However, you can collect your own dataset using YouTube's Data API or other scraping tools.

## Installation

To run the project locally, you need to have Python installed. Follow the steps below to set up the environment:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/youtube-comments-sentiment-analysis.git
   cd youtube-comments-sentiment-analysis
   ```

2. **Create a virtual environment:**

   ```bash
   conda create --prefix venv python=3.10.0
   ```

3. **If you want to add any extra packages:**

   ```bash
   pip install yourPackageName
   ```


3. **To remove any packages:**

   ```bash
   pip uninstall yourPackageName
   ```


4. **To deactivate the poetry environment:**

   ```bash
   source deactivate
   ```

<!-- ## Usage

Once the environment is set up, you can use the following scripts and notebooks to perform various tasks:

- **Data Preprocessing:**  
  Use the `data_preprocessing.py` script to clean and preprocess the dataset.

  ```bash
  python scripts/data_preprocessing.py --input data/raw_comments.csv --output data/processed_comments.csv
  ```

- **Model Training:**  
  Train the sentiment analysis model using the `train_model.py` script.

  ```bash
  python scripts/train_model.py --input data/processed_comments.csv --model_output models/sentiment_model.pkl
  ```

- **Inference:**  
  Use the trained model to predict the sentiment of new comments.

  ```bash
  python scripts/predict.py --model models/sentiment_model.pkl --input data/new_comments.csv --output results/predictions.csv
  ```

## Model Training

The project uses a supervised machine learning approach to train a sentiment analysis model. The training process involves the following steps:

1. **Data Splitting:**  
   The dataset is split into training and testing sets.

2. **Feature Extraction:**  
   Text features are extracted using techniques like TF-IDF or word embeddings.

3. **Model Selection:**  
   Various machine learning models (e.g., Logistic Regression, SVM, Random Forest) are evaluated.

4. **Training:**  
   The selected model is trained on the training data.

5. **Hyperparameter Tuning:**  
   Hyperparameters are optimized using techniques like Grid Search or Random Search.

## Evaluation

The model's performance is evaluated using the testing set. Key evaluation metrics include:

- **Accuracy:** The percentage of correct predictions.
- **Precision:** The number of true positive results divided by the number of positive results predicted by the model.
- **Recall:** The number of true positive results divided by the number of positives that should have been predicted.
- **F1 Score:** The harmonic mean of precision and recall.

Evaluation results are saved in the `results/` directory for further analysis.

## Results

The results of the sentiment analysis, including confusion matrices, classification reports, and visualizations, are documented in the `results/` directory. You can review these outputs to understand the model's strengths and weaknesses. -->

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.