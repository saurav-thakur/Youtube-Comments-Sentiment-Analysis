# YouTube Comments Sentiment Analysis

This repository contains the code and resources for performing sentiment analysis on YouTube comments. The project aims to analyze the sentiment (positive, negative, or neutral) expressed in comments to gain insights into viewers' opinions and reactions to specific content.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results) -->
- [License](#license)

## Introduction

Sentiment analysis is a common task in natural language processing (NLP) that involves determining the emotional tone behind a body of text. This project focuses on applying sentiment analysis to YouTube comments, which can be useful for content creators, marketers, and researchers to understand audience feedback.

The project uses machine learning techniques to classify comments into three sentiment categories: positive, negative, and neutral.

## Dataset

The dataset used for training is basically mixture of different sentiment dataset.

1. [IMDB dataset (Sentiment analysis) in CSV format](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format)
1. [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
1. [Sentiment Analysis Evaluation Dataset](https://www.kaggle.com/datasets/prishasawhney/sentiment-analysis-evaluation-dataset)

To make sure the dataset consists of various sources, this project combines data from various sources. The data is preprocessed to remove noise and irrelevant information. Each comment is labeled with its corresponding sentiment (positive and negative). The neutral class is removed for now and will be incorporated in the future.

## Project Folder Structure

This structure is organized to facilitate clean and modular code, making it easier to manage various components of the project, such as data ingestion, transformation, model training, and evaluation.

#### Folder Structure Overview

```
├── .dockerignore
├── .gitignore
├── app.py
├── Dockerfile
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
├── setup.py
├── .github/
│   └── workflows/
│       └── action.yaml
├── artifacts/
│
├── config/
│   ├── model.yaml
│   └── schema.yaml
├── dataset/
│   └── final_dataset/
│       └── sentiment_analysis_dataset.csv
├── notebooks/
│   ├── 01_experiment.ipynb
│   ├── 02_exploring_data.ipynb
│   ├── 03_model_building_ml.ipynb
│   └── helper.py
├── tests/
│   ├── test_app.py
│   └── __init__.py
└── youtube_sentiment/
    ├── __init__.py
    ├── cloud_storage/
    │   ├── aws_storage.py
    │   └── __init__.py
    ├── components/
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── data_validation.py
    │   ├── model_evaluation.py
    │   ├── model_pusher.py
    │   ├── model_trainer.py
    │   └── __init__.py
    ├── configuration/
    │   ├── aws_connection.py
    │   ├── mongo_db_connection.py
    │   └── __init__.py
    ├── constants/
    │   └── __init__.py
    ├── data_access/
    │   ├── database_configuration.py
    │   ├── exporting_data_configuration.py
    │   ├── extracting_data_from_youtube.py
    │   └── __init__.py
    ├── entity/
    │   ├── artifact_entity.py
    │   ├── config_entity.py
    │   ├── s3_estimator.py
    │   └── __init__.py
    ├── exception/
    │   └── __init__.py
    ├── logger/
    │   └── __init__.py
    ├── ml/
    │   ├── model.py
    │   └── __init__.py
    ├── pipline/
    │   ├── prediction_pipeline.py
    │   ├── training_pipeline.py
    │   └── __init__.py
    └── utils/
        ├── utilities.py
        └── __init__.py

```

## Project Directory and Files Description

### 1. `youtube_sentiment/`

- **`__init__.py`**: Initializes the `youtube_sentiment` package.
  Here’s a detailed breakdown of the files and their functions, similar to the format you requested:

### 1. **Project Root Files**

- **`.dockerignore`**: Lists files and directories to exclude from the Docker image build process.
- **`.gitignore`**: Specifies files and directories to be ignored by Git for version control.
- **`app.py`**: The main application entry point, likely running the FastAPI application.
- **`Dockerfile`**: Instructions to build a Docker image for the application, specifying dependencies and build steps.
- **`LICENSE`**: Contains the legal license under which the project is distributed.
- **`main.py`**: Likely used as an auxiliary script for specific tasks or running services.
- **`README.md`**: A markdown file providing an overview of the project, including instructions for setup, usage, and contribution.
- **`requirements.txt`**: Lists all Python dependencies required to run the project.
- **`setup.py`**: A Python script used to package the project, making it installable as a Python package.

### 2. **`.github/workflows/`**

- **`action.yaml`**: Defines GitHub Actions for automating tasks such as testing, building, or deployment.

### 3. **`artifacts/training_artifacts/`**

Stores the results of the data validation process, including information about data consistency and integrity.

### 4. **`config/`**

- **`model.yaml`**: Configuration file that specifies parameters and settings for the machine learning model.
- **`schema.yaml`**: Defines the structure and data types expected in the dataset, ensuring data validation.

### 5. **`dataset/final_dataset/`**

- **`sentiment_analysis_dataset.csv`**: The final dataset used for training and evaluating the sentiment analysis model, likely containing text data and labels.

### 6. **`notebooks/`**

- **`01_experiment.ipynb`**: Jupyter notebook for initial experimentation and exploratory analysis.
- **`02_exploring_data.ipynb`**: Notebook focused on exploring and visualizing the dataset.
- **`03_model_building_ml.ipynb`**: Notebook for building and training the machine learning model.
- **`helper.py`**: A helper script containing utility functions used across the notebooks.

### 7. **`tests/`**

- **`__init__.py`**: Initializes the `tests` module.
- **`test_app.py`**: Contains unit tests to validate the core functionality of the application.

### 8. **`youtube_sentiment/`**

#### 8.1. `cloud_storage/`

- **`aws_storage.py`**: Handles operations related to uploading and retrieving data from AWS S3.
- **`__init__.py`**: Initializes the `cloud_storage` module.

#### 8.2. `components/`

- **`__init__.py`**: Initializes the `components` module.
- **`data_ingestion.py`**: Responsible for ingesting raw data, possibly from sources like YouTube or APIs.
- **`data_validation.py`**: Validates the ingested data, checking for errors, missing values, or inconsistencies.
- **`data_transformation.py`**: Transforms the raw data into a structured format ready for machine learning.
- **`model_trainer.py`**: Contains code to train the machine learning model using the preprocessed data.
- **`model_evaluation.py`**: Evaluates the model’s performance using metrics like accuracy, precision, recall, etc.
- **`model_pusher.py`**: Handles the deployment of the trained model to production environments.

#### 8.3. `configuration/`

- **`aws_connection.py`**: Manages connections to AWS S3 services.
- **`mongo_db_connection.py`**: Manages the connection to a MongoDB database.
- **`__init__.py`**: Initializes the `configuration` module.

#### 8.4. `constants/`

- **`__init__.py`**: Initializes the `constants` module, which contains project-wide constants.

#### 8.5. `data_access/`

- **`database_configuration.py`**: Manages configurations related to database connections.
- **`exporting_data_configuration.py`**: Handles configurations for exporting data.
- **`extracting_data_from_youtube.py`**: Extracts data from YouTube, using the YouTube API.
- **`__init__.py`**: Initializes the `data_access` module.

#### 8.6. `entity/`

- **`artifact_entity.py`**: Defines entities representing artifacts generated during different stages of the ML pipeline.
- **`config_entity.py`**: Defines configuration entities for managing various configuration settings.
- **`s3_estimator.py`**: Contains logic for loading,saving and predicting operations related to AWS S3.
- **`__init__.py`**: Initializes the `entity` module.

#### 8.7. `exception/`

- **`__init__.py`**: Initializes the `exception` module for custom exception handling.

#### 8.8. `logger/`

- **`__init__.py`**: Initializes the `logger` module, which provides logging capabilities for tracking events and errors.

#### 8.9. `ml/`

- **`model.py`**: Contains the machine learning model architecture and related functionality.
- **`__init__.py`**: Initializes the `ml` module.

#### 8.10. `pipline/`

- **`prediction_pipeline.py`**: Manages the process of making predictions with the trained model.
- **`training_pipeline.py`**: Manages the end-to-end process of training the machine learning model, from data ingestion to evaluation.
- **`__init__.py`**: Initializes the `pipline` module.

#### 8.11. `utils/`

- **`utilities.py`**: Contains utility functions that are used across the project for common operations like file handling or logging.
- **`__init__.py`**: Initializes the `utils` module.

This folder structure ensures that the project is well-organized, modular, and scalable, making it easier to maintain and extend in the future.

## Installation

To run the project locally, you need to have Python installed. Follow the steps below to set up the environment:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/youtube-comments-sentiment-analysis.git
   ```

   ```bash
   cd youtube-comments-sentiment-analysis
   ```

2. **Create a virtual environment:**

   ```bash
   conda create --prefix .venv python=3.11.2 -y
   ```

3. **Activate your environment**

   ```bash
   conda activate .venv
   ```

4. **If you want to add any extra packages:**

   ```bash
   pip install -r requirements.txt
   ```

5. **If you want to add any extra packages:**

   ```bash
   pip install yourPackageName
   ```

6. **To remove any packages:**

   ```bash
   pip uninstall yourPackageName
   ```

7. **To deactivate the conda environment:**

   ```bash
   conda deactivate
   ```

## Usage

To run the fast api server. Run the following command.

```
python app.py
```

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.
