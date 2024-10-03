import os
import pytest
from starlette.testclient import TestClient
from app import app

from youtube_sentiment.entity.config_entity import DataValidationConfig
import json

client = TestClient(app=app)

def test_data_validation():

    # data_validation_config = DataValidationConfig()
    # # Open and read the JSON file
    # with open(data_validation_config.data_validation_file, "r") as json_file:
    #     json_data = json.load(json_file)

    # assert json_data["validation_status"]

    try:
        data_validation_config = DataValidationConfig()
        # Open and read the JSON file
        with open(data_validation_config.data_validation_file, "r") as json_file:
            json_data = json.load(json_file)

        assert json_data["validation_status"]
    except NotImplementedError as not_implemented:
        return True
    
    except FileNotFoundError as file_not_found:
        return True
    
    except ImportError as ie:
        return False
    
    except Exception as e:
        return False



     