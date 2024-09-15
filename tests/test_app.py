from youtube_sentiment.entity.config_entity import DataValidationConfig
import json

def test_data_validation():
    data_validation_config = DataValidationConfig()
    # Open and read the JSON file
    with open(data_validation_config.data_validation_file, "r") as json_file:
        json_data = json.load(json_file)

    assert json_data["validation_status"]


     