from youtube_sentiment.pipline.prediction_pipeline import YoutubeSentimentPredictor
import pandas as pd
from youtube_sentiment.utils.utilities import load_tokenizer


if __name__ == "__main__":


    df = pd.read_csv("artifacts/test/data_ingestion/ingested/test.csv")
    transformer = load_tokenizer("artifacts/test/data_transformation/preprocessed_objects/tokenizer.pkl")
    X = df["text"]
    youtube_sentiment_predictor = YoutubeSentimentPredictor(preprocessing_object=transformer)
    predictions = youtube_sentiment_predictor.predict(X)
    print(predictions)
    