import pandas as pd
from youtube_sentiment.pipline.training_pipeline import TrainingPipeline
from youtube_sentiment.pipline.prediction_pipeline import PredictionPipeline
from youtube_sentiment.data_access.database_configuration import push_data_config
from youtube_sentiment.data_access.extracting_data_from_youtube import fetch_comments
from youtube_sentiment.constants import (
    SENTIMENT_ANALYSIS_DATASET,
    DATA_TRANSFORMATION_NEGATIVE_SENTIMENT_MAP,
    DATA_TRANSFORMATION_POSITIVE_SENTIMENT_MAP,
    PORT,
    HOST,
)

from youtube_sentiment.utils.utilities import retain_youtube_csv_files

from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import pandas as pd
from urllib.parse import urlparse
from collections import Counter

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/train")
async def training():
    try:
        training = TrainingPipeline()
        training.run_pipeline()
        return Response("Training Sucessfull")
    except Exception as e:
        return Response(f"Error Occured :{e}")


@app.get("/api/v1/predict")
async def predict(youtube_url: str):
    try:
        # Validate URL using urllib.parse
        parsed_url = urlparse(youtube_url)

        # Check if scheme and netloc are present (basic URL validation)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            # raise HTTPException(status_code=400, detail="Invalid URL")

            return JSONResponse(status_code=400, content={"error": "Invalid URL"})

        if "/shorts/" in youtube_url:  # For YouTube Shorts
            video_id = youtube_url.split("/shorts/")[
                1
            ]  # Split on '/shorts/' and take the remaining part
        elif "watch?v=" in youtube_url:  # For standard YouTube videos
            video_id = youtube_url.split("=")[
                1
            ]  # Split on 'v=' and take the first part before any '&'
        else:
            return JSONResponse(
                status_code=500, content={"comments": "no comments in the video"}
            )

        df = fetch_comments(videoId=video_id)

        prediction_pipeline = PredictionPipeline()
        prediction, predicted_class = prediction_pipeline.predict(df)
        prediction_np = prediction.numpy()
        prediction_list = prediction_np.tolist()  # Convert to Python list

        retain_youtube_csv_files(df)
        # Now use the list with Counter
        counter = Counter(prediction_list)

        # # Define the mapping for class labels
        # label_map = {
        #     DATA_TRANSFORMATION_NEGATIVE_SENTIMENT_MAP: "Negative",
        #     DATA_TRANSFORMATION_POSITIVE_SENTIMENT_MAP: "Positive",
        # }

        # # Apply the mapping to the counter using a dictionary comprehension
        # mapped_counter = {label_map[k]: v for k, v in counter.items()}

        # Return the prediction in JSON format
        return {
            "Analysis": f"{counter[DATA_TRANSFORMATION_POSITIVE_SENTIMENT_MAP]} people are talking positive about this video while {counter[DATA_TRANSFORMATION_NEGATIVE_SENTIMENT_MAP]} people are talking negative.",
            "Percentage": f"{round((counter[DATA_TRANSFORMATION_POSITIVE_SENTIMENT_MAP]/len(prediction_list)) * 100,2)}% of people are positive while {round((counter[DATA_TRANSFORMATION_NEGATIVE_SENTIMENT_MAP]/len(prediction_list)) * 100,2)}% are negative.",
        }

    except Exception as e:
        # Handle any exceptions and return an error message
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/v1/push_to_mongo")
async def push_data_to_mongo():
    try:
        # pushing data to mongodb
        data_path = SENTIMENT_ANALYSIS_DATASET
        push_data_config(data_path=data_path)
        return JSONResponse(content={"message": "data sucessfully pushed to mongodb"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
