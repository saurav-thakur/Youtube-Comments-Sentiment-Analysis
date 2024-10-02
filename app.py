import pandas as pd
from youtube_sentiment.pipline.training_pipeline import TrainingPipeline
from youtube_sentiment.pipline.prediction_pipeline import PredictionPipeline
from youtube_sentiment.data_access.database_configuration import push_data_config

from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import pandas as pd

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
async def predict(text: str):
    try:
        # Create a DataFrame with the input text
        df = pd.DataFrame({'text': [text]})
        print("DataFrame created:", df)

        # Run prediction
        prediction_pipeline = PredictionPipeline()
        predicted_class = prediction_pipeline.predict(df)
        print("Predicted class:", predicted_class)

        # Return the prediction in JSON format
        return JSONResponse(content={
            "prediction": "negative" if predicted_class == 0 else "positive",
        })

    except Exception as e:
        # Handle any exceptions and return an error message
        return JSONResponse(status_code=500, content={
            "error": str(e)
        })
    
@app.get("/api/v1/push_to_mongo")
async def push_data_to_mongo():
    try:
        # pushing data to mongodb

        data_path = "./dataset/final_dataset/sentiment_analysis_dataset.csv"
        push_data_config(data_path=data_path)
    except Exception as e:
            return JSONResponse(status_code=500, content={
            "error": str(e)
        })


if __name__ == "__main__":
    uvicorn.run("app:app",host="localhost",port=8080,reload=True)