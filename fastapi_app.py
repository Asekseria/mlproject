from fastapi import FastAPI
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()

predictor = PredictPipeline()

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API"}

@app.post("/predict")
def predict(data: dict):

    prediction = predictor.predict(data)

    return {"Survival Prediction": prediction}