from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

# load trained model
# model = joblib.load("src/models/model.pkl")
predictor = PredictPipeline()

app = FastAPI()

# Input schema using Pydantic
class Passenger(BaseModel):
    pclass: int
    sex: str
    age: float
    fare: float
    embarked: str

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API"}

@app.post("/predict")
def predict(data: Passenger):

    # convert input to dataframe
    input_data = pd.DataFrame([data.dict()])

    # prediction
    prediction = predictor.predict(input_data)[0]

    return {"prediction": int(prediction)}