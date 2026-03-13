from flask import Flask, request, jsonify
import joblib
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

predictor = PredictPipeline()

@app.route("/")
def home():
    return "ML API Running"

@app.route("/predict")
def predict():

    pclass = int(request.args.get("pclass"))
    sex = request.args.get("sex")
    age = float(request.args.get("age"))
    fare = float(request.args.get("fare"))
    embarked = request.args.get("embarked")

    data = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "fare": fare,
        "embarked": embarked
    }])

    prediction = model.predict(data)[0]

    return jsonify({"prediction": int(prediction)})