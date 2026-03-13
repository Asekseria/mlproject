from flask import Flask, request, jsonify
import pandas as pd
import joblib
from src.pipeline.predict_pipeline import PredictPipeline

# load pipeline
predictor = PredictPipeline()

app = Flask(__name__)

@app.route("/")
def home():
    return "Titanic Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_df = pd.DataFrame([data])

    prediction = predictor.predict(input_df)[0]

    return jsonify({
        "prediction": int(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True,port=5000)