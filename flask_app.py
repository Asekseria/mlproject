from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

predictor = PredictPipeline()
app = Flask(__name__)

# Simple HTML template for the browser
HTML_FORM = """
<!DOCTYPE html>
<html>
<head><title>Titanic Predictor</title></head>
<body>
    <h2>Titanic Survival Prediction</h2>
    <form action="/predict" method="POST">
        <input type="number" name="pclass" placeholder="Pclass (1, 2, 3)" required><br><br>
        <input type="text" name="sex" placeholder="Sex (male/female)" required><br><br>
        <input type="number" name="age" placeholder="Age" required><br><br>
        <input type="number" name="fare" step="any" placeholder="Fare" required><br><br>
        <input type="text" name="embarked" placeholder="Embarked (S, C, Q)" required><br><br>
        <button type="submit">Predict Survival</button>
    </form>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_FORM)

# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "GET":
#         return render_template_string(HTML_FORM)

#     # Step 1: Get data (works for BOTH curl JSON and Browser Form)
#     if request.is_json:
#         data = request.get_json()
#     else:
#         # This picks up data from the HTML form
#         data = request.form.to_dict()
#         # Convert numeric strings to actual numbers for the ML model
#         data['pclass'] = int(data['pclass'])
#         data['age'] = float(data['age'])
#         data['fare'] = float(data['fare'])

#     # Step 2: Create DataFrame
#     input_df = pd.DataFrame([data])

#     # Step 3: Predict
#     prediction = predictor.predict(input_df)

#     # Return JSON if it's a curl request, otherwise return text for the browser
#     if request.is_json:
#         return jsonify({"prediction": int(prediction)})
#     else:
#         result = "Survived" if prediction == 1 else "Did Not Survive"
#         return f"<h1>Result: {result}</h1><br><a href='/'>Go Back</a>"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template_string(HTML_FORM)

    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            data['pclass'] = int(data['pclass'])
            data['age'] = float(data['age'])
            data['fare'] = float(data['fare'])

        input_df = pd.DataFrame([data])
        prediction = predictor.predict(input_df)

        if request.is_json:
            return jsonify({"prediction": int(prediction)})
        else:
            result = "Survived" if prediction == 1 else "Did Not Survive"
            return f"<h1>Result: {result}</h1><br><a href='/'>Go Back</a>"

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0", port=8080) #local host and local ip
