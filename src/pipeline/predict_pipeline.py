import joblib
import pandas as pd

class PredictPipeline:

    def __init__(self):

        self.model = joblib.load("src/models/model.pkl")

    def predict(self, data):

        df = pd.DataFrame([data])

        prediction = self.model.predict(df)

        return int(prediction[0])