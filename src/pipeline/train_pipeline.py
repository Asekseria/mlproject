import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.components.data_ingestion import load_data
from src.components.data_transformation import get_preprocessor

def train_model():

    X_train, X_test, y_train, y_test = load_data()

    preprocessor = get_preprocessor()

    model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "src/models/model.pkl")

    print("Model trained and saved")

if __name__ == "__main__":
    train_model()