import pytest
from flask_app import app


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


# 1. Health check (basic API working)
def test_home(client):
    response = client.get("/")
    assert response.status_code == 200


# 2. Prediction endpoint works
def test_prediction(client):
    sample_input = {
        "pclass": 3,
        "sex": "male",
        "age": 22,
        "fare": 7.25,
        "embarked" : 'S'
    }

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data


# 3. Prediction value sanity (0 or 1)
def test_prediction_range(client):
    sample_input = {
        "pclass": 1,
        "sex": "female",
        "age": 30,
        "fare": 100,
        "embarked" : 'Q'
    }

    response = client.post("/predict", json=sample_input)
    data = response.get_json()

    assert data["prediction"] in [0, 1]


def test_invalid_input(client):
    bad_input = {
        "pclass": "wrong",
        "age": "unknown"
    }

    response = client.post("/predict", json=bad_input)

    assert response.status_code == 400
