import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def load_data():

    titanic = fetch_openml(name="titanic", version=1, as_frame=True)
    df = titanic.frame

    df = df[["survived", "pclass", "sex", "age", "fare", "embarked"]]

    X = df.drop("survived", axis=1)
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test