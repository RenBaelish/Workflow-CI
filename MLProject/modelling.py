import os
import joblib
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


os.environ.pop(
    "MLFLOW_TRACKING_URI",
    None
)


mlflow.set_experiment(
    "Wine Quality CI Training"
)


df = pd.read_csv(
    "winequality-red-preprocessing.csv"
)


X = df.drop(
    columns=[
        "quality"
    ]
)

y = df[
    "quality"
]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    mlflow.log_param(
        "model",
        "RandomForestRegressor"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "random_state",
        42
    )


    mlflow.log_metric(
        "rmse",
        rmse
    )

    mlflow.log_metric(
        "mae",
        mae
    )

    mlflow.log_metric(
        "r2_score",
        r2
    )


    model_path = "model.pkl"

    joblib.dump(
        model,
        model_path
    )


    mlflow.log_artifact(
        model_path
    )


    print(
        "Model training completed."
    )

    print(
        f"RMSE: {rmse}"
    )

    print(
        f"MAE: {mae}"
    )

    print(
        f"R2 Score: {r2}"
    )
