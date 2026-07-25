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


# MLflow menggunakan path lokal relatif
mlflow.set_tracking_uri(
    "file:./mlruns"
)

mlflow.set_experiment(
    "Wine Quality CI Training"
)


# Membaca dataset
df = pd.read_csv(
    "winequality-red-preprocessing.csv"
)


# Memisahkan fitur dan target
X = df.drop(
    columns=["quality"]
)

y = df["quality"]


# Membagi dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


with mlflow.start_run():

    # Membuat model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )


    # Training
    model.fit(
        X_train,
        y_train
    )


    # Prediksi
    predictions = model.predict(
        X_test
    )


    # Evaluasi
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


    # Logging parameter
    mlflow.log_param(
        "model",
        "RandomForestRegressor"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )


    # Logging metrics
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


    # Simpan model ke file lokal
    model_path = "model.pkl"

    joblib.dump(
        model,
        model_path
    )


    # Upload model sebagai artifact MLflow
    mlflow.log_artifact(
        model_path
    )


    print("Model training completed.")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)
