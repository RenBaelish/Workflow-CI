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
    "Bike Sharing Prediction"
)


mlflow.autolog()


df = pd.read_csv(
    "day_preprocessing.csv"
)


X = df.drop(
    columns=[
        "cnt"
    ]
)

y = df[
    "cnt"
]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


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


joblib.dump(
    model,
    "model.pkl"
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
