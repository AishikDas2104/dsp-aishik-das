import pandas as pd
import mlflow
import mlflow.sklearn
from house_prices.train import build_model

if __name__ == "__main__":
    data = pd.read_csv("data/train.csv")

    with mlflow.start_run():
        metrics = build_model(data)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_metric("rmsle", metrics["rmsle"])

        print(f"Model training completed. RMSLE: {metrics['rmsle']}")