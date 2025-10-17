from typing import Dict
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

from house_prices.preprocess import (
    split_data,
    fit_preprocessors,
    transform_df,
    save_preprocessors,
)


def compute_rmsle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    precision: int = 4
) -> float:
    rmsle = float(np.sqrt(mean_squared_log_error(y_true, y_pred)))
    return round(rmsle, precision)


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    continuous = ["GrLivArea", "TotalBsmtSF"]
    categorical = ["Neighborhood", "HouseStyle"]

    X_train, X_test, y_train, y_test = split_data(
        data[continuous + categorical + ["SalePrice"]],
        "SalePrice",
    )

    preprocessors = fit_preprocessors(
        X_train,
        categorical=categorical,
        continuous=continuous,
    )
    X_train_proc = transform_df(X_train, preprocessors)

    os.makedirs("models", exist_ok=True)
    processed_df = pd.concat(
        [X_train_proc.reset_index(drop=True),
         y_train.reset_index(drop=True)],
        axis=1,
    )
    processed_df.to_parquet(
        os.path.join("models", "processed_train_df.parquet"),
        index=False,
    )

    y_train_log = np.log1p(y_train.values)
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_proc.values, y_train_log)

    joblib.dump(model, os.path.join("models", "model.joblib"))
    save_preprocessors(preprocessors, out_dir="models")

    X_test_proc = transform_df(X_test, preprocessors)
    y_pred_log = model.predict(X_test_proc.values)
    y_pred = np.expm1(y_pred_log)
    rmsle = compute_rmsle(y_true=y_test.values, y_pred=y_pred)

    return {"rmsle": rmsle}
