import pandas as pd
import numpy as np
import joblib
import os
from house_prices.preprocess import load_preprocessors, transform_df


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    model = joblib.load(os.path.join("models", "model.joblib"))
    preprocessors = load_preprocessors(out_dir="models")

    if "SalePrice" in input_data.columns:
        input_features = input_data[
            preprocessors["continuous"] +
            preprocessors["categorical"]
        ].copy()
    else:
        input_features = input_data[
            preprocessors["continuous"] +
            preprocessors["categorical"]
        ].copy()

    X_proc = transform_df(input_features, preprocessors)
    y_log_pred = model.predict(X_proc.values)
    y_pred = np.expm1(y_log_pred)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    return y_pred
