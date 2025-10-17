import sys
import os
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from house_prices.preprocess import load_preprocessors, transform_df

expected_path = os.path.join("notebooks", "models", "processed_train_df.parquet")
expected = pd.read_parquet(expected_path)

df = pd.read_csv(os.path.join("data", "train.csv"))

preprocessors = load_preprocessors(out_dir=os.path.join("notebooks", "models"))
cols = preprocessors["continuous"] + preprocessors["categorical"]

X = df[cols].copy()
actual = transform_df(X.loc[expected.index], preprocessors)

if "SalePrice" in expected.columns:
    actual = actual.reset_index(drop=True)
    expected = expected.reset_index(drop=True)

pd.testing.assert_frame_equal(
    actual, 
    expected.drop(columns=["SalePrice"]) if "SalePrice" in expected.columns else expected
)

print("Processed dataframe matches expected. Test passed.")
