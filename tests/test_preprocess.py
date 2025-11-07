import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from house_prices.preprocess import load_preprocessors, transform_df

def test_transform_df():
    preprocessors = load_preprocessors(out_dir="models")
    data = pd.DataFrame({
    "OverallQual": [5, 7],
    "GrLivArea": [1500, 2000],
    "TotalBsmtSF": [800, 1000],
    "Neighborhood": ["CollgCr", "Veenker"],
    "HouseStyle": ["2Story", "1Story"]
})
    X_proc = transform_df(data, preprocessors)
    assert X_proc.shape[0] == 2
