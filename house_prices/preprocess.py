from typing import Any, Dict, List, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os


def split_data(
    df: pd.DataFrame,
    target: str = "SalePrice",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def fit_preprocessors(
    X_train: pd.DataFrame,
    categorical: List[str],
    continuous: List[str]
) -> Dict[str, Any]:
    encoder = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore"
    )
    scaler = StandardScaler()
    encoder.fit(X_train[categorical])
    scaler.fit(X_train[continuous])
    return {
        "encoder": encoder,
        "scaler": scaler,
        "categorical": categorical,
        "continuous": continuous,
    }


def transform_df(
    X: pd.DataFrame,
    preprocessors: Dict[str, Any]
) -> pd.DataFrame:
    enc = preprocessors["encoder"]
    sc = preprocessors["scaler"]
    cat_cols = preprocessors["categorical"]
    cont_cols = preprocessors["continuous"]

    cont_arr = sc.transform(X[cont_cols])
    cont_df = pd.DataFrame(
        cont_arr, columns=cont_cols, index=X.index
    )

    enc_arr = enc.transform(X[cat_cols])
    enc_cols = list(enc.get_feature_names_out(cat_cols))
    enc_df = pd.DataFrame(
        enc_arr, columns=enc_cols, index=X.index
    )

    processed = pd.concat([cont_df, enc_df], axis=1)
    return processed


def save_preprocessors(
    preprocessors: Dict[str, Any],
    out_dir: str = "models"
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(
        preprocessors["encoder"],
        os.path.join(out_dir, "encoder.joblib"),
    )
    joblib.dump(
        preprocessors["scaler"],
        os.path.join(out_dir, "scaler.joblib"),
    )
    meta = {
        "categorical": preprocessors["categorical"],
        "continuous": preprocessors["continuous"],
    }
    joblib.dump(
        meta, os.path.join(out_dir, "preprocessor_meta.joblib")
    )


def load_preprocessors(out_dir: str = "models") -> Dict[str, Any]:
    encoder = joblib.load(os.path.join(out_dir, "encoder.joblib"))
    scaler = joblib.load(os.path.join(out_dir, "scaler.joblib"))
    meta = joblib.load(
        os.path.join(out_dir, "preprocessor_meta.joblib")
    )
    return {
        "encoder": encoder,
        "scaler": scaler,
        "categorical": meta["categorical"],
        "continuous": meta["continuous"],
    }
