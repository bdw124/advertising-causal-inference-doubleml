"""
feature_engineering.py

Feature engineering pipeline for the iPinYou dataset.
Prepares raw log data into a model-ready DataFrame ahead of
double machine learning.

Note: target encoding for high-cardinality categorical features
(city, domain, slotid, slotprice, creative) is deliberately NOT
performed here. It should be done alongside the DoubleML estimation
(e.g. within cross-fitting folds) to avoid leaking target information
into the covariates.

"""

import numpy as np
import pandas as pd

from ad_causal_doubleml.config.paths import DATA_DIR

from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

# Return transformed outputs as DataFrames rather than numpy arrays
set_config(transform_output="pandas")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

FILE_PATH = DATA_DIR / "train.log.txt"

DTYPES = {
    "click": "int8",
    "weekday": "int8",
    "hour": "int8",
    "logtype": "int8",
    "region": "int16",
    "city": "int16",
    "adexchange": "category",
    "slotwidth": "int16",
    "slotheight": "int16",
    "slotvisibility": "category",
    "slotformat": "int16",
    "slotprice": "int16",
    "bidprice": "int16",
    "payprice": "int16",
    "advertiser": "category",
    # High-cardinality categorical features
    "bidid": "string",
    "timestamp": "string",
    "ipinyouid": "string",
    "useragent": "category",
    "IP": "string",
    "domain": "string",
    "url": "string",
    "urlid": "string",
    "slotid": "string",
    "creative": "string",
    "keypage": "string",
    "usertag": "string",
}

COLUMNS_TO_DROP = ["bidid", "logtype", "ipinyouid", "IP", "url", "urlid", "payprice"]

VARIABLES_ONE_HOT = [
    "adexchange", "useragent", "weekday", "region", "slotwidth",
    "slotheight", "slotvisibility", "slotformat", "bidprice",
    "advertiser", "keypage",
]

MULTI_LABEL_COLUMN = "usertag"

# Left un-encoded on purpose - to be target-encoded during DoubleML
# cross-fitting, not here.
TARGET_CATEGORICAL_FEATURES = ["city", "domain", "slotid", "slotprice", "creative"]


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #

def load_data(file_path: str = FILE_PATH, dtypes: dict = DTYPES) -> pd.DataFrame:
    """Load the raw iPinYou log file with explicit dtypes to keep memory down."""
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=dtypes,
        na_values=["null"],
    )
    return df


def drop_unused_columns(df: pd.DataFrame, columns: list = COLUMNS_TO_DROP) -> pd.DataFrame:
    """Drop identifier / redundant columns not used in feature engineering."""
    return df.drop(columns=columns)


def add_cyclical_hour_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode 'hour' (0-23) as sine/cosine pairs to preserve cyclical structure."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df = df.drop(columns=["hour"])
    return df


def one_hot_encode(df: pd.DataFrame, columns: list = VARIABLES_ONE_HOT) -> pd.DataFrame:
    """One-hot encode low-cardinality categorical features and append to df."""
    sample_df_one_hot = df[columns]

    encoder = OneHotEncoder(sparse_output=False)
    encoded_array = encoder.fit_transform(sample_df_one_hot)

    encoded_cols = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(sample_df_one_hot.columns),
        index=sample_df_one_hot.index,
    )

    df_out = pd.concat([df.drop(columns=columns), encoded_cols], axis=1)
    return df_out


def multi_label_binarize_usertag(df: pd.DataFrame, column: str = MULTI_LABEL_COLUMN) -> pd.DataFrame:
    """Turn the comma-separated 'usertag' column into one binary column per tag."""
    df = df.copy()
    df["usertag_list"] = df[column].fillna("").str.split(",")

    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(df["usertag_list"])

    tag_df = pd.DataFrame(
        tag_matrix,
        columns=[f"usertag_{t}" for t in mlb.classes_],
        index=df.index,
    )

    df_out = pd.concat(
        [df.drop(columns=[column, "usertag_list"]), tag_df],
        axis=1,
    )
    return df_out


def build_feature_matrix(file_path: str = FILE_PATH) -> pd.DataFrame:
    """
    Run the feature engineering pipeline end-to-end.

    Returns a DataFrame with cyclical hour features, one-hot encoded
    low-cardinality categoricals, and multi-label binarized user tags.
    The columns in TARGET_CATEGORICAL_FEATURES are left untouched -
    encode these downstream, inside your DoubleML cross-fitting loop.
    """
    df = load_data(file_path)
    df = drop_unused_columns(df)
    df = add_cyclical_hour_features(df)
    df = one_hot_encode(df)
    df = multi_label_binarize_usertag(df)
    return df


if __name__ == "__main__":
    df = build_feature_matrix()
    print(df.shape)
    print(df.head())