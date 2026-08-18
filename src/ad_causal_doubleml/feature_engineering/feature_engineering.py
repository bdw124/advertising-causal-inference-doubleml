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

import pandas as pd

import string # mapping creative ids

from sklearn.preprocessing import SplineTransformer # hour transformation

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
# dropping slot width and slot height as there is no variation with respect to creative

COLUMNS_TO_DROP = ["bidid", "logtype", "ipinyouid", "IP", "url", "urlid", "payprice","timestamp","slotheight","slotwidth","slotid","domain"]

# advertiser not included here as constant when only looking at 1458
VARIABLES_ONE_HOT = [
    "adexchange", "useragent", "weekday", "region", "slotvisibility", "bidprice","keypage","slotformat"
]

MULTI_LABEL_COLUMN = "usertag"

# Left un-encoded on purpose - to be target-encoded during DoubleML
# cross-fitting, not here.
TARGET_CATEGORICAL_FEATURES = ["city", "slotprice"] # no creative here as D = creative


CREATIVES_TO_DROP = ["fb5afa9dba1274beaf3dad86baf97e89", "832b91d59d0cb5731431653204a76c0e","a499988a822facd86dd0e8e4ffef8532"]



# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #

def load_data(file_path: str = FILE_PATH, dtypes: dict = DTYPES) -> pd.DataFrame:
    """Load the raw iPinYou log file with explicit dtypes to keep memory down."""
    print("Reading in csv...")
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=dtypes,
        na_values=["null"],
    )
    print("Finished loading df.")
    return df

def map_creatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    The creative ids are long hashed variables which are hard to interpret. 
    This function changes them to A, B, C,...
    """

    creative_ids = sorted(df["creative"].astype(str).unique())
    labels = list(string.ascii_uppercase[:len(creative_ids)])
    creative_mapping = pd.DataFrame({
        "creative_original": creative_ids,
        "creative_short": labels
    })
    creative_map = dict(zip(
        creative_mapping["creative_original"],
        creative_mapping["creative_short"]
    ))
    print(creative_map)
    df["creative_label"] = df["creative"].astype(str).map(creative_map)
    df["creative"] = df["creative_label"]
    df.drop(columns=["creative_label"], inplace=True)
    return df


def drop_unused_columns(df: pd.DataFrame, columns: list = COLUMNS_TO_DROP) -> pd.DataFrame:
    """Drop identifier / redundant columns not used in feature engineering."""
    return df.drop(columns=columns)

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    print("Start drop_duplicates function...")

    # identify the duplicate rows
    df_duplicated = df[df["bidid"].duplicated(keep=False)]

    print(f"df_duplicated before drop: {df_duplicated.shape}")
    # dropping rows that are identical 
    df_duplicated = df_duplicated.drop_duplicates()
    print(f"df_duplicated after drop: {df_duplicated.shape}")

    # dropping rows where usertag is null and we have non-null usertag in other row
    df_usertag_null = df_duplicated[df_duplicated["usertag"].isna()]
    df_usertag_not_null = df_duplicated[df_duplicated["usertag"].notna()]
    # idenitfy rows from df_usertag_null that are in df_usertag_not_null
    df_usertag_null_in_not_null = df_usertag_null[df_usertag_null['bidid'].isin(df_usertag_not_null['bidid'])]

    result = df_duplicated.merge(df_usertag_null_in_not_null.drop_duplicates(), how='left', indicator=True)
    df_duplicated = result[result['_merge'] == 'left_only'].drop(columns='_merge')



    """Retain only latest timestamp"""
    # Make a copy
    df_clean = df_duplicated.copy()

    # Convert timestamp to datetime
    df_clean["timestamp_dt"] = pd.to_datetime(
        df_clean["timestamp"].astype(str),
        format="%Y%m%d%H%M%S%f"
    )

    # Columns that must be identical for two rows to be considered duplicates
    compare_cols = [
        col for col in df_clean.columns
        if col not in ["timestamp", "ipinyouid", "timestamp_dt"]
    ]

    # Sort by timestamp so the newest row comes first
    df_clean = df_clean.sort_values("timestamp_dt", ascending=False)

    # Remove duplicates where every column except timestamp and iPinYouid
    # is identical. The newest timestamp is retained.
    df_clean = df_clean.drop_duplicates(
        subset=compare_cols,
        keep="first"
    )

    # Remove helper column
    df_clean = df_clean.drop(columns="timestamp_dt")

    # Reset index
    df_clean = df_clean.reset_index(drop=True)

    df_duplicated = df_clean 

    """Combine usertags"""
    random_rows = df_duplicated.groupby("bidid").sample(n=1, random_state=42)

    # Combine all usertags for each bidid
    combined_tags = (
        df_duplicated.groupby("bidid")["usertag"]
        .apply(lambda x: ",".join(sorted(set(
            tag
            for value in x.dropna()
            for tag in value.split(",")
        ))))
    )

    # Replace the usertag in the randomly selected rows
    random_rows["usertag"] = random_rows["bidid"].map(combined_tags)

    # Result
    df_combined = random_rows.reset_index(drop=True)

    """Remove duplicates from df and then append the cleaned duplicates"""
    # remove all original rows whose bidid has been tidied 
    df_clean = df[~df['bidid'].isin(df_combined['bidid'])].copy()

    # append the tidied duplicates back in 
    df_clean = pd.concat(
        [df_clean, df_combined],
        ignore_index=True
    )
    print(f"Before cleaning the df has shape {df.shape}")
    print(f"After cleaning the df has shape {df_clean.shape}")

    return df_clean


def drop_creatives_with_no_variation_in_slot_visibility(df: pd.DataFrame, rows: list = CREATIVES_TO_DROP) -> pd.DataFrame:
    """"""
    return df[~df["creative"].isin(rows)]


def add_cyclical_hour_features(df: pd.DataFrame) -> pd.DataFrame:
    """"""
    df = df.copy()
    spline = SplineTransformer(
    degree=3,
    n_knots=8,
    extrapolation='periodic', # hours are cyclical 
    include_bias=False 
    )
    spline_features = spline.fit_transform(df["hour"].to_numpy().reshape(-1,1))
    spline_columns = [f"hour_spline_{i}"
                      for i in range(spline_features.shape[1])
                      ]
    df[spline_columns] = spline_features

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
    df = map_creatives(df)
    df = drop_duplicates(df)
    df = drop_unused_columns(df)
    # commented out whilst testing if trimming does this appropriately
    df = drop_creatives_with_no_variation_in_slot_visibility(df)
    df = add_cyclical_hour_features(df)
    df = one_hot_encode(df)
    df = multi_label_binarize_usertag(df)

    return df


if __name__ == "__main__":
    df = build_feature_matrix()

    print('Feature engineering script finished.')
