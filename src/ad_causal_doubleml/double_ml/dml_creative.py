import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
 
from econml.dml import LinearDML

#from feature_engineering import build_feature_matrix, TARGET_CATEGORICAL_FEATURES
sklearn.set_config(transform_output="default")

from ad_causal_doubleml.feature_engineering.feature_engineering import build_feature_matrix, TARGET_CATEGORICAL_FEATURES
sklearn.set_config(transform_output="default")
# --------------------------------------------------------------------------
# 0. Config
# --------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5  # EconML's outer cross-fitting folds
 
OUTCOME_COL = "click"
TREATMENT_COL = "creative"
HIGH_CARD_COLS = TARGET_CATEGORICAL_FEATURES  # ["city", "domain", "slotid", "slotprice"]
 
# The iPinYou dataset is partitioned by advertiser at the file level, so
# FILE_PATH in feature_engineering.py should already point at the
# advertiser-1458 log file. This is just a sanity check that the resulting
# creative column has the cardinality expected.
EXPECTED_N_TREATMENT_LEVELS = 8

# --------------------------------------------------------------------------
# 1. Load features
# --------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    """
    Runs feature_engineering.py pipeline unmodified (it
    should already be pointed at the advertiser-1458 log file), and checks
    that `creative` resolves to EXPECTED_N_TREATMENT_LEVELS levels before
    returning, so a mismatch (e.g. wrong file, or a creative that appears
    in some other advertiser's traffic) fails loudly here rather than
    silently downstream.
    """
    print('Loading feature matrix.')
    df = build_feature_matrix() # uses the func from feature_engineering
    print('Finished loading feature matrix.')
    print(df.columns)
    for col in ["domain", "slotid"]:
        df[col] = df[col].astype(str)

    n_levels = df[TREATMENT_COL].nunique()
    if n_levels != EXPECTED_N_TREATMENT_LEVELS:
        print(
            f"WARNING: expected {EXPECTED_N_TREATMENT_LEVELS} creative "
            f"levels, found {n_levels}. Check that FILE_PATH in "
            f"feature_engineering.py points at the right advertiser's data."
        )
 
    return df
 
 # --------------------------------------------------------------------------
# 2. Nuisance-model pipelines: target encoding embedded in cross-fitting
# --------------------------------------------------------------------------

# replace domain_id with the average click rate for that domain 
# the target encoded columns are no longer features in the traditional
# sense but instead a small model output

# this model output will be different for every fold. A static df cant
# hold five different  context-dependent versions of itself at once

# Pipeline class from scikit-learn, which is used to chain multiplie
# data processing and modelling steps into a single object. 
# good to ensure data preprocessing and model training without
# data leakage

def make_nuisance_pipeline(
    final_estimator,
    high_card_idx: list[int],
    low_card_idx: list[int],
    inner_cv_folds: int = 5,
) -> Pipeline:
    """
    Builds one nuisance-model pipeline:
      1) TargetEncoder on the high-cardinality columns (own internal
         cross-fitting via `cv=inner_cv_folds`, target_type="auto" so it
         works for both the binary outcome (model_y) and the 8-class
         treatment (model_t) without any special-casing).
      2) Passthrough for the already-engineered low-cardinality columns.
      3) The supplied final_estimator (a classifier, since both Y and T
         are discrete here).
 
    IMPORTANT: EconML internally concatenates X and W into a plain numpy
    array before calling model_y.fit(...) / model_t.fit(...), so column
    selection inside the ColumnTransformer must use integer positions,
    not names. high_card_idx / low_card_idx give those positions and must
    match the column order you build W in (see build_design_matrix below).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "high_card_target_encoding",
                TargetEncoder(
                    target_type="auto",
                    cv=inner_cv_folds,
                    random_state=RANDOM_STATE,
                ),
                high_card_idx,
            ),
            ("low_card_passthrough", "passthrough", low_card_idx),
        ]
    )
    # A fresh Pipeline instance must be created for model_y and again for
    # model_t (do not reuse the same object) -- EconML clones whatever you
    # pass, but two independent instances keeps intent unambiguous and
    # avoids any accidental shared fitted state during debugging.
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", final_estimator),
    ])
 
 
def build_design_matrix(df: pd.DataFrame, low_card_cols: list[str]):
    """
    Fixes the column order used for W (and therefore the integer indices
    the nuisance pipelines rely on): high-cardinality columns first, then
    the already-engineered low-cardinality block.
    """
    print('Starting build_design_matrix.')
    ordered_cols = HIGH_CARD_COLS + low_card_cols
    W = df[ordered_cols]
    high_card_idx = list(range(len(HIGH_CARD_COLS)))
    low_card_idx = list(range(len(HIGH_CARD_COLS), len(ordered_cols)))
    print('Finished build_design_matrix.')
    return W, high_card_idx, low_card_idx

# --------------------------------------------------------------------------
# 3. Fit DML
# --------------------------------------------------------------------------
def fit_dml(df: pd.DataFrame, low_card_cols: list[str]) -> LinearDML:
    print('Starting fit_dml')
    print('Calling build_design_matrix')
    W, high_card_idx, low_card_idx = build_design_matrix(df, low_card_cols)
 
    model_y_pipeline = make_nuisance_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                       max_iter=30,
                                       verbose=1),
        high_card_idx,
        low_card_idx,
    )
    model_t_pipeline = make_nuisance_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                       max_iter=30,
                                       verbose=1),
        high_card_idx,
        low_card_idx,
    )
 
    Y = df[OUTCOME_COL].to_numpy()
    T = df[TREATMENT_COL].to_numpy()
    categories = sorted(pd.unique(T).tolist())
 
    # LinearDML gives an average treatment effect per creative category
    # (vs. a baseline category) with a linear final stage 

    # During LinearDML.fit(), EconML performs cross-fitting. For each
    # training fold it clones model_y_pipeline and model_t_pipeline and
    # calls their fit() methods. That Pipeline.fit() automatically fits
    # TargetEncoder on the training portion of the fold only. Transforms
    # the high-cardinality columns using those learned target encodings.
    # Passes through the remaining engineered features unchanged. 
    # Fits the HistGradientBoostingClassifier on the transformed
    # feature matrix.
    
    est = LinearDML(
        model_y=model_y_pipeline,
        model_t=model_t_pipeline,
        discrete_treatment=True,   # T is the 8-level creative variable
        discrete_outcome=True,     # Y (click) is binary -> model_y is a classifier
        categories=categories,
        cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
 
    # X=None: all covariates enter as controls (W) only, giving one ATE per
    # treatment category rather than a CATE surface. See note above if you
    # want heterogeneity.
    est.fit(Y, T, X=None, W=W)
    return est
 

  
# --------------------------------------------------------------------------
# 4. Report results
# --------------------------------------------------------------------------
def report(est: LinearDML) -> None:
    print("Per-treatment-level average effect on P(click), vs. baseline creative:")
    print(est.summary())
 
 
if __name__ == "__main__":
    df = load_features()
 
    # ASSUMPTION: adjust this to the actual list of already-engineered
    # low-cardinality column names your feature_engineering.py produces
    # (one-hot dummies, hour_sin/hour_cos, usertag_* MLB columns, etc.)
    engineered_low_card_cols = [
        c for c in df.columns
        if c not in HIGH_CARD_COLS + [OUTCOME_COL, TREATMENT_COL]
    ]
 
    est = fit_dml(df, engineered_low_card_cols)
    report(est)