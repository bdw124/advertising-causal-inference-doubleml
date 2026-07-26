import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import cross_val_predict
 
from econml.dml import LinearDML

from datetime import datetime

start_time = datetime.now()

print(f"Script started: {start_time:%Y-%m-%d %H:%M:%S}")

# setting options to be able to see the entire table output 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)          # Automatically detect terminal width
pd.set_option('display.max_colwidth', None)   # Don't truncate long cell contents

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



def build_model_y_estimator() -> HistGradientBoostingClassifier:
    """
    Outcome (click) classifier. class_weight="balanced" addresses the
    extreme rarity of click (0.08%)
    """
    return HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_iter=60
    )

def build_model_t_estimator() -> HistGradientBoostingClassifier:
    """
    Treatment (creative) classifier. No class weight as creative
    does not suffer same sparsity issue.
    """
    return HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_iter=60,
    )


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
    #for col in ["domain"]:
    #    df[col] = df[col].astype(str)

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

def compute_overlap_trim_mask(df: pd.DataFrame, low_card_cols: list[str], threshold: float = 0.1):
    """
    Fits model_t via honest out-of-fold cross-validation across the WHOLE
    dataset (same fold discipline EconML uses internally -- a row's
    propensity is never predicted by a model that was trained on that
    row), then returns a boolean mask for rows with
    P(observed creative | X) >= threshold (Crump et al. 2009's standard
    rule of thumb), plus a before/after diagnostic table printed to screen.

    NOTE: this refits model_t via cross-validation independently of the
    real DML fit -- it's necessarily a second, separate cost on top of
    what EconML does internally, since trimming has to happen BEFORE the
    real fit can run on the trimmed sample.
    """
    W, high_card_idx, low_card_idx = build_design_matrix(df, low_card_cols)
    T = df[TREATMENT_COL].to_numpy()

    pipeline = make_nuisance_pipeline(build_model_t_estimator(), high_card_idx, low_card_idx)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("Fitting model_t out-of-fold across the full dataset for trimming (this refits model_t; expect similar runtime to one nuisance-model pass)...")
    proba = cross_val_predict(pipeline, W, T, cv=cv, method="predict_proba")


    # Overlap requires EVERY treatment category to have a real chance for
    # each row's covariates -- not just that the observed category was
    # plausible. Use the minimum probability across all 8 categories, per
    # row, as the overlap statistic. (Previously this used P(observed
    # creative|X) instead, which flags the wrong thing -- a HIGH value
    # there means the row's covariates near-uniquely determine the
    # creative it got, which is the deterministic case, not the safe one.)
    min_propensity = proba.min(axis=1)

    n_categories = len(np.unique(T))
    ceiling = 1 / n_categories
    print(f"(Theoretical ceiling for {n_categories} categories: 1/{n_categories} = {ceiling:.4f} -- "
        f"the minimum can never exceed this even under perfectly even assignment, "
        f"so pick threshold relative to that, not a flat number.)")
    print(pd.Series(min_propensity).describe())

    mask = min_propensity >= threshold

    # --- before/after diagnostics, per creative ---
    diag = pd.DataFrame({TREATMENT_COL: T, OUTCOME_COL: df[OUTCOME_COL].to_numpy(), "kept": mask})
    before = diag.groupby(TREATMENT_COL).agg(n_rows_before=(OUTCOME_COL, "size"), n_clicks_before=(OUTCOME_COL, "sum"))
    after = diag[diag["kept"]].groupby(TREATMENT_COL).agg(n_rows_after=(OUTCOME_COL, "size"), n_clicks_after=(OUTCOME_COL, "sum"))
    summary = before.join(after).fillna(0)
    summary["rows_trimmed"] = summary["n_rows_before"] - summary["n_rows_after"]
    summary["clicks_trimmed"] = summary["n_clicks_before"] - summary["n_clicks_after"]
    summary["pct_rows_trimmed"] = (summary["rows_trimmed"] / summary["n_rows_before"] * 100).round(2)

    print(f"\n=== Overlap trim (threshold={threshold}) ===")
    print(summary.to_string()) # as an attempt to view the entire table
    summary.to_csv("overlap_summary_table.csv")
    print(f"\nTotal rows: {len(df):,} -> {mask.sum():,} ({mask.mean():.1%} kept)")
    print(f"Total clicks: {diag[OUTCOME_COL].sum():,} -> {diag.loc[mask, OUTCOME_COL].sum():,}")

    return mask

# --------------------------------------------------------------------------
# 3. Fit DML
# --------------------------------------------------------------------------
def fit_dml(df: pd.DataFrame, low_card_cols: list[str]) -> LinearDML:
    print('Starting fit_dml')
    print('Calling build_design_matrix')
    W, high_card_idx, low_card_idx = build_design_matrix(df, low_card_cols)
 
    model_y_pipeline = make_nuisance_pipeline(
        build_model_y_estimator(), high_card_idx,  low_card_idx,
    )

    model_t_pipeline = make_nuisance_pipeline(
        build_model_t_estimator(), high_card_idx, low_card_idx,
    )

 
    Y = df[OUTCOME_COL].to_numpy()
    T = df[TREATMENT_COL].to_numpy()
    categories = ["48f2e9ba15708c0146bda5e1dd653caa"] + [
    c for c in sorted(pd.unique(T).tolist()) if c != "48f2e9ba15708c0146bda5e1dd653caa"
    ]
    # sorting alphabetically resulted in the group with the lowest clicks being the comparison 
    #categories = sorted(pd.unique(T).tolist())
 
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
    print(f"df shape: {df.shape}")
    # Testing
    #df = df.sample(frac=0.1)
    #print(f"df shape: {df.shape}")
    engineered_low_card_cols = [
        c for c in df.columns
        if c not in HIGH_CARD_COLS + [OUTCOME_COL, TREATMENT_COL]
    ]

    mask = compute_overlap_trim_mask(df, engineered_low_card_cols, threshold=0.001)
    df = df.loc[mask].reset_index(drop=True)  # reset_index matters -- same reason build_design_matrix needs clean positional indices

    est = fit_dml(df, engineered_low_card_cols)
    report(est)

end_time = datetime.now()
elapsed = end_time - start_time

print("\n" + "=" * 80)
print(f"Script finished: {end_time:%Y-%m-%d %H:%M:%S}")
print(f"Total runtime:   {elapsed}")
print("=" * 80)