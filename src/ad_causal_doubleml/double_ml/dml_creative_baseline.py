import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import cross_val_predict
# ClassifierMixin tells sklearn that the estimator is a classifier
# clone creates a new estimator with same params but no learned data
from sklearn.base import BaseEstimator, ClassifierMixin, clone
 
from econml.dml import LinearDML

from datetime import datetime

start_time = datetime.now()

print(f"Script started: {start_time:%Y-%m-%d %H:%M:%S}")

# setting options to be able to see the entire table output 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)          # Automatically detect terminal width
pd.set_option('display.max_colwidth', None)   # Don't truncate long cell contents
pd.set_option('display.float_format', lambda x: f'{x:.8f}')

from ad_causal_doubleml.feature_engineering.feature_engineering import build_feature_matrix, TARGET_CATEGORICAL_FEATURES
sklearn.set_config(transform_output="default")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
RANDOM_STATE = 42
print(f"Random State: {RANDOM_STATE}")
N_SPLITS = 5  # EconML's outer cross-fitting folds
 
OUTCOME_COL = "click"
TREATMENT_COL = "creative"
HIGH_CARD_COLS = TARGET_CATEGORICAL_FEATURES  # decided and import from feature_engineering
 
# The iPinYou dataset is partitioned by advertiser at the file level, so
# FILE_PATH in feature_engineering.py should already point at the
# advertiser-1458 log file. This is just a sanity check that the resulting
# creative column has the cardinality expected.
EXPECTED_N_TREATMENT_LEVELS = 8 - 3 # I have removed 3 creatives

N_UNDERSAMPLE_REPEATS = 10
print(f"Number of Undersample Repeats: {N_UNDERSAMPLE_REPEATS}")

# configs for Logistic Regression 
LR_MAX_ITER = 5000
LR_C_GRID = np.logspace(-4, 4, 20)
TUNED_C = {"click": None, "creative": None} # later populated by tune_logistic_C()

# --------------------------------------------------------------------------
# Models 
# --------------------------------------------------------------------------

def build_model_y_estimator() -> LogisticRegression:
    """
    Outcome (click) classifier.
    
    Used in fit_dml() -> model_y_pipeline.
    Model estimates P(click | covariates) combined with undersampling
    during EconML cross-fitting. 
    """
    C = TUNED_C["click"] if TUNED_C["click"] is not None else 1.0
    return LogisticRegression(
        penalty="l2", C=C, solver="lbfgs",
        max_iter=LR_MAX_ITER, random_state=RANDOM_STATE,
    )

def build_model_t_estimator() -> LogisticRegression:
    """
    Treatment (creative) classifier. 

    Used in fit_dml() -> model_t_pipeline. 
    Model estimates P(creative | covariates) during EconML cross
    fitting and in compute_overlap_trim_mask() for the overlap
    diagnostics. No class weighting.
    """
    C = TUNED_C["creative"] if TUNED_C["creative"] is not None else 1.0
    return LogisticRegression(
        penalty="l2", C=C, solver="lbfgs",
        max_iter=LR_MAX_ITER, random_state=RANDOM_STATE,
    )


# --------------------------------------------------------------------------
# Hyperparameter Tuning
# --------------------------------------------------------------------------

def tune_logistic_C(W: pd.DataFrame, y: np.ndarray, 
                    high_card_idx: list[int], 
                    low_card_idx: list[int], 
                    label: str, 
                    multi_class: bool) -> float:
    """
    Tunes C via GridSearchCV over the full pipeline on the same Kfolds 
    used for DML cross-fitting and the HGB baseline - identical folds/
    scoring making it a fair comparison 
    """
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    base_lr = LogisticRegression(
        penalty='l2', solver='lbfgs', 
        max_iter=LR_MAX_ITER, random_state=RANDOM_STATE,
    )
    pipeline = make_nuisance_pipeline(base_lr, high_card_idx,
                                      low_card_idx)
    grid = GridSearchCV(
        pipeline, param_grid={"model__C": LR_C_GRID}, cv=cv,
        scoring="roc_auc_ovr" if multi_class else "roc_auc", n_jobs=-1,
    )
    grid.fit(W, y)
    best_C = grid.best_params_["model__C"]
    TUNED_C[label] = best_C
    print(f"[{label}] Tuned C = {best_C:.6g} "
          f"({N_SPLITS}-fold CV, scoring={'AUC-OVR' if multi_class else 'AUC'})")
    return best_C

# --------------------------------------------------------------------------
# Load features
# --------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    """
    Runs feature_engineering.py pipeline.
    """

    print('Loading feature matrix.')
    df = build_feature_matrix() # uses the func from feature_engineering
    print('Finished loading feature matrix.')

    # If include domain need to specify col type to prevent error
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
# Undersampling 
# --------------------------------------------------------------------------

class UndersampledCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """
    Calibrating doubly-robust estimators with unbalanced treatment assignment.
    Undersampling changes the class distribution, P(click=1) is biased relative
    to population.

    Final prediction is the average of the calibrated probabilities,
    calibrating first, then averaging. 
    """

    def __init__(self, base_estimator, n_repeats=10, random_state=None):
        self.base_estimator = base_estimator 
        self.n_repeats = n_repeats 
        self.random_state = random_state

    def fit(self, X, y):
        # fit() builds two things that predict_proba() needs 
        # self.gamma_ 
        # self.fitted_estimators_ (the ensemble to avg over)
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.array([0,1])

        minority_idx = np.where(y == 1)[0] # clicks always kept
        majority_idx = np.where(y == 0)[0] # non-clicks, undersampled
        n_minority = len(minority_idx)

        # gamma is the ratio of clicks to non-clicks in the population
        self.gamma_ = n_minority / len(majority_idx)

        rng = np.random.RandomState(self.random_state)
        self.fitted_estimators_ = []

        # each loop iteration draws a fresh random undersample of the 
        # majority class
        for _ in range(self.n_repeats):
            sampled_majority_idx = rng.choice(
                majority_idx, size=n_minority, replace=False
            )
            balanced_idx = np.concatenate([minority_idx, sampled_majority_idx])
            rng.shuffle(balanced_idx)

            estimator = clone(self.base_estimator)
            estimator.fit(X[balanced_idx], y[balanced_idx])
            self.fitted_estimators_.append(estimator)

        return self # return the estimator instance itself 

    def predict_proba(self, X):
        X = np.asarray(X)
        calibrated_probas = []

        for estimator in self.fitted_estimators_:
            # each estimator was trained on a balanced 50/50 sample, 
            # so its raw output p_s is biased toward the minority class
            pos_col = list(estimator.classes_).index(1)
            p_s = estimator.predict_proba(X)[:, pos_col]

            # invert undersampling distortion to recover the calibrated
            # probability on the true population 
            gamma = self.gamma_ 
            p_calibrated = (gamma * p_s) / (gamma * p_s + (1 - p_s))
            calibrated_probas.append(p_calibrated)

        # calibrate-then-average
        p_final = np.mean(calibrated_probas, axis=0)
        return np.column_stack([1 - p_final, p_final])

    def predict(self, X):
        """EconML calls .score() internally during cross-fitting.
        sklearn default ClassifierMixin.score() calls .predict() 
        under the hood."""

        # EconML uses score for interal diagnostics and is not 
        # used in model output
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# --------------------------------------------------------------------------
# Nuisance-model pipelines: target encoding embedded in cross-fitting
# --------------------------------------------------------------------------

# the target encoded columns are no longer features in the traditional
# sense but instead a small model output

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
         works for both the binary outcome (model_y) and the multi-class
         treatment (model_t) without any special-casing).
      2) Passthrough for the already-engineered low-cardinality columns.
      3) The supplied final_estimator (a classifier, since both Y and T
         are discrete here).
 
    Note: EconML internally concatenates X and W into a plain numpy
    array before calling model_y.fit(...) / model_t.fit(...), so column
    selection inside the ColumnTransformer must use integer positions,
    not names. high_card_idx / low_card_idx give those positions and must
    match the column order W was built in. 
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
    # A fresh Pipeline instance created for model_y and  for model_t 
    # EconML clones whatever is passed , but two independent instances 
    # keeps intent unambiguous and avoids any accidental shared 
    # fitted state during debugging.

    # whilst HGBs trees are scale-invariant, logistic regression's C
    # and convergence are scale sensitive, so StandardScalar is requried
    return Pipeline([
        ("preprocess", preprocessor),
        ("scale", StandardScaler()), # used only for logistic regression
        ("model", final_estimator),
    ])
 
 
def build_design_matrix(df: pd.DataFrame, low_card_cols: list[str]):
    """
    Fixes the column order used for W (and therefore the integer indices
    the nuisance pipelines rely on): high-cardinality columns first, then
    the already-engineered low-cardinality block.
    """

    print('Starting build_design_matrix...')
    ordered_cols = HIGH_CARD_COLS + low_card_cols
    W = df[ordered_cols]
    high_card_idx = list(range(len(HIGH_CARD_COLS)))
    low_card_idx = list(range(len(HIGH_CARD_COLS), len(ordered_cols)))
    print('Finished build_design_matrix.')
    return W, high_card_idx, low_card_idx

def compute_overlap_trim_mask(df: pd.DataFrame, 
                              low_card_cols: list[str], 
                              threshold: float = 0.1):
    """
    Fits model_t via out-of-fold cross-validation across the 
    dataset then returns boolean mask for rows with 
    P(observed creative | X) >= threshold

    NOTE: this refits model_t via cross-validation independently of the
    DML fit -- it's a second, separate cost on top of
    what EconML does internally, since trimming has to happen before the
    real fit can run on the trimmed sample.
    """

    # W is the design matrix used to estimate the propensities here
    # high/low card idx tells pipeline which features need encoding
    W, high_card_idx, low_card_idx = build_design_matrix(df, low_card_cols)
    T = df[TREATMENT_COL].to_numpy()

    # pipeline produces out-of-fold propensity scores for trimming
    # (different to the one used for DML estimation)
    pipeline = make_nuisance_pipeline(build_model_t_estimator(), high_card_idx, low_card_idx)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("Fitting model_t out-of-fold across the full dataset for trimming (this refits model_t; expect similar runtime to one nuisance-model pass)...")
    proba = cross_val_predict(pipeline, W, T, cv=cv, method="predict_proba")

    # takes the min across categories, if the min is high then 
    # every categroy has a decent chance of being selected
    min_propensity = proba.min(axis=1)

    n_categories = len(np.unique(T))
    ceiling = 1 / n_categories
    print(f"(Theoretical ceiling for {n_categories} categories: 1/{n_categories} = {ceiling:.4f} -- "
        f"the minimum can never exceed this even under perfectly even assignment, "
        f"so pick threshold relative to that, not a flat number.)")
    print(pd.Series(min_propensity).describe())

    # mask is returned at end of function
    mask = min_propensity >= threshold

    # Summaries before/after diagnostics and per creative
    diag = pd.DataFrame({TREATMENT_COL: T, OUTCOME_COL: df[OUTCOME_COL].to_numpy(), "kept": mask})
    before = diag.groupby(TREATMENT_COL).agg(n_rows_before=(OUTCOME_COL, "size"), n_clicks_before=(OUTCOME_COL, "sum"))
    after = diag[diag["kept"]].groupby(TREATMENT_COL).agg(n_rows_after=(OUTCOME_COL, "size"), n_clicks_after=(OUTCOME_COL, "sum"))
    summary = before.join(after).fillna(0)
    summary["rows_trimmed"] = summary["n_rows_before"] - summary["n_rows_after"]
    summary["clicks_trimmed"] = summary["n_clicks_before"] - summary["n_clicks_after"]
    summary["pct_rows_trimmed"] = (summary["rows_trimmed"] / summary["n_rows_before"] * 100).round(2)
    summary["pct_clicks_trimmed"] = (summary["clicks_trimmed"] / summary["n_clicks_before"] * 100).round(2)
    print(f"\n=== Overlap trim (threshold={threshold}) ===")
    print(summary.to_string()) # as an attempt to view the entire table
    summary.to_csv("overlap_summary_table.csv")
    print(f"\nTotal rows: {len(df):,} -> {mask.sum():,} ({mask.mean():.1%} kept)")
    print(f"Total clicks: {diag[OUTCOME_COL].sum():,} -> {diag.loc[mask, OUTCOME_COL].sum():,}")

    return mask

# --------------------------------------------------------------------------
# Fit DML
# --------------------------------------------------------------------------
def fit_dml(df: pd.DataFrame, low_card_cols: list[str]) -> LinearDML:
    print('Starting fit_dml')
    print('Calling build_design_matrix')
    W, high_card_idx, low_card_idx = build_design_matrix(df, low_card_cols)
 
    model_y_pipeline = make_nuisance_pipeline(
        UndersampledCalibratedClassifier(
            build_model_y_estimator(),
            n_repeats=N_UNDERSAMPLE_REPEATS,
            random_state=RANDOM_STATE,
        ),
        high_card_idx, low_card_idx,
    )

    model_t_pipeline = make_nuisance_pipeline(
            build_model_t_estimator(), high_card_idx, low_card_idx,
        )

 
    Y = df[OUTCOME_COL].to_numpy()
    T = df[TREATMENT_COL].to_numpy()
    global categories 
    categories = ["B"] + [
    c for c in sorted(pd.unique(T).tolist()) if c != "B"
    ]
    # sorting alphabetically resulted in the group with the lowest clicks being the comparison 
    #categories = sorted(pd.unique(T).tolist())
 
    # LinearDML gives an average treatment effect per creative category
    # (vs. a baseline category) with a linear final stage
    # the baseline category is the one at the start of the list

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
        discrete_treatment=True,   # T is the creative variable
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
# Nuisance Model Metrics 
# --------------------------------------------------------------------------
def evaluate_nuisance_model(pipeline: Pipeline, 
                            W: pd.DataFrame, 
                            y: np.ndarray, 
                            label: str, 
                            multi_class: bool,) -> dict:
    """
    Out-of-fold (cross_val_predict, same folds) evaluation.
    AUC / PR-AUC: threshold-independent ranking quality; PR-AUC is the
    primary metric for click given class imbalance.
    AIC / BIC: computed from CV'd (out-of-fold) log-loss rather than
    classical in-sample MLE — this is a stated approximation. It gives a
    generalisation-based figure comparable to a cross-fitted GBM, but k
    (raw coefficient count) is NOT adjusted for the L2 penalty's effective
    degrees of freedom. Flag this as a limitation in your methodology.
    """
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(pipeline, W, y, cv=cv, method="predict_proba")

    fitted = clone(pipeline).fit(W, y)
    model_step = fitted.named_steps["model"]
    # UndersampledCalibratedClassifier wraps an ensemble of identical-
    # architecture LRs; k reflects model COMPLEXITY, not ensemble size,
    # so use one member's coefficient count.
    ref = model_step.fitted_estimators_[0] if hasattr(model_step, "fitted_estimators_") else model_step
    k = np.size(ref.coef_) + np.size(ref.intercept_)
    classes = model_step.classes_
    n = len(y)

    if multi_class:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        y_bin = label_binarize(y, classes=classes)
        pr_auc = average_precision_score(y_bin, proba, average="macro")
    else:
        pos_col = list(classes).index(1)
        auc = roc_auc_score(y, proba[:, pos_col])
        pr_auc = average_precision_score(y, proba[:, pos_col])

    nll_sum = log_loss(y, proba, normalize=False, labels=classes)
    aic = 2 * k + 2 * nll_sum
    bic = k * np.log(n) + 2 * nll_sum

    metrics = {"label": label, "AUC": auc, "PR_AUC": pr_auc, "AIC": aic, "BIC": bic, "k_params": k, "n": n}
    print(f"[{label}] AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  AIC={aic:,.1f}  BIC={bic:,.1f}  (k={k}, n={n:,})")
    return metrics

def build_baseline_relative_table(metrics_list):
    """
    Report-ready table: worst-performing model per metric = 0 (baseline),
    every other value = improvement over worst, so all reported numbers
    are non-negative. AUC/PR-AUC: higher is better -> delta = value - worst.
    AIC/BIC: lower is better -> delta = worst - value (worst is the MAX).
    """
    df_raw = pd.DataFrame(metrics_list).set_index("label")
    higher_is_better = {"AUC": True, "PR_AUC": True, "AIC": False, "BIC": False}
    rows = {}
    for metric, hib in higher_is_better.items():
        worst = df_raw[metric].min() if hib else df_raw[metric].max()
        rows[f"{metric}_raw"] = df_raw[metric]
        rows[f"{metric}_vs_worst"] = (df_raw[metric] - worst) if hib else (worst - df_raw[metric])
    table = pd.DataFrame(rows)
    table.to_csv("nuisance_model_comparison_table.csv")
    print("\n=== Nuisance model comparison (worst = 0 baseline) ===")
    print(table.to_string())
    return table
  
# --------------------------------------------------------------------------
# Report results
# --------------------------------------------------------------------------
def report(est: LinearDML) -> None:
    print("Per-treatment-level average effect on P(click), vs. baseline creative:")
    print(est.summary())
 
 
if __name__ == "__main__":
    df = load_features()
    #print(f"df shape: {df.shape}")
    # Testing
    #df = df.sample(frac=0.1)
    #print(f"df shape: {df.shape}")
    engineered_low_card_cols = [
        c for c in df.columns
        if c not in HIGH_CARD_COLS + [OUTCOME_COL, TREATMENT_COL]
    ]

    W_full, hci, lci = build_design_matrix(df, engineered_low_card_cols)

    tune_logistic_C(W_full, df[OUTCOME_COL].to_numpy(), hci, lci, "click", multi_class=False)
    tune_logistic_C(W_full, df[TREATMENT_COL].to_numpy(), hci, lci, "creative", multi_class=True)

    metrics_click = evaluate_nuisance_model(
        make_nuisance_pipeline(
            UndersampledCalibratedClassifier(build_model_y_estimator(), n_repeats=N_UNDERSAMPLE_REPEATS, random_state=RANDOM_STATE),
            hci, lci,
        ),
        W_full, df[OUTCOME_COL].to_numpy(), "click", multi_class=False,
    )
    metrics_creative = evaluate_nuisance_model(
        make_nuisance_pipeline(build_model_t_estimator(), hci, lci),
        W_full, df[TREATMENT_COL].to_numpy(), "creative", multi_class=True,
    )
    build_baseline_relative_table([metrics_click, metrics_creative])

    mask = compute_overlap_trim_mask(df, engineered_low_card_cols, threshold=0.001)
    df = df.loc[mask].reset_index(drop=True)  # reset_index matters -- same reason build_design_matrix needs clean positional indices

    est = fit_dml(df, engineered_low_card_cols)
    report(est)


    inference = est.intercept__inference()
    point_estimates = inference.point_estimate
    standard_errors = inference.stderr
    zstats = inference.zstat()
    pvalues = inference.pvalue()
    ci = inference.conf_int()  # tuple of (lower, upper) arrays

    print("Point estimates:", point_estimates)
    print("Standard errors:", standard_errors)

    treatment_names = categories[1:]

    summary_df = pd.DataFrame({
        "creative": treatment_names,
        "point_estimate": point_estimates,
        "stderr": standard_errors,
        "zstat": zstats,
        "pvalue": pvalues,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    })

    print(summary_df)
    summary_df.to_csv("dml_ate_full_precision.csv", index=False)


end_time = datetime.now()
elapsed = end_time - start_time

print("\n" + "=" * 80)
print(f"Script finished: {end_time:%Y-%m-%d %H:%M:%S}")
print(f"Total runtime:   {elapsed}")
print("=" * 80)