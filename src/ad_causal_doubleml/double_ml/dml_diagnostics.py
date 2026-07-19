"""
After creating dml_creative.py realised that I wasn't testing the model 
performance before getting results, so this script is a littler of an 
after thought. 


Diagnostics for the nuisance models inside dml_ipinyou.py's LinearDML fit.

Run this BEFORE trusting any DML point estimate. It checks two 
different things:

  1. NUISANCE MODEL QUALITY -- does model_y / model_t actually predict
     well out of sample, or is it overfitting / underfitting? DML's
     cross-fitting protects you from *bias from reusing the same rows*,
     but it does nothing to guarantee the nuisance models are any good --
     a badly-fit model_y or model_t still cross-fits "correctly" and still
     produces a treatment effect estimate, just an unreliable one.

  2. OVERLAP / POSITIVITY -- for every treatment category, are there rows
     where the estimated propensity of receiving the creative they
     actually got is close to 0? DML's residual-on-residual estimator
     divides by treatment-residual variance; when propensities are near
     0 or 1 for some covariate patterns, that variance is tiny and the
     resulting point estimate can become wildly inflated or even flip
     sign relative to the raw data -- which is exactly the pattern seen
     comparing this project's DML output to raw per-creative CTRs.

Neither check feeds back into fit_dml() automatically -- this is a
standalone, run-it-yourself companion script, not part of the estimation
pipeline itself.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from dml_creative import (
    HIGH_CARD_COLS,
    OUTCOME_COL,
    RANDOM_STATE,
    TREATMENT_COL,
    build_design_matrix,
    load_features,
    make_nuisance_pipeline,
)

TEST_SIZE = 0.3


# --------------------------------------------------------------------------
# 1. model_y (click) quality: is it beating the naive "predict the overall
#    click rate for everyone" baseline, and by how much?
# --------------------------------------------------------------------------
def check_outcome_model(W, Y):
    W_train, W_test, Y_train, Y_test = train_test_split(
        W, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y
    )
    high_card_idx = list(range(len(HIGH_CARD_COLS)))
    low_card_idx = list(range(len(HIGH_CARD_COLS), W.shape[1]))

    pipeline = make_nuisance_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        high_card_idx, low_card_idx,
    )
    pipeline.fit(W_train, Y_train)
    p_hat = pipeline.predict_proba(W_test)[:, 1]

    baseline_rate = Y_train.mean()
    baseline_pred = np.full_like(p_hat, baseline_rate)

    print("=== model_y (click) held-out diagnostics ===")
    print(f"Positive rate (train):       {baseline_rate:.5f}")
    print(f"Model log-loss:              {log_loss(Y_test, p_hat):.5f}")
    print(f"Baseline log-loss:           {log_loss(Y_test, baseline_pred):.5f}"
          "  <- predicting the marginal rate for every row, no covariates at all")
    print(f"Model AUC:                   {roc_auc_score(Y_test, p_hat):.4f}"
          "  <- 0.5 = no better than random ranking")
    print(f"Predicted probability range: [{p_hat.min():.6f}, {p_hat.max():.6f}]")
    print(
        "\nInterpretation: if model log-loss is barely better than baseline "
        "log-loss, or AUC is close to 0.5, model_y isn't finding real "
        "structure in the rare click outcome -- residualizing against it "
        "won't remove much confounding, and the resulting Y-residuals may "
        "just be noisy versions of raw Y.\n"
    )


# --------------------------------------------------------------------------
# 2. model_t (creative) quality + overlap/positivity
# --------------------------------------------------------------------------
def check_treatment_model(W, T):
    W_train, W_test, T_train, T_test = train_test_split(
        W, T, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=T
    )
    high_card_idx = list(range(len(HIGH_CARD_COLS)))
    low_card_idx = list(range(len(HIGH_CARD_COLS), W.shape[1]))

    pipeline = make_nuisance_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        high_card_idx, low_card_idx,
    )
    pipeline.fit(W_train, T_train)
    proba = pipeline.predict_proba(W_test)
    classes = pipeline.classes_

    ll = log_loss(T_test, proba, labels=classes)
    marginal = pd.Series(T_train).value_counts(normalize=True).reindex(classes).to_numpy()
    baseline_proba = np.tile(marginal, (len(T_test), 1))
    baseline_ll = log_loss(T_test, baseline_proba, labels=classes)

    print("=== model_t (creative) held-out diagnostics ===")
    print(f"Model multiclass log-loss:    {ll:.5f}")
    print(f"Baseline log-loss (marginal): {baseline_ll:.5f}"
          "  <- predicting the overall creative mix for every row, no covariates")
    print(
        "\nInterpretation: if these are close, model_t isn't learning much "
        "about WHICH creative a row gets beyond the overall mix -- which "
        "would actually be reassuring for overlap (nothing strongly "
        "determines treatment), but bad for removing confounding if "
        "creative assignment genuinely does depend on covariates.\n"
    )

    # --- overlap / positivity ---
    # For each held-out row, the model's estimated probability of the
    # creative that ACTUALLY was shown. A value near 0 means "this
    # covariate pattern almost never gets this creative in the data" --
    # exactly the condition that blows up DML's residual-based estimator.
    class_to_idx = {c: i for i, c in enumerate(classes)}
    observed_idx = np.array([class_to_idx[t] for t in T_test])
    p_observed = proba[np.arange(len(T_test)), observed_idx]

    print("=== Overlap / positivity check ===")
    print("Distribution of P(observed creative | X) on held-out data:")
    print(pd.Series(p_observed).describe())
    print()
    for thresh in (0.05, 0.01, 0.005):
        frac = (p_observed < thresh).mean()
        print(f"  fraction of rows with P(observed creative | X) < {thresh}: {frac:.3%}")

    print("\nPer-creative-category summary of P(observed creative | X):")
    summary = (
        pd.DataFrame({"creative": T_test, "p_observed": p_observed})
        .groupby("creative")["p_observed"]
        .describe()[["mean", "min", "25%", "50%"]]
    )
    print(summary)
    print(
        "\nInterpretation: low 'min' / '25%' values for a given creative "
        "category mean that a meaningful chunk of rows shown that creative "
        "had a covariate profile the model thinks almost never gets it -- "
        "i.e. weak overlap for that category. Categories with the smallest "
        "raw sample sizes (check your earlier value_counts()) are the ones "
        "most likely to show this. If you see it, consider trimming rows "
        "with p_observed below a threshold (e.g. Crump et al. 2009's "
        "common rule of thumb, 0.1) before fitting DML, and reporting the "
        "trim in your methodology -- this is standard practice, not a "
        "hack.\n"
    )


if __name__ == "__main__":
    df = load_features()
    engineered_low_card_cols = [
        c for c in df.columns
        if c not in HIGH_CARD_COLS + [OUTCOME_COL, TREATMENT_COL]
    ]
    W, _, _ = build_design_matrix(df, engineered_low_card_cols)
    Y = df[OUTCOME_COL].to_numpy()
    T = df[TREATMENT_COL].to_numpy()

    check_outcome_model(W, Y)
    check_treatment_model(W, T)