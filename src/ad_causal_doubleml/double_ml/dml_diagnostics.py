"""

Diagnostics for the models inside dml_ipinyou.py's LinearDML fit.

Checks:
  1. NUISANCE MODEL QUALITY -- does model_y / model_t actually predict
     well out of sample, or is it overfitting / underfitting? 
  2. OVERLAP / POSITIVITY -- for every treatment category, are there rows
     where the estimated propensity of receiving a creative close to 0? 


Neither check feeds back into fit_dml() automatically 
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

from dml_creative import (
    HIGH_CARD_COLS,
    OUTCOME_COL,
    RANDOM_STATE,
    TREATMENT_COL,
    build_design_matrix,
    build_model_t_estimator,
    build_model_y_estimator,
    load_features,
    make_nuisance_pipeline,
    compute_overlap_trim_mask,
    UndersampledCalibratedClassifier,
    N_UNDERSAMPLE_REPEATS
)

TEST_SIZE = 0.3

# --------------------------------------------------------------------------
# model_y (click) quality: is it beating the naive "predict the overall
# click rate for everyone" baseline, and by how much?
# --------------------------------------------------------------------------
def check_outcome_model(W, Y):
    W_train, W_test, Y_train, Y_test = train_test_split(
        W, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y
    )

    # high_card_idx - which cols of W are high cardinality by position
    # works because build_design_matrix() always puts HIGH_CARD_COLS first
    high_card_idx = list(range(len(HIGH_CARD_COLS)))
    low_card_idx = list(range(len(HIGH_CARD_COLS), W.shape[1]))

    pipeline = make_nuisance_pipeline(
        build_model_y_estimator(), high_card_idx, low_card_idx,
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
    print(f"Model Average Preciosion Score (way of summarizing a PR-AUC):                   {average_precision_score(Y_test, p_hat):.4f}"
        "  <- 0.5 = no better than random ranking")
    print(f"Predicted probability range: [{p_hat.min():.6f}, {p_hat.max():.6f}]")
    print(
        "\nInterpretation: if model log-loss is barely better than baseline "
        "log-loss, or AUC is close to 0.5, model_y isn't finding real "
        "structure in the rare click outcome -- residualizing against it "
        "won't remove much confounding, and the resulting Y-residuals may "
        "just be noisy versions of raw Y.\n"
    )

def check_calibration(Y_test, p_hat, model_name, n_bins=10):
    """
    log-loss and AUC are metrics for if model sepeteres clicks 
    from non-clicks well, but not whether the probability values 
    themselves are correct. Does not tell you if the model is 
    well calibrated. 

    If the model is not well calibrated it could be a sign that the 
    re-calibration step post undersampling is not working well. 
    """

    # called by compare_outcome_models() below, once per candidate
    # model, so the two calibration tables that print are directly
    # comparable - same Y_test, same bins
    observed_rate, predicted_rate = calibration_curve(
        Y_test, p_hat, n_bins=n_bins, strategy="quantile"
    )
    print(f"=== Calibration curve: {model_name} ===")
    print(f"{'Predicted (mean)':>20} | {'Observed (actual)':>20}")
    for pred, obs in zip(predicted_rate, observed_rate):
        print(f"{pred:>20.6f} | {obs:>20.6f}")
    print(
        "\nInterpretation: predicted and observed columns should track "
        "each other closely. If predicted is systematically much higher "
        "or lower than observed, probabilities aren't calibrated -- "
        "residualizing Y against them (Y - mu_hat(X)) will introduce "
        "bias, not just noise.\n"
    )

def compare_outcome_models(W, Y):
    """
    Runs the plain HistGradientBoostingClassifier and the
    UndersampledCalibratedClassifier on the SAME held-out split, so their
    metrics are directly comparable.
    """
    W_train, W_test, Y_train, Y_test = train_test_split(
        W, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y
    )
    high_card_idx = list(range(len(HIGH_CARD_COLS)))
    low_card_idx = list(range(len(HIGH_CARD_COLS), W.shape[1]))

    # candidates dict pairs a label with an estimator
    # that has not yet been fitted
    # only the classifier is different between the two runs
    candidates = {
        "plain HGB": build_model_y_estimator(),
        "undersampled + calibrated": UndersampledCalibratedClassifier(
            build_model_y_estimator(),
            n_repeats=N_UNDERSAMPLE_REPEATS,
            random_state=RANDOM_STATE,
        ),
    }

    for name, estimator in candidates.items():
        pipeline = make_nuisance_pipeline(estimator, high_card_idx, low_card_idx)
        pipeline.fit(W_train, Y_train)
        p_hat = pipeline.predict_proba(W_test)[:, 1]

        print(f"\n{'=' * 20} {name} {'=' * 20}")
        print(f"Log-loss:        {log_loss(Y_test, p_hat):.5f}")
        print(f"AUC:             {roc_auc_score(Y_test, p_hat):.4f}")
        print(f"PR-AUC:          {average_precision_score(Y_test, p_hat):.4f}")
        print(f"Predicted range: [{p_hat.min():.6f}, {p_hat.max():.6f}]")
        check_calibration(Y_test, p_hat, name) 


# --------------------------------------------------------------------------
# model_t (creative) quality + overlap/positivity
# --------------------------------------------------------------------------
def check_treatment_model(W, T):
    W_train, W_test, T_train, T_test = train_test_split(
        W, T, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=T
    )
    high_card_idx = list(range(len(HIGH_CARD_COLS)))
    low_card_idx = list(range(len(HIGH_CARD_COLS), W.shape[1]))

    pipeline = make_nuisance_pipeline(
        build_model_t_estimator(), high_card_idx, low_card_idx,
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

    # overlap / positivity
    # For each held-out row, the model's estimated probability of the creative that was shown. 
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

if __name__ == "__main__":
    df = load_features()
    engineered_low_card_cols = [
        c for c in df.columns
        if c not in HIGH_CARD_COLS + [OUTCOME_COL, TREATMENT_COL]
    ]
    # compute overlap mask 
    mask = compute_overlap_trim_mask(df,
                                     engineered_low_card_cols,
                                     threshold = 0.001)
    # apply trimming
    df = df.loc[mask].reset_index(drop=True)

    # build design matrix on trimmed data
    W, _, _ = build_design_matrix(df, engineered_low_card_cols)
    Y = df[OUTCOME_COL].to_numpy()
    T = df[TREATMENT_COL].to_numpy()

    check_outcome_model(W, Y)
    compare_outcome_models(W, Y)
    check_treatment_model(W, T)