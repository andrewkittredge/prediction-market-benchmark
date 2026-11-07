"""
Calibration metrics for the outcome-confidence experiment, after
https://arxiv.org/pdf/2512.16030 (KalshiBench). `forecasts` is the frame
`build_data.load_trace_df` returns, restricted to scoreable rows and with
a `confidence_bin` column (`pd.cut` of `confidence`).
"""

import pandas as pd
from sklearn.metrics import f1_score


def expected_calibration_error(forecasts: pd.DataFrame) -> float:
    """sum_b |B_b|/N * |acc(B_b) - conf(B_b)| over the confidence bins."""
    bins = forecasts.groupby("confidence_bin", observed=True)
    acc = bins["correct"].mean()
    conf = bins["confidence"].mean()
    weight = bins.size() / len(forecasts)
    return (weight * (acc - conf).abs()).sum()


def maximum_calibration_error(forecasts: pd.DataFrame) -> float:
    """max_b |acc(B_b) - conf(B_b)| over the confidence bins."""
    bins = forecasts.groupby("confidence_bin", observed=True)
    acc = bins["correct"].mean()
    conf = bins["confidence"].mean()
    return (acc - conf).abs().max()


def kalshibench_metrics(forecasts: pd.DataFrame) -> pd.Series:
    """
    One row of the paper's Table 4: accuracy in percent, the rest as is.
    Needs `predicted_yes` (the model called the priced outcome) and `p_yes`
    (its probability of that outcome) alongside `outcome_occurred`.
    """
    predicted_yes = forecasts["predicted_yes"].astype(bool)
    actual_yes = forecasts["outcome_occurred"].astype(bool)
    brier = ((forecasts["p_yes"] - forecasts["outcome_occurred"]) ** 2).mean()
    base_rate = actual_yes.mean()
    brier_climatology = base_rate * (1 - base_rate)
    return pd.Series(
        {
            "Acc": 100 * forecasts["correct"].mean(),
            "F1": f1_score(actual_yes, predicted_yes, average="macro"),
            "F1yes": f1_score(actual_yes, predicted_yes, pos_label=True),
            "Brier": brier,
            "BSS": 1 - brier / brier_climatology,
            "ECE": expected_calibration_error(forecasts),
            "MCE": maximum_calibration_error(forecasts),
        }
    )
