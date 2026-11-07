"""
Calibration metrics for the outcome-confidence experiment, after
https://arxiv.org/pdf/2512.16030 (KalshiBench). `forecasts` is the frame
`build_data.load_trace_df` returns, restricted to scoreable rows and with
a `confidence_bin` column (`pd.cut` of `confidence`).
"""

import pandas as pd
from sklearn.metrics import f1_score


def expected_calibration_error(forecasts: pd.DataFrame) -> float:
    """
    Expected calibration error (Naeini et al., 2015): the average gap between
    confidence and accuracy, with forecasts binned by their stated confidence
    into B bins:

        ECE = sum_b |B_b|/N * |acc(B_b) - conf(B_b)|

    A forecast is the label the model predicts the market resolves to,
    `confidence` its stated probability that the label is right, and it is
    accurate (`correct`) when the label matches the market's `resolution`,
    ignoring case. `confidence_bin` is the forecast's bin.
    """
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
    One row of the paper's Table 4 (https://arxiv.org/pdf/2512.16030), which
    reports per model: accuracy (here in percent), macro F1 and the F1 of the
    "yes" class, the Brier score of P(yes) against the outcome, the Brier
    skill score against a forecast of the base rate,
    BSS = 1 - BS / (ybar * (1 - ybar)), and ECE and MCE (the worst bin's gap,
    max_b |acc(B_b) - conf(B_b)|).

    "Yes" means the priced outcome occurs (`outcome_occurred`): the model said
    yes when its predicted resolution was that outcome, and its P(yes) is its
    confidence when it said yes and 1 - confidence otherwise (`predicted_yes`
    and `p_yes`, which `build_data.load_trace_df` adds). N is the number of
    questions answered, which the paper's table doesn't carry.
    """
    predicted_yes = forecasts["predicted_yes"].astype(bool)
    actual_yes = forecasts["outcome_occurred"].astype(bool)
    brier = ((forecasts["p_yes"] - forecasts["outcome_occurred"]) ** 2).mean()
    base_rate = actual_yes.mean()
    brier_climatology = base_rate * (1 - base_rate)
    return pd.Series(
        {
            "N": len(forecasts),
            "Acc": 100 * forecasts["correct"].mean(),
            "F1": f1_score(actual_yes, predicted_yes, average="macro"),
            "F1yes": f1_score(actual_yes, predicted_yes, pos_label=True),
            "Brier": brier,
            "BSS": 1 - brier / brier_climatology,
            "ECE": expected_calibration_error(forecasts),
            "MCE": maximum_calibration_error(forecasts),
        }
    )


def category_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    The paper's Table 2, the category distribution of the evaluation sample:
    per category, the number of markets (N), their share of the sample in
    percent (%) and the share that resolved to the priced outcome (Yes%),
    sorted by N. The sweep holds one forecast per model per market, so
    markets are counted once, by their dataset `record_id`.
    """
    markets = forecasts.drop_duplicates(subset="record_id")
    by_category = markets.groupby("category")
    table = pd.DataFrame(
        {
            "N": by_category.size(),
            "%": 100 * by_category.size() / len(markets),
            "Yes%": 100 * by_category["outcome_occurred"].mean(),
        }
    )
    return table.sort_values("N", ascending=False)
