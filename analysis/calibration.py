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


def overconfidence_rate(forecasts: pd.DataFrame, threshold: float) -> float:
    """
    OCR@threshold (the paper's eq. 6): the share of forecasts with confidence
    above `threshold` that were wrong.
    """
    confident = forecasts[forecasts["confidence"] > threshold]
    return 1 - confident["correct"].mean()


def confidence_metrics(forecasts: pd.DataFrame) -> pd.Series:
    """
    One row of the paper's Table 5, the confidence analysis of one model:
    its average confidence (Avg Conf), its average confidence on the
    forecasts it got wrong (Conf_wrong), and its overconfidence rate above
    70%, 80% and 90% confidence (OCR@70, OCR@80, OCR@90), all in percent.
    The paper reads the table as showing models stay confident when wrong
    and are wrong far too often when they claim 90%+.
    """
    wrong = forecasts[~forecasts["correct"].astype(bool)]
    return pd.Series(
        {
            "Avg Conf": 100 * forecasts["confidence"].mean(),
            "Conf_wrong": 100 * wrong["confidence"].mean(),
            "OCR@70": 100 * overconfidence_rate(forecasts, 0.7),
            "OCR@80": 100 * overconfidence_rate(forecasts, 0.8),
            "OCR@90": 100 * overconfidence_rate(forecasts, 0.9),
        }
    )


# upper edges, in days, of the horizon buckets of `horizon_table`
HORIZON_DAYS = [0, 30, 60, 90, 180, 365, float("inf")]


def horizon_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Accuracy and confidence as a function of how far past the model's
    training cutoff the market closed: per model and bucket of days from
    `cutoff` to `market_close` (`HORIZON_DAYS`), the number of forecasts (N),
    accuracy and average confidence in percent, and the ECE. A widening gap
    between Acc and Avg Conf, or a rising ECE, with the horizon means the
    model's confidence doesn't account for how far it is forecasting past
    what it knows.
    """
    horizon = (forecasts["market_close"] - forecasts["cutoff"]).dt.days
    bucket = pd.cut(horizon, HORIZON_DAYS, include_lowest=True)
    return accuracy_and_confidence(forecasts, ["model", bucket])


def category_accuracy_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Accuracy and confidence per model and market category: the number of
    forecasts (N), accuracy and average confidence in percent, and the ECE.
    """
    return accuracy_and_confidence(forecasts, ["model", "category"])


def accuracy_and_confidence(forecasts: pd.DataFrame, by: list) -> pd.DataFrame:
    """N, Acc, Avg Conf (percent) and ECE of `forecasts` grouped by `by`."""
    groups = forecasts.groupby(by, observed=True)
    return pd.DataFrame(
        {
            "N": groups.size(),
            "Acc": 100 * groups["correct"].mean(),
            "Avg Conf": 100 * groups["confidence"].mean(),
            "ECE": groups.apply(expected_calibration_error, include_groups=False),
        }
    )
