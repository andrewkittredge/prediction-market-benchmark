"""
Calibration metrics for the outcome-confidence experiment, after
https://arxiv.org/pdf/2512.16030 (KalshiBench). `forecasts` is the frame
`build_data.load_trace_df` returns, restricted to scoreable rows and with
a `confidence_bin` column (`pd.cut` of `confidence`).
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
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


def horizon_label(bucket: pd.Interval) -> str:
    """A `HORIZON_DAYS` bucket as an axis tick: "0–30", ..., ">365"."""
    if bucket.right == float("inf"):
        return f">{bucket.left:g}"
    left = max(bucket.left, 0)
    return f"{left:g}–{bucket.right:g}"


# one hue per model, in fixed order (the dataviz skill's categorical slots 1-3)
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def horizon_plot(table: pd.DataFrame) -> plt.Figure:
    """
    `horizon_table` as a figure: three panels sharing the horizon-bucket
    axis, one each for accuracy, average confidence and ECE, with a line per
    model. Buckets sit at equal spacing although their widths differ. The
    models' cutoffs differ while the markets are shared, so each model covers
    its own range of horizons.
    """
    models = table.index.get_level_values(0).unique()
    buckets = table.index.get_level_values(1).unique().sort_values()
    positions = list(range(len(buckets)))
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    panels = [("Acc", "accuracy (%)"), ("Avg Conf", "avg confidence (%)"), ("ECE", "ECE")]
    for ax, (column, ylabel) in zip(axes, panels):
        for model, color in zip(models, MODEL_COLORS):
            rows = table.xs(model).reindex(buckets)
            ax.plot(positions, rows[column], color=color, marker="o", label=model)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, None)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 100)
    axes[0].legend(frameon=False)
    axes[0].set_title("Accuracy, confidence and calibration by forecast horizon")
    axes[-1].set_xlabel("days from model cutoff to market close")
    axes[-1].set_xticks(positions)
    labels = [horizon_label(b) for b in buckets]
    axes[-1].set_xticklabels(labels)
    fig.tight_layout()
    return fig


def reliability_plot(reliability: pd.DataFrame) -> plt.Figure:
    """
    Reliability diagram of the notebook's `reliability` frame (per model and
    confidence bin: `n`, `conf` and `acc`): each bin's mean confidence
    against its accuracy, one line per model, with the marker area scaled
    by the bin's size and the diagonal as perfect calibration. A point below
    the diagonal is an overconfident bin, above it an underconfident one;
    its vertical distance to the diagonal is the gap ECE averages.
    """
    models = reliability.index.get_level_values(0).unique()
    largest_bin = reliability["n"].max()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", zorder=1)
    for model, color in zip(models, MODEL_COLORS):
        rows = reliability.xs(model)
        ax.plot(rows["conf"], rows["acc"], color=color, label=model, zorder=2)
        sizes = 20 + 300 * rows["n"] / largest_bin
        ax.scatter(
            rows["conf"],
            rows["acc"],
            s=sizes,
            color=color,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title("Reliability diagram (marker area ∝ forecasts in bin)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.9")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def category_accuracy_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Accuracy and confidence per model and market category: the number of
    forecasts (N), accuracy and average confidence in percent, and the ECE.
    """
    return accuracy_and_confidence(forecasts, ["model", "category"])


def category_plot(table: pd.DataFrame) -> plt.Figure:
    """
    `category_accuracy_table` as a dumbbell chart: one panel per model, one
    row per market category, a filled dot at the accuracy and a hollow dot at
    the average confidence joined by a line, so the line is the gap between
    them (confidence to the right of accuracy means overconfidence). The
    number of forecasts is printed at the right of each row, since models
    cover the categories unevenly.
    """
    models = table.index.get_level_values(0).unique()
    categories = table.index.get_level_values(1).unique().sort_values()
    positions = list(range(len(categories)))
    fig, axes = plt.subplots(
        1, len(models), sharey=True, figsize=(3.2 * len(models), 4)
    )
    for ax, model, color in zip(axes, models, MODEL_COLORS):
        rows = table.xs(model).reindex(categories)
        ax.hlines(positions, rows["Acc"], rows["Avg Conf"], color=color, linewidth=2)
        ax.scatter(rows["Acc"], positions, color=color, s=60, zorder=3)
        ax.scatter(
            rows["Avg Conf"],
            positions,
            facecolor="white",
            edgecolor=color,
            linewidth=2,
            s=60,
            zorder=3,
        )
        for position, n in zip(positions, rows["N"]):
            if pd.notna(n):
                ax.text(102, position, f"n={n:.0f}", va="center", fontsize=8, color="0.4")
        ax.set_title(model, fontsize=10)
        ax.set_xlim(40, 100)
        ax.set_xlabel("%")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="0.9")
        ax.set_axisbelow(True)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(categories)
    axes[0].invert_yaxis()
    filled = Line2D([], [], color="0.4", marker="o", linestyle="", label="accuracy")
    hollow = Line2D(
        [],
        [],
        markeredgecolor="0.4",
        markerfacecolor="white",
        marker="o",
        linestyle="",
        label="avg confidence",
    )
    fig.legend(handles=[filled, hollow], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Accuracy vs. confidence by market category")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


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
