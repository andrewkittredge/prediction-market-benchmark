"""
Pulls a finished evaluate.py sweep back out of MLflow for analysis: the
parent run's child runs (one per model), every trace of each
(one per predict call), and the traces flattened into one row per forecast.
"""

import mlflow
import pandas as pd

# number of equal-width confidence bins for the calibration metrics
CONFIDENCE_BINS = 10


def child_runs(experiment_id: str, run_name: str | None) -> pd.DataFrame:
    """
    One row per child run of every parent run named `run_name` (a sweep
    continued over several invocations with the same --run-name has several),
    or of every parent run in the experiment when it is None, with each
    scorer's mean over the run; the model is only recorded in the run name.
    """
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    # only child runs carry the parentRunId tag, which holds the parent's
    # run_id, not its name
    parent_ids = runs["tags.mlflow.parentRunId"]
    if run_name is None:
        return runs[parent_ids.notna()]
    parents = runs[runs["tags.mlflow.runName"] == run_name]
    return runs[parent_ids.isin(parents["run_id"])]


def search_traces(experiment_id: str, runs: pd.DataFrame) -> pd.DataFrame:
    """Every trace of every run in `runs`, tagged with the run it came from."""
    return pd.concat(
        [
            mlflow.search_traces(locations=[experiment_id], run_id=run_id).assign(
                run=name
            )
            for run_id, name in zip(runs["run_id"], runs["tags.mlflow.runName"])
        ],
        ignore_index=True,
    )


def assessment_value(assessments: list, name: str) -> float | None:
    # scorer outputs are stored as "feedback", dataset expectations (the
    # market's own probability) as "expectation"
    for a in assessments:
        if a["assessment_name"] == name:
            return (a.get("feedback") or a["expectation"])["value"]
    return None


def label_matches(predicted: str | None, actual: str | None) -> bool | None:
    """Whether a predicted resolution label is the actual one, ignoring case."""
    if predicted is None or actual is None:
        return None
    return predicted.strip().lower() == actual.strip().lower()


def trace_frame(traces: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """
    One row per forecast. The record's inputs sit under request["inputs"],
    the market category is a tag copied onto the trace, and each scorer's
    value is an assessment on the trace.
    """
    # the outcome-confidence experiment's predicted resolution label; None for
    # the price experiment
    predicted_resolution = traces["response"].map(lambda r: r and r.get("resolution"))
    # the label the market settled on: "yes"/"no" on Kalshi, "Yes"/"No" or the
    # winning outcome's name (e.g. "Up", a candidate) on Polymarket
    resolution = traces["assessments"].map(lambda a: assessment_value(a, "resolution"))
    matches = [
        label_matches(predicted, actual)
        for predicted, actual in zip(predicted_resolution, resolution)
    ]
    return pd.DataFrame(
        {
            "run": traces["run"],
            # the dataset record (market and priced outcome) the forecast is on
            "record_id": traces["tags"].map(lambda t: t["record_id"]),
            "request_time": pd.to_datetime(traces["request_time"], unit="ms"),
            "model": traces["request"].map(lambda r: r["model"]),
            # sweeps from before experiment.py recorded the prompt's name as
            # "system_prompt" instead
            "experiment": traces["request"].map(
                lambda r: r.get("experiment") or r["system_prompt"]
            ),
            "exchange": traces["tags"].map(lambda t: t.get("exchange")),
            "category": traces["tags"].map(lambda t: t.get("event_category")),
            "event_type": traces["tags"].map(lambda t: t.get("std_event_type")),
            "market_title": traces["request"].map(
                lambda r: r["inputs"]["market_title"]
            ),
            "response": traces["response"].map(lambda r: r and r["raw"]),
            "as_of": traces["request"].map(
                lambda r: pd.Timestamp(r["inputs"]["as_of"])
            ),
            "predicted_probability": traces["response"].map(
                lambda r: r and r["probability"]
            ),
            # the price experiment's confidence is the model's probability that
            # its estimate is within CONFIDENCE_TOLERANCE of the market price,
            # the outcome-confidence experiment's that its predicted resolution
            # is right
            "confidence": traces["response"].map(lambda r: r and r.get("confidence")),
            "predicted_resolution": predicted_resolution,
            "market_probability": traces["assessments"].map(
                lambda a: assessment_value(a, "market_probability")
            ),
            # what actually happened; None while the market is unresolved
            "outcome_occurred": traces["assessments"]
            .map(lambda a: assessment_value(a, "outcome_occurred"))
            .astype(bool),
            "resolution": resolution,
            # whether the predicted resolution matched; <NA> without one (the
            # price experiment) or while the market is unresolved
            "correct": pd.array(matches, dtype="boolean"),
            "state": traces["state"],
        }
    )


def load_trace_df(experiment_name: str, run_name: str | None = None) -> pd.DataFrame:
    """
    The scoreable forecasts of the sweep `run_name` in `experiment_name`, or
    of every sweep in it when no `run_name` is given: one row per trace that
    returned a confidence (errored traces have no response), minus any
    predicted resolution that cannot be checked against the market's
    resolution. A sweep continued under the same run name re-evaluates the
    same sample, so each model keeps only its latest forecast per
    `record_id`. Adds the "yes" view of each forecast: `predicted_yes`, whether
    the model called the priced outcome, and `p_yes`, its probability of it;
    and `confidence_bin`, the forecast's bin among `CONFIDENCE_BINS`
    equal-width confidence bins.
    """
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id  # pyright: ignore[reportOptionalMemberAccess]
    runs = child_runs(experiment_id, run_name)
    traces = search_traces(experiment_id, runs)
    forecasts = trace_frame(traces, runs)
    answered = forecasts["confidence"].notna()
    # the price experiment predicts no resolution, so its `correct` is always
    # <NA>; only drop a forecast that made a prediction it can't be scored on
    unscoreable = (
        forecasts["predicted_resolution"].notna() & forecasts["correct"].isna()
    )
    forecasts = forecasts[answered & ~unscoreable]
    forecasts = forecasts.sort_values("request_time")
    forecasts = forecasts.drop_duplicates(subset=["model", "record_id"], keep="last")
    forecasts = forecasts.copy()
    # the model called the priced outcome: its label was right on a market
    # where that outcome occurred, or wrong where it didn't; <NA> for the
    # price experiment
    forecasts["predicted_yes"] = forecasts["correct"] == forecasts["outcome_occurred"]
    forecasts["p_yes"] = forecasts["confidence"].where(
        forecasts["predicted_yes"], 1 - forecasts["confidence"]
    )
    breaks = [b / CONFIDENCE_BINS for b in range(CONFIDENCE_BINS + 1)]
    confidence_bins = pd.IntervalIndex.from_breaks(breaks)
    forecasts["confidence_bin"] = pd.cut(
        forecasts["confidence"], confidence_bins, include_lowest=True
    )
    return forecasts
