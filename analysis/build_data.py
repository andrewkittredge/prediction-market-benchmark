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
    """
    Every trace of every run in `runs`, tagged with the run it came from and
    the run's model cutoff date (the `cutoff` param, None when the model was
    evaluated on every row).
    """
    frames = [
        mlflow.search_traces(locations=[experiment_id], run_id=run_id).assign(
            run=name, cutoff=cutoff
        )
        for run_id, name, cutoff in zip(
            runs["run_id"], runs["tags.mlflow.runName"], runs["params.cutoff"]
        )
    ]
    return pd.concat(frames, ignore_index=True)


def assessment_value(assessments: list, name: str) -> float | None:
    # scorer outputs are stored as "feedback", dataset expectations (the
    # market's own probability) as "expectation"
    for a in assessments:
        if a["assessment_name"] == name:
            return (a.get("feedback") or a["expectation"])["value"]
    return None


def trace_frame(traces: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """
    One row per scoreable forecast, with only the columns `calibration.ipynb`
    uses. The record's inputs sit under request["inputs"], the market
    category is a tag copied onto the trace, and each scorer's value is an
    assessment on the trace.

    Keeps the traces that returned a confidence, minus any predicted
    resolution that cannot be checked against the market's resolution. A
    sweep continued under the same run name re-evaluates the same sample, so
    each model keeps only its latest forecast per `record_id`. Adds the "yes"
    view of each forecast: `predicted_yes`, whether the model called the
    priced outcome, and `p_yes`, its probability of it; and `confidence_bin`,
    the forecast's bin among `CONFIDENCE_BINS` equal-width confidence bins.
    """
    # the outcome-confidence experiment's predicted resolution label; None for
    # the price experiment
    predicted_resolution = traces["response"].map(lambda r: r and r.get("resolution"))
    # the `resolution_correct` scorer's 1.0/0.0, absent (None) for the price
    # experiment, an unparsed label or an unresolved market
    resolution_correct = (
        traces["assessments"]
        .map(lambda a: assessment_value(a, "resolution_correct"))
        .astype(bool)
    )
    forecasts = pd.DataFrame(
        {
            # the model's training cutoff the run was restricted to
            "cutoff": pd.to_datetime(traces["cutoff"]),
            # the dataset record (market and priced outcome) the forecast is on
            "record_id": traces["tags"].map(lambda t: t["record_id"]),
            "request_time": pd.to_datetime(traces["request_time"], unit="ms"),
            "model": traces["request"].map(lambda r: r["model"]),
            "category": traces["tags"].map(lambda t: t.get("event_category")),
            "market_close": traces["request"].map(
                lambda r: pd.Timestamp(r["inputs"]["market_close"])
            ),
            # the price experiment's confidence is the model's probability that
            # its estimate is within CONFIDENCE_TOLERANCE of the market price,
            # the outcome-confidence experiment's that its predicted resolution
            # is right
            "confidence": traces["response"].map(lambda r: r and r.get("confidence")),
            "predicted_resolution": predicted_resolution,
            # what actually happened; None while the market is unresolved
            "outcome_occurred": traces["assessments"]
            .map(lambda a: assessment_value(a, "outcome_occurred"))
            .astype(bool),
            # whether the predicted resolution matched; <NA> without one (the
            # price experiment) or while the market is unresolved
            "correct": resolution_correct,
        }
    )
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


def load_trace_df(experiment_name: str, run_name: str | None = None) -> pd.DataFrame:
    """
    The scoreable forecasts (see `trace_frame`) of the sweep `run_name` in
    `experiment_name`, or of every sweep in it when no `run_name` is given,
    from the traces that didn't error and carry a `record_id` tag.
    """
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id  # pyright: ignore[reportOptionalMemberAccess]
    runs = child_runs(experiment_id, run_name)
    traces = search_traces(experiment_id, runs)
    traces = traces[traces["state"] != "ERROR"]
    # a few early traces predate the record_id tag and can't be matched to a
    # dataset record
    has_record_id = traces["tags"].map(lambda t: "record_id" in t)
    traces = traces[has_record_id]
    return trace_frame(traces, runs)


if __name__ == "__main__":
    load_trace_df("outcome-confidence", run_name="2k-balanced")
