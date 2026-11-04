"""
Pulls a finished evaluate.py sweep back out of MLflow for analysis: the
parent run's child runs (one per model x system prompt), every trace of each
(one per predict call), and the traces flattened into one row per forecast.
"""

import mlflow
import pandas as pd


def child_runs(experiment_id: str, run_name: str) -> pd.DataFrame:
    """
    One row per child run of the parent run named `run_name`, with each
    scorer's mean over the run; the model is only recorded in the run name.
    """
    # the parentRunId tag holds the parent's run_id, not its name, so look the
    # parent up by name first
    parent = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    if len(parent) != 1:
        raise ValueError(f"expected one run named {run_name!r}, found {len(parent)}")
    return mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.run_id.iloc[0]}'",
    )


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


def trace_frame(traces: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """
    One row per forecast. The record's inputs sit under request["inputs"],
    the market category is a tag copied onto the trace, and each scorer's
    value is an assessment on the trace.
    """
    return pd.DataFrame(
        {
            "run": traces["run"],
            "model": traces["request"].map(lambda r: r["model"]),
            "system_prompt": traces["request"].map(lambda r: r["system_prompt"]),
            # a tag on the child run rather than on the trace, as "YYYY-MM"; taken
            # as the last day of that month so it compares against as_of
            "training_data_cutoff": pd.PeriodIndex(
                traces["run"].map(
                    dict(
                        zip(runs["tags.mlflow.runName"], runs["tags.training_data_cutoff"])
                    )
                ),
                freq="M",
            )
            .to_timestamp(how="end")
            .normalize(),
            "exchange": traces["tags"].map(lambda t: t.get("exchange")),
            "category": traces["tags"].map(lambda t: t.get("event_category")),
            "event_type": traces["tags"].map(lambda t: t.get("std_event_type")),
            "market_title": traces["request"].map(lambda r: r["inputs"]["market_title"]),
            "response": traces["response"].map(lambda r: r and r["raw"]),
            "as_of": traces["request"].map(lambda r: pd.Timestamp(r["inputs"]["as_of"])),
            "predicted_probability": traces["response"].map(
                lambda r: r and r["probability"]
            ),
            # the model's probability that its estimate is within
            # CONFIDENCE_TOLERANCE of the market price; absent in older runs
            "confidence": traces["response"].map(lambda r: r and r.get("confidence")),
            "market_probability": traces["assessments"].map(
                lambda a: assessment_value(a, "market_probability")
            ),
            "state": traces["state"],
        }
    )


def load_trace_df(experiment_name: str, run_name: str) -> pd.DataFrame:
    """The flattened forecasts of the sweep `run_name` in `experiment_name`."""
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    runs = child_runs(experiment_id, run_name)
    return trace_frame(search_traces(experiment_id, runs), runs)
