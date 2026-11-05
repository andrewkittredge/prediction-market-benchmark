"""
Translates the pandas frame from the `build.py` query into MLflow
evaluation-dataset records: `inputs` is what the model sees (everything known
at `as_of`), `expectations` is what it is scored against, `tags` are copied
onto the evaluation traces.
"""

from datetime import datetime
from typing import Any

import pandas as pd

_INPUT_COLUMNS = (
    "exchange",
    "event_ticker",
    "event_title",
    "market_title",
    "market_description",
    "market_start",
    "market_close",
    "outcome",
    "as_of",
)
_TAG_COLUMNS = ("record_id", "exchange", "event_category", "std_event_type")
# only resolved markets have these; tags must be strings, so they're left out
# of unresolved markets' tags rather than set to None
_RESOLVED_TAG_COLUMNS = ("resolved_at",)


def mlflow_records_from_pandas(df: pd.DataFrame) -> list[dict[str, dict[str, Any]]]:
    # Snowflake upper-cases unquoted column names; nulls arrive as NaN / NaT.
    df = df.rename(columns=str.lower)
    df = df.astype(object).where(df.notna(), None)
    return [_mlflow_record(row) for row in df.to_dict("records")]


def _mlflow_record(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    One MLflow evaluation-dataset record: `inputs` is what the model sees,
    `expectations` what it is scored against, `tags` are copied onto the
    evaluation traces (and so must be strings). Expectations become
    `Expectation` assessments and MLflow rejects `None` there, so an
    unresolved market's `resolution` and `outcome_occurred` are omitted
    rather than null.
    """
    row = {
        **row,
        "record_id": str(row["record_id"]),
        "event_ticker": _none_if_blank(row["event_ticker"]),
        "event_category": _none_if_blank(row["event_category"]),
        "std_event_type": _none_if_blank(row["std_event_type"]),
        "market_description": row["market_description"] or "",
        "market_start": _isoformat(row["market_start"]),
        "market_close": _isoformat(row["market_close"]),
        "as_of": _isoformat(row["as_of"]),
        "resolved_at": _isoformat(row["resolved_at"]),
        "market_probability": float(row["market_probability"]),
    }
    expectations = {"market_probability": row["market_probability"]}
    if row["outcome_occurred"] is not None:
        expectations["resolution"] = row["resolution"]
        expectations["outcome_occurred"] = int(row["outcome_occurred"])
    tags = {k: str(row[k]) for k in _TAG_COLUMNS}
    tags.update({k: row[k] for k in _RESOLVED_TAG_COLUMNS if row[k] is not None})
    return {
        "inputs": {k: row[k] for k in _INPUT_COLUMNS},
        "expectations": expectations,
        "tags": tags,
    }


def _isoformat(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _none_if_blank(value: Any) -> str | None:
    return None if value is None or not str(value).strip() else str(value)
