"""
One benchmark row: a market frozen at the moment of its first observed price,
plus what an LLM forecast is scored against: the market's own probability and,
once the market has resolved, whether the outcome occurred.

`inputs` is what the model sees (everything known at `as_of`); `expectations`
is what it is compared to. The split mirrors MLflow's evaluation-dataset
record shape so `to_mlflow_record()` is a plain reshuffle.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class BenchmarkRecord:
    # identity: hash(exchange, event_ticker, outcome)
    record_id: str
    # what the model sees
    exchange: str
    event_ticker: str | None
    event_title: str
    market_title: str
    market_description: str
    market_close: datetime | None
    outcome: str
    as_of: datetime
    # what it is scored against
    market_probability: float
    # 1 if the outcome occurred, 0 if not, None while the market is unresolved
    outcome_occurred: int | None
    # trace tag, for filtering runs by category in the UI
    event_category: str | None

    _INPUT_FIELDS = (
        "exchange",
        "event_ticker",
        "event_title",
        "market_title",
        "market_description",
        "market_close",
        "outcome",
        "as_of",
    )
    _EXPECTATION_FIELDS = ("market_probability", "outcome_occurred")
    _TAG_FIELDS = ("record_id", "event_category")

    @classmethod
    def from_row(cls, row: Any) -> "BenchmarkRecord":
        """
        Build from a row of the query in `build_dataset.py` (attribute access,
        lowercase column names).
        """
        return cls(
            record_id=str(row.record_id),
            exchange=row.exchange,
            event_ticker=_none_if_blank(row.event_ticker),
            event_title=row.event_title,
            market_title=row.market_title,
            market_description=row.market_description or "",
            market_close=row.market_close,
            outcome=row.outcome,
            as_of=row.as_of,
            market_probability=float(row.market_probability),
            outcome_occurred=(
                None if row.outcome_occurred is None else int(row.outcome_occurred)
            ),
            event_category=_none_if_blank(row.event_category),
        )

    def to_mlflow_record(self) -> dict[str, dict[str, Any]]:
        d = _jsonable(asdict(self))
        return {
            "inputs": {k: d[k] for k in self._INPUT_FIELDS},
            "expectations": {k: d[k] for k in self._EXPECTATION_FIELDS},
            # tags are copied onto evaluation traces, which want strings
            "tags": {k: str(d[k]) for k in self._TAG_FIELDS},
        }


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value if value.strip() else None


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in d.items()
    }
