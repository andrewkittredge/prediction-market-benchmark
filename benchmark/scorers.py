"""
Scorers for mlflow.genai.evaluate.

`outputs` is whatever predict_fn returned ({"probability": p, ...});
`expectations` is the record's expectations dict. A scorer returns None to
skip a row, which MLflow leaves out of the run's mean.
"""

from typing import Any

from mlflow.genai.scorers import scorer


def _probability(outputs: Any) -> float | None:
    p = outputs.get("probability") if isinstance(outputs, dict) else outputs
    return None if p is None else float(p)


@scorer
def abs_error_vs_market(outputs: Any, expectations: dict) -> float | None:
    """|p - market_probability|; None when no probability could be parsed."""
    p = _probability(outputs)
    if p is None:
        return None
    return abs(p - expectations["market_probability"])


@scorer(aggregations=["mean"], name="brier_score")
def squared_error(outputs: dict, expectations: dict) -> float | None:
    """
    (p - market_probability)^2, the squared error against the market's first
    price; None when no probability could be parsed.
    """

    forecast_probability = outputs.get("probability")
    expected_probability = expectations.get("market_probability")
    if forecast_probability is not None and expected_probability is not None:
        return (forecast_probability - expected_probability) ** 2
    return None


@scorer
def return_output(outputs: Any, expectations: dict) -> float | None:
    """Return the model's output probability for logging purposes."""
    return _probability(outputs)
