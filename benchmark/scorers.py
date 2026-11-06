"""
Scorers for mlflow.genai.evaluate; experiment.py picks which ones each
experiment runs.

`outputs` is whatever predict_fn returned: {"probability": p, "confidence":
c, ...} for the price experiment, {"outcome": bool, "confidence": c, ...} for
the outcome experiment, whose implied probability is c when the call is yes
and 1 - c otherwise. `expectations` is the record's expectations dict. A scorer returns None to
skip a row, which MLflow leaves out of the run's mean.
"""

from typing import Any

from mlflow.genai.scorers import scorer


def _probability(outputs: Any) -> float | None:
    """
    The forecast as a probability of the outcome: the price experiment's own
    "probability", or the outcome experiment's confidence folded through its
    yes/no call. None when neither could be parsed.
    """
    if not isinstance(outputs, dict):
        return None if outputs is None else float(outputs)
    if outputs.get("probability") is not None:
        return float(outputs["probability"])
    outcome, confidence = outputs.get("outcome"), outputs.get("confidence")
    if outcome is None or confidence is None:
        return None
    return float(confidence) if outcome else 1 - float(confidence)


@scorer(aggregations=["mean"], name="squared_error")
def squared_error(outputs: dict, expectations: dict) -> float | None:
    """
    (p - market_probability)^2, the squared error against the market's first
    price; None when no probability could be parsed.
    """

    forecast_probability = _probability(outputs)
    expected_probability = expectations.get("market_probability")
    if forecast_probability is not None and expected_probability is not None:
        return (forecast_probability - expected_probability) ** 2
    return None


@scorer(aggregations=["mean"], name="squared_error_vs_outcome")
def squared_error_vs_outcome(outputs: dict, expectations: dict) -> float | None:
    """
    (p - outcome_occurred)^2, the Brier score against what actually happened;
    None while the market is unresolved or no probability could be parsed.
    """
    forecast_probability = _probability(outputs)
    outcome = expectations.get("outcome_occurred")
    if forecast_probability is not None and outcome is not None:
        return (forecast_probability - outcome) ** 2
    return None


@scorer(aggregations=["mean"], name="outcome_correct")
def outcome_correct(outputs: dict, expectations: dict) -> float | None:
    """
    1 if the yes/no call matched what happened, else 0 (the mean is
    accuracy); None when no call could be parsed.
    """
    call = outputs.get("outcome")
    outcome = expectations.get("outcome_occurred")
    if call is None or outcome is None:
        return None
    return float(call == bool(outcome))
