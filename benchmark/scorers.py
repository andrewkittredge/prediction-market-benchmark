"""
Scorers for mlflow.genai.evaluate; experiment.py picks which ones each
experiment runs.

`outputs` is whatever predict_fn returned: {"probability": p, "confidence":
c, ...} for the price experiment, {"resolution": label, "confidence": c, ...}
for the outcome-confidence experiment, whose implied probability of the priced
outcome is c when the predicted label is that outcome and 1 - c otherwise.
`inputs` is the record's inputs dict, `expectations` its expectations dict. A
scorer returns None to skip a row, which MLflow leaves out of the run's mean.
"""

from typing import Any

from mlflow.genai.scorers import scorer


def _same_label(a: Any, b: Any) -> bool | None:
    """Whether two outcome labels match, ignoring case and whitespace."""
    if a is None or b is None:
        return None
    return str(a).strip().lower() == str(b).strip().lower()


def _probability(inputs: dict, outputs: Any) -> float | None:
    """
    The forecast as a probability of the priced outcome: the price
    experiment's own "probability", or the outcome-confidence experiment's
    confidence folded through whether its predicted resolution is that
    outcome. None when neither could be parsed.
    """
    if not isinstance(outputs, dict):
        return None if outputs is None else float(outputs)
    if outputs.get("probability") is not None:
        return float(outputs["probability"])
    predicted = _same_label(outputs.get("resolution"), inputs.get("outcome"))
    confidence = outputs.get("confidence")
    if predicted is None or confidence is None:
        return None
    return float(confidence) if predicted else 1 - float(confidence)


@scorer(aggregations=["mean"], name="squared_error")
def squared_error(inputs: dict, outputs: dict, expectations: dict) -> float | None:
    """
    (p - market_probability)^2, the squared error against the market's first
    price; None when no probability could be parsed.
    """
    forecast_probability = _probability(inputs, outputs)
    expected_probability = expectations.get("market_probability")
    if forecast_probability is not None and expected_probability is not None:
        return (forecast_probability - expected_probability) ** 2
    return None


@scorer(aggregations=["mean"], name="squared_error_vs_outcome")
def squared_error_vs_outcome(
    inputs: dict, outputs: dict, expectations: dict
) -> float | None:
    """
    (p - outcome_occurred)^2, the Brier score against what actually happened;
    None while the market is unresolved or no probability could be parsed.
    """
    forecast_probability = _probability(inputs, outputs)
    outcome = expectations.get("outcome_occurred")
    if forecast_probability is not None and outcome is not None:
        return (forecast_probability - outcome) ** 2
    return None


@scorer(aggregations=["mean"], name="resolution_correct")
def resolution_correct(outputs: dict, expectations: dict) -> float | None:
    """
    1 if the predicted resolution label matches the one the market settled
    on, else 0 (the mean is accuracy); None when no label could be parsed or
    the market is unresolved.
    """
    correct = _same_label(outputs.get("resolution"), expectations.get("resolution"))
    if correct is None:
        return None
    return float(correct)
