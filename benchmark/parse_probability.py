import json
import re

MIN_PROBABILITY = 0.001
MAX_PROBABILITY = 0.999

_JSON_RE = re.compile(r"\{[^{}]*\"probability\"\s*:\s*([0-9]*\.?[0-9]+)\s*%?[^{}]*\}")
_NUMBER_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*(%?)")


def _as_probability(value: float) -> float:
    """A value above 1 was given in percent."""
    if value > 1:
        value /= 100
    return _clamp(value)


def _clamp(p: float) -> float:
    return min(MAX_PROBABILITY, max(MIN_PROBABILITY, p))


def parse_probability(text: str) -> float | None:
    """
    Pull the probability out of the model's reply. Prefers the last JSON object
    with a "probability" key; falls back to the last number in the reply (a
    trailing % is understood). None when nothing usable is found.
    """
    matches = _JSON_RE.findall(text)
    if matches:
        return _as_probability(float(matches[-1]))
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "probability" in parsed:
            return _as_probability(float(parsed["probability"]))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    value, percent = numbers[-1]
    value = float(value)
    if percent:
        value /= 100
    return _as_probability(value)
