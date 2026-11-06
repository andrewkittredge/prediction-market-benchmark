import json
import re
from dataclasses import dataclass

MIN_PROBABILITY = 0.001
MAX_PROBABILITY = 0.999

_NUMBER_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*(%?)")


@dataclass
class Forecast:
    probability: float | None
    confidence: float | None
    # the outcome-confidence experiment's predicted resolution label; None for
    # the price experiment
    resolution: str | None


def _json_field_re(key: str) -> re.Pattern[str]:
    """A JSON object containing `key` with a numeric value (a trailing % allowed)."""
    return re.compile(
        r"\{[^{}]*\"" + key + r"\"\s*:\s*([0-9]*\.?[0-9]+)\s*%?[^{}]*\}"
    )


_PROBABILITY_RE = _json_field_re("probability")
_CONFIDENCE_RE = _json_field_re("confidence")
# a JSON object whose "resolution" is a string
_RESOLUTION_RE = re.compile(r"\{[^{}]*\"resolution\"\s*:\s*\"([^\"]*)\"[^{}]*\}")


def _as_probability(value: float) -> float:
    """A value above 1 was given in percent."""
    if value > 1:
        value /= 100
    return _clamp(value)


def _clamp(p: float) -> float:
    return min(MAX_PROBABILITY, max(MIN_PROBABILITY, p))


def _json_field(text: str, pattern: re.Pattern[str], key: str) -> float | None:
    """The field's value from the last matching JSON object in `text`, else None."""
    matches = pattern.findall(text)
    if matches:
        return _as_probability(float(matches[-1]))
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and key in parsed:
            return _as_probability(float(parsed[key]))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def parse_probability(text: str) -> float | None:
    """
    Pull the probability out of the model's reply. Prefers the last JSON object
    with a "probability" key; falls back to the last number in the reply (a
    trailing % is understood), unless that number is a "confidence" value.
    None when nothing usable is found.
    """
    probability = _json_field(text, _PROBABILITY_RE, "probability")
    if probability is not None:
        return probability
    if _CONFIDENCE_RE.search(text):
        return None
    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    value, percent = numbers[-1]
    value = float(value)
    if percent:
        value /= 100
    return _as_probability(value)


def parse_confidence(text: str) -> float | None:
    """The "confidence" value from the reply's JSON object; None when absent."""
    return _json_field(text, _CONFIDENCE_RE, "confidence")


def parse_resolution(text: str) -> str | None:
    """
    The "resolution" of the reply's last JSON object that has one, stripped;
    None when absent or blank.
    """
    matches = _RESOLUTION_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip() or None


def parse_forecast(text: str) -> Forecast:
    return Forecast(
        probability=parse_probability(text),
        confidence=parse_confidence(text),
        resolution=parse_resolution(text),
    )
