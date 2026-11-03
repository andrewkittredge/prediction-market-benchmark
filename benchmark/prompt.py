"""
The forecasting prompt shown to the model, and the parser for its answer.
"""

from typing import Any

MAX_DESCRIPTION_CHARS = 2500

# the model's "confidence" is its probability that its price estimate lands
# within this distance of the market's actual opening price
CONFIDENCE_TOLERANCE = 0.10

_ANSWER_FORMAT = f"""Think briefly, then answer on the last line with a single
JSON object of the form {{"probability": 0.42, "confidence": 0.7}}, where
"probability" is your estimate of the opening price and "confidence" is your
probability that this estimate is within {CONFIDENCE_TOLERANCE} of the actual
opening price."""

SYSTEM_PROMPTS = {
    "cutoff": f"""
You are a careful forecaster. You will be shown a prediction-market question
and the date the market opened. Estimate the price (a probability between 0
and 1) that the market opened at for the named outcome on that date, using
only information that was available on that date — do not use anything you
know about what happened afterwards. {_ANSWER_FORMAT}""",
    "no-cutoff": f"""
You are a careful forecaster. You will be shown a prediction-market question
and the date the market opened. Estimate the price (a probability between 0
and 1) that the market opened at for the named outcome on that date.
{_ANSWER_FORMAT}""",
}


def build_forecast_prompt(inputs: dict[str, Any]) -> str:
    description = (inputs.get("market_description") or "").strip()
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS] + " […]"
    lines = [
        f"Market open date: {_day(inputs.get('as_of'))}",
        f"Exchange: {inputs.get('exchange')}",
        f"Event: {inputs.get('event_title')}",
        f"Market question: {inputs.get('market_title')}",
    ]
    if inputs.get("market_close"):
        lines.append(f"Market closes: {_day(inputs['market_close'])}")
    if description:
        lines.append(f"Resolution rules:\n{description}")
    lines.append(
        f'Outcome priced: "{inputs.get("outcome")}"\n\n'
        "What price did the market open at for this outcome, and how confident "
        f"are you that your estimate is within {CONFIDENCE_TOLERANCE} of the "
        "actual opening price? Answer on the last line with "
        '{"probability": <number between 0 and 1>, '
        '"confidence": <number between 0 and 1>}.'
    )
    return "\n\n".join(lines)


def _day(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)[:10]
