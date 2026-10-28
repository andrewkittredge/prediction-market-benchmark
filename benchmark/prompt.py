"""
The forecasting prompt shown to the model, and the parser for its answer.
"""

from typing import Any

MAX_DESCRIPTION_CHARS = 2500

SYSTEM_PROMPTS = {
    "cutoff": """
You are a careful forecaster. You will be shown a prediction-market question
and a date. Estimate the probability that the named outcome occurs, using only
information that was available on that date — do not use anything you know
about what happened afterwards. Think briefly, then answer on the last line
with a single JSON object of the form {"probability": 0.42}.""",
    "no-cutoff": """
You are a careful forecaster. You will be shown a prediction-market question
and a date. Estimate the probability that the named outcome occurs. Think
briefly, then answer on the last line with a single JSON object of the form
{"probability": 0.42}.""",
}


def build_forecast_prompt(inputs: dict[str, Any]) -> str:
    description = (inputs.get("market_description") or "").strip()
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS] + " […]"
    lines = [
        f"Today's date: {_day(inputs.get('as_of'))}",
        f"Exchange: {inputs.get('exchange')}",
        f"Event: {inputs.get('event_title')}",
        f"Market question: {inputs.get('market_title')}",
    ]
    if inputs.get("market_close"):
        lines.append(f"Market closes: {_day(inputs['market_close'])}")
    if description:
        lines.append(f"Resolution rules:\n{description}")
    lines.append(
        f'Outcome to forecast: "{inputs.get("outcome")}"\n\n'
        "What is the probability that this outcome occurs? Answer on the last "
        'line with {"probability": <number between 0 and 1>}.'
    )
    return "\n\n".join(lines)


def _day(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)[:10]
