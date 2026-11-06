"""
The forecasting prompts shown to the model: each names a task (the market's
opening price as a probability, or a yes/no call on whether the outcome
actually occurs) as a system prompt plus the question that closes the user
prompt. Both tasks also ask for a 0-1 confidence in the answer.
"""

from dataclasses import dataclass
from typing import Any

MAX_DESCRIPTION_CHARS = 2500

# for the price prompts, the model's "confidence" is its probability that its
# price estimate lands within this distance of the market's actual opening
# price; for the outcome prompts it is its probability that its yes/no call
# is right
CONFIDENCE_TOLERANCE = 0.10

_PRICE_ANSWER_FORMAT = f"""Think briefly, then answer on the last line with a single
JSON object of the form {{"probability": 0.42, "confidence": 0.7}}, where
"probability" is your estimate of the opening price and "confidence" is your
probability that this estimate is within {CONFIDENCE_TOLERANCE} of the actual
opening price."""

_PRICE_QUESTION = (
    "What price did the market open at for this outcome, and how confident "
    f"are you that your estimate is within {CONFIDENCE_TOLERANCE} of the "
    "actual opening price? Answer on the last line with "
    '{"probability": <number between 0 and 1>, '
    '"confidence": <number between 0 and 1>}.'
)

_OUTCOME_ANSWER_FORMAT = """Think briefly, then answer on the last line with a
single JSON object of the form {"outcome": true, "confidence": 0.8}, where
"outcome" is true if you predict the outcome occurs (the market resolves to
it) and false otherwise, and "confidence" is your probability (between 0 and
1) that this prediction is right."""

_OUTCOME_QUESTION = (
    "Will this outcome occur, i.e. will the market resolve to it, and how "
    "confident are you in that prediction? Answer on the last line with "
    '{"outcome": <true or false>, "confidence": <number between 0 and 1>}.'
)


@dataclass(frozen=True)
class Prompt:
    system: str
    # closes the user prompt built by build_forecast_prompt
    question: str


PROMPTS = {
    "price": Prompt(
        system=f"""
You are a careful forecaster. You will be shown a prediction-market question
and the date the market opened. Estimate the price (a probability between 0
and 1) that the market opened at for the named outcome on that date.
{_PRICE_ANSWER_FORMAT}""",
        question=_PRICE_QUESTION,
    ),
    "outcome": Prompt(
        system=f"""
You are a careful forecaster. You will be shown a prediction-market question
and the date the market opened. Predict whether the named outcome actually
occurs. {_OUTCOME_ANSWER_FORMAT}""",
        question=_OUTCOME_QUESTION,
    ),
}


def build_forecast_prompt(inputs: dict[str, Any], question: str) -> str:
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
    lines.append(f'Outcome priced: "{inputs.get("outcome")}"\n\n{question}')
    return "\n\n".join(lines)


def _day(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)[:10]
