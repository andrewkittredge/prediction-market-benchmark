"""
The predict_fn for mlflow.genai.evaluate: one market's inputs in, a forecast out.

`model`, `max_tokens` and `system_prompt` are bound with functools.partial in
evaluate.py; the remaining keyword arguments are the record's inputs.
"""

from typing import Any

import anthropic
import mlflow
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from benchmark.prompt import SYSTEM_PROMPTS, build_forecast_prompt

MAX_TOKENS = 1024

# reads ANTHROPIC_API_KEY from the environment
client = anthropic.Anthropic()


# retry only what can succeed on a second try: rate limits, dropped
# connections and 5xx; a 400 (bad model id, no credit) fails straight away
@retry(
    retry=retry_if_exception_type(
        (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        )
    ),
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=2, max=60),
    reraise=True,
)
def ask(
    model: str, max_tokens: int, system_prompt: str, prompt: str
) -> anthropic.types.ParsedMessage[Forecast]:
    """Structured output: the API constrains the reply to the Forecast schema."""
    return client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPTS[system_prompt],
        messages=[{"role": "user", "content": prompt}],
        output_format=Forecast,
    )


@mlflow.trace(name="forecast")
def predict_api(
    model: str, max_tokens: int, system_prompt: str, **inputs: Any
) -> dict[str, Any]:
    prompt = build_forecast_prompt(inputs)
    message = ask(model, max_tokens, system_prompt, prompt)
    forecast = message.parsed_output
    raw = "".join(block.text for block in message.content if block.type == "text")
    return {
        "probability": None if forecast is None else forecast.probability,
        "raw": raw,
        "model": model,
    }
