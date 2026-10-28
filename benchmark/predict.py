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

from benchmark.parse_probability import parse_probability
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
def ask(model: str, max_tokens: int, system_prompt: str, prompt: str) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPTS[system_prompt],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


@mlflow.trace(name="forecast")
def predict(
    model: str, max_tokens: int, system_prompt: str, **inputs: Any
) -> dict[str, Any]:
    prompt = build_forecast_prompt(inputs)
    raw = ask(model, max_tokens, system_prompt, prompt)
    probability = parse_probability(raw)

    return {"probability": probability, "raw": raw, "model": model}
