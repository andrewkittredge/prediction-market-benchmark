"""
A predict_fn that forecasts with a local Claude Code agent (claude-agent-sdk)
instead of a bare Messages API call.

The SDK runs the `claude` CLI bundled inside the claude-agent-sdk package as
a subprocess (not the globally installed one), so it uses whatever
that CLI is logged in as — with ANTHROPIC_API_KEY unset, your Claude
subscription rather than the API. The agent gets no tools (no web search:
looking up what happened after as_of would leak the answer), and project
CLAUDE.md and settings are switched off, so the forecast depends only on the
prompt and the model's own knowledge.

`model` and `system_prompt` are bound with functools.partial in evaluate.py;
the remaining keyword arguments are the record's inputs.
"""

import asyncio
from typing import Any

import mlflow
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from benchmark.parse_probability import parse_forecast
from benchmark.prompt import SYSTEM_PROMPTS, build_forecast_prompt

TOOLS: list[str] = []


async def ask_agent(model: str, system_prompt: str, prompt: str) -> ResultMessage:
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPTS[system_prompt],
        tools=TOOLS,
        allowed_tools=TOOLS,
        setting_sources=[],
    )
    # the ResultMessage is the last message; drain the iterator rather than
    # returning from inside it, which closes the SDK's generator mid-run
    result = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    if result is None:
        raise RuntimeError("agent finished without a ResultMessage")
    return result


@mlflow.trace(name="forecast")
def predict_with_agent(model: str, system_prompt: str, **inputs: Any) -> dict[str, Any]:
    prompt = build_forecast_prompt(inputs)
    # evaluate() calls predict_fn from worker threads, each of which gets its
    # own event loop here
    result = asyncio.run(ask_agent(model, system_prompt, prompt))
    raw = result.result or ""
    if result.is_error:
        raise RuntimeError(f"agent run failed ({result.subtype}): {result.errors}")
    forecast = parse_forecast(raw)
    return {
        "probability": forecast.probability,
        "confidence": forecast.confidence,
        "raw": raw,
        "model": model,
        "num_turns": result.num_turns,
        "cost_usd": result.total_cost_usd,
    }
