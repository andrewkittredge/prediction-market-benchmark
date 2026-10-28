"""
A predict_fn that forecasts with a local Claude Code agent (claude-agent-sdk)
instead of a bare Messages API call.

The SDK runs the installed `claude` CLI as a subprocess, so it uses whatever
that CLI is logged in as — with ANTHROPIC_API_KEY unset, your Claude
subscription rather than the API. The agent gets WebSearch and WebFetch, so
unlike predict.py it can look things up before answering; everything else
(project CLAUDE.md, settings, other tools) is switched off so the forecast
depends only on the prompt.

`model` and `system_prompt` are bound with functools.partial in evaluate.py;
the remaining keyword arguments are the record's inputs.
"""

import asyncio
from typing import Any

import mlflow
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from benchmark.parse_probability import parse_probability
from benchmark.prompt import SYSTEM_PROMPTS, build_forecast_prompt

TOOLS = ["WebSearch", "WebFetch"]
MAX_TURNS = 10


async def ask_agent(model: str, system_prompt: str, prompt: str) -> ResultMessage:
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPTS[system_prompt],
        tools=TOOLS,
        allowed_tools=TOOLS,
        permission_mode="bypassPermissions",
        setting_sources=[],
        max_turns=MAX_TURNS,
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
    return {
        "probability": parse_probability(raw),
        "raw": raw,
        "model": model,
        "num_turns": result.num_turns,
        "cost_usd": result.total_cost_usd,
    }
