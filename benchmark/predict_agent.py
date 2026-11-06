"""
The predict_fn for mlflow.genai.evaluate: forecasts with a local Claude Code
agent (claude-agent-sdk).

The SDK runs the `claude` CLI bundled inside the claude-agent-sdk package as
a subprocess (not the globally installed one), so it uses whatever
that CLI is logged in as — with ANTHROPIC_API_KEY unset, your Claude
subscription rather than the API. The agent gets no tools (no web search:
looking up what happened after as_of would leak the answer), and project
CLAUDE.md and settings are switched off, so the forecast depends only on the
prompt and the model's own knowledge.

`model` and `experiment` (a key of experiment.EXPERIMENTS) are bound with
functools.partial in evaluate.py; the remaining keyword arguments are the
record's inputs.
"""

import asyncio
from typing import Any

import mlflow
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from benchmark.experiment import EXPERIMENTS
from benchmark.parse_probability import parse_forecast
from benchmark.prompt import build_forecast_prompt

TOOLS: list[str] = []


async def _ask_agent(model: str, system_prompt: str, prompt: str) -> ResultMessage:
    """
    Do not allow the agent to use websearch to avoid leaking the answer; the forecast should depend only on the prompt and the model's own knowledge.
    """
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
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
def predict_with_agent(model: str, experiment: str, **inputs: Any) -> dict[str, Any]:
    # the experiment's key rather than the Experiment itself, so the trace's
    # request records just the name
    settings = EXPERIMENTS[experiment]
    prompt = build_forecast_prompt(inputs, settings.question)
    agent_run = _ask_agent(model, settings.system_prompt, prompt)
    # evaluate() calls predict_fn from worker threads, each of which gets its
    # own event loop here
    result = asyncio.run(agent_run)
    raw = result.result or ""
    if result.is_error:
        raise RuntimeError(f"agent run failed ({result.subtype}): {result.errors}")
    forecast = parse_forecast(raw)
    return {
        "probability": forecast.probability,
        "confidence": forecast.confidence,
        "resolution": forecast.resolution,
        "raw": raw,
        "model": model,
        "num_turns": result.num_turns,
        "cost_usd": result.total_cost_usd,
    }
