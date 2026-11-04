# LLM forecasting benchmark

Measures how closely an LLM's probability for a prediction-market outcome agrees
with the market's own probability and, for markets that have since resolved,
how accurate it was. Built from `STANDARDIZED_MARKETS` (the
politics, companies, finance, elections and economics categories, minus
share-price markets), packaged as an
[MLflow evaluation dataset](https://mlflow.org/docs/latest/genai/datasets/),
and run with `mlflow.genai.evaluate`.

## Dataset

One row per `(exchange, event_ticker, outcome)`, frozen at its **first observed
price**.

- `inputs` — what the model sees: exchange, event ticker, event/market title,
  resolution rules (`market_description`), market close, the outcome being
  priced, and the as-of date (`as_of` = that first price time).
- `expectations` — `market_probability`, the price at `as_of`, and
  `outcome_occurred`, 1/0 once the market has resolved (its `RESOLUTION` label
  matches the outcome) and null while it hasn't.
- `tags` — `record_id` (= `HASH(exchange, event_ticker, outcome)`),
  `exchange`, `event_category` and `std_event_type`; tags are copied onto each
  evaluation trace, so traces can be filtered by exchange, category or event
  type in the UI.

The query takes every qualifying market (about 11.9k as of October 2026: 658
Kalshi, 11.3k Polymarket).

## Commands

Run from this folder (its own uv project; `uv sync` installs `benchmark/`
editable, which provides the `build-dataset` and `evaluate` scripts). It depends on the `prediction-markets`
library from the sibling checkout `../prediction-markets`. Snowflake and
Anthropic credentials come from a gitignored `.env` here, the same file as the
one at that repo's root; pass it with `uv run --env-file .env ...`.

```bash
uv sync

# build: registers an MLflow evaluation dataset (re-running merges into it)
uv run build-dataset

# forecast with a local Claude Code agent instead of a bare API call. The
# claude-agent-sdk runs the installed `claude` CLI, so with ANTHROPIC_API_KEY
# unset (no --env-file) it uses the CLI's own login, i.e. your Claude
# subscription. The agent may use WebSearch/WebFetch before answering; see
# the leakage caveat below. Each child run logs backend=agent.
uv run evaluate --dataset-name smoke --max-rows 10 --model claude-sonnet-5-5 --agent

# browse datasets, evaluation runs and traces at http://localhost:5000
uv run mlflow server --port 5000 --backend-store-uri "$env:MLFLOW_TRACKING_URI"
