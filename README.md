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
  matches the outcome) and null while it hasn't, plus `resolution`, the label
  the market settled on, when it has.
- `tags` — `record_id` (= `HASH(exchange, event_ticker, outcome)`),
  `exchange`, `event_category` and `std_event_type`; tags are copied onto each
  evaluation trace, so traces can be filtered by exchange, category or event
  type in the UI.

The query takes every qualifying market (about 11.9k as of October 2026: 658
Kalshi, 11.3k Polymarket).

## Commands

Run from this folder (its own uv project; `uv sync` installs `benchmark/`
editable, which provides the `build-dataset` and `evaluate` scripts).
Snowflake credentials come from `~/.snowflake/connections.toml`.

```bash
uv sync

# build: registers an MLflow evaluation dataset (re-running merges into it)
uv run build-dataset
# the same markets restricted to those that have resolved, as
# prediction-markets-resolved: every row carries resolution / outcome_occurred
uv run build-dataset --resolved

# --prompt: cutoff / no-cutoff ask for the market's opening price,
# outcome-cutoff / outcome-no-cutoff for a yes/no call on the outcome plus a
# 0-1 confidence in it (scored against outcome_occurred once the market resolves)
uv run evaluate --models claude-opus-5-5 claude-haiku-4-5 --prompt cutoff

# the outcome-confidence experiment: yes/no calls with confidence on the
# resolved markets, so outcome_correct (accuracy) and squared_error_vs_outcome
# (Brier against what happened) are scored on every row
uv run evaluate --dataset-name prediction-markets-resolved --models claude-opus-5-5 claude-haiku-4-5 --prompt outcome-cutoff --run-name outcome-confidence

# browse datasets, evaluation runs and traces at http://localhost:5000
uv run mlflow server --port 5000 --backend-store-uri "$env:MLFLOW_TRACKING_URI"
