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
Kalshi, 11.3k Polymarket); `--exchange` narrows it to one exchange.

## Commands

Run from this folder (its own uv project; `benchmark/` is a plain package
imported from the project root). It depends on the `prediction-markets`
library from the sibling checkout `../prediction-markets`. Snowflake and
Anthropic credentials come from a gitignored `.env` here, the same file as the
one at that repo's root; pass it with `uv run --env-file .env ...`.

```bash
uv sync

# build: registers an MLflow evaluation dataset (re-running merges into it)
uv run python -m benchmark.build_dataset --name smoke --exchange kalshi
uv run python -m benchmark.build_dataset --name forecasting-v2

# evaluate models (needs ANTHROPIC_API_KEY) on a fixed random sample; --max-rows defaults to 100.
# One parent MLflow run with a nested child run per (model, system prompt)
# pair: every --model crossed with every --prompt (default: all the prompts in
# prompt.py, i.e. with and without the "use only information available on this
# date" instruction), same rows each. --model must be in model_cutoffs.py's
# TRAINING_DATA_CUTOFF table. Each reply is a JSON object with the model's
# "probability" (its estimate of the opening price) and its "confidence" (see
# Scorers below).
uv run python -m benchmark.evaluate --dataset-name smoke --max-rows 10
uv run python -m benchmark.evaluate --dataset-name smoke --max-rows 10 --prompt cutoff
uv run python -m benchmark.evaluate --dataset-name forecasting-v2 --model claude-opus-5-5 claude-haiku-4-5

# forecast with a local Claude Code agent instead of a bare API call. The
# claude-agent-sdk runs the installed `claude` CLI, so with ANTHROPIC_API_KEY
# unset (no --env-file) it uses the CLI's own login, i.e. your Claude
# subscription. The agent may use WebSearch/WebFetch before answering; see
# the leakage caveat below. Each child run logs backend=agent.
uv run python -m benchmark.evaluate --dataset-name smoke --max-rows 10 --model claude-sonnet-5-5 --agent

# browse datasets, evaluation runs and traces at http://localhost:5000
uv run mlflow server --port 5000 --backend-store-uri "$env:MLFLOW_TRACKING_URI"
```

MLflow evaluation datasets need a SQL tracking store (the file store doesn't
support them). Both scripts and the notebook use whatever `MLFLOW_TRACKING_URI`
points at — the SQLite file `%LOCALAPPDATA%\mlflow\mlflow.db` on the
development machine. The server needs the same location passed explicitly
(`--backend-store-uri` or `MLFLOW_BACKEND_STORE_URI`); it does not read
`MLFLOW_TRACKING_URI`. The database lives outside the repo, so a fresh clone
or machine rebuilds the dataset from Snowflake.

## Scorers

The run's MLflow scorer is `squared_error` = (p − market_probability)²: the
model's disagreement with the market's opening price, squared so large misses
weigh more. It is averaged over the dataset and skipped (None) when no
probability could be parsed from the reply. Lower is better. `scorers.py` also
has `abs_error_vs_market` = |p − market_probability|, which the notebook
computes as `error`.

### Confidence

Besides its price estimate the model states a **confidence**: its own
probability that the estimate lands within `CONFIDENCE_TOLERANCE` (0.10, set
in `prompt.py`) of the actual opening price. `parse_probability.parse_forecast`
reads both numbers from the reply's JSON object, and the predict functions
return `confidence` next to `probability`.

The confidence is scored in `analysis/brier_score.ipynb`, not by an MLflow scorer:

- `within_tolerance` = |p − market_probability| ≤ `CONFIDENCE_TOLERANCE`: the
  event the model was asked to rate.
- `confidence_brier` = (confidence − within_tolerance)²: the Brier score of
  the stated confidence against that event. 0.25 is what a constant 0.5 would
  score; lower is better.
- A calibration curve per model: hit rate (`within_tolerance` mean) per
  confidence decile, plotted against the mean stated confidence in that
  decile, with the diagonal as perfect calibration.

## Caveat: leakage

Models trained after a market closed may simply know how it resolved, and the
prompt's "use only information available on this date" instruction is not
enforceable. For a fair test of a given model, restrict the dataset to markets
that closed after its training cutoff (a `market_close` filter in the query).
Each child run carries a `training_data_cutoff` tag (from the hand-maintained
`TRAINING_DATA_CUTOFF` table in `evaluate.py`) so the comparison is at hand.

With `--agent` the training cutoff is not the only leak: the agent can search
the live web, which may surface how a market resolved (or other markets'
prices for it) regardless of what the model was trained on. Read the
`WebSearch` calls in the trace before trusting an agent run's score.
