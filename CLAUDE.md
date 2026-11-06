# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

An LLM forecasting benchmark: asks Claude models for a probability on real
prediction-market questions and scores the answers against the market's own
price (and, once a market resolves, the outcome). `README.md` explains the
dataset shape, the scorers and the leakage caveat; read it first. Everything
here is one small uv project:

- `benchmark/` — the code, installed editable into `.venv` by `uv sync`
  (hatchling builds just this package) so `pyproject.toml` can declare
  console scripts (`build-dataset`, `evaluate`).
  - `experiment.py` — `EXPERIMENTS`: the two experiments, `price` (estimate
    the market's opening price) and `outcome-confidence` (a yes/no call on whether the
    outcome occurs), each an `Experiment` fixing its evaluation dataset, its
    prompt and its scorers. The keys are what `--experiment` accepts in both
    scripts.
  - `dataset/` — building the evaluation datasets: `build.py` runs
    `snowflake_markets.fetch_markets()` (the Snowpark query, as a pandas
    frame) → MLflow evaluation records (`records.py`) → the experiment's
    MLflow evaluation dataset (re-running merges into the existing one). The
    `outcome-confidence` experiment's dataset holds only markets that have resolved,
    with `market_start`, a `resolved_at` tag (the view's `CLOSED_TIME`) and
    the `resolution` label.
  - `evaluate.py` — `mlflow.genai.evaluate` sweep of one experiment: one
    parent run per invocation, one child run per `--models` entry, all on
    the same fixed-seed sample.
  - `predict_agent.py` — the `predict_fn`: a local Claude Code agent via
    `claude-agent-sdk`, no tools. Model and experiment key are bound with
    `functools.partial`.
  - `prompt.py` — the prompt texts (per experiment, a system prompt and the
    question closing the user prompt, each asking for a 0-1 confidence) and
    the user-prompt builder. `parse_probability.py` pulls the answer out
    of the reply. `scorers.py` holds the MLflow scorers.
  - `config.py` — `EXPERIMENT_NAME` (the MLflow experiment, `evaluate`'s
    default) and `SNOWFLAKE_CONNECTION`.
- `analysis/` — notebooks and their helpers. `run_traces.py` pulls a
  finished sweep back out of MLflow (the parent's child runs, their traces,
  and `load_trace_df`, one row per forecast); `brier_score.ipynb` loads a
  sweep with it into `forecasts` and computes Brier scores by category and
  model; `calibration.ipynb` does the same load and computes the expected
  calibration error of the model's stated confidence.

The `STANDARDIZED_MARKETS` view it queries is built by the sibling repo
`../prediction-markets`, whose `CLAUDE.md` documents the data.

## Commands

Run from this folder. The agent runs on the Claude CLI's own login unless
`ANTHROPIC_API_KEY` is set (the gitignored `.env` here, a copy of the one at
the sibling repo's root, passed with `--env-file .env`). Snowflake credentials come
from the `[prediction-markets]` section of `~/.snowflake/connections.toml`
(`config.SNOWFLAKE_CONNECTION`), not from the environment.

```bash
uv sync
uv run --env-file .env build-dataset --experiment outcome-confidence
uv run --env-file .env evaluate --experiment outcome-confidence --max-rows 2
uv run --env-file .env evaluate --experiment price --models claude-haiku-4-5
uv run mlflow server --port 5000 --backend-store-uri "$MLFLOW_TRACKING_URI"
```

**Don't launch real evaluation runs unprompted.** Every row is a paid API
call (there is no stub model). To check the harness, smoke-test with
`--max-rows 2` and hand the full command to the user.
`build-dataset` hits Snowflake and takes a while; leave it to the user too.

## MLflow

- The tracking store is a SQLite file outside the repo,
  `%LOCALAPPDATA%\mlflow\mlflow.db` (evaluation datasets need a SQL store).
  Scripts and the notebook read it from the `MLFLOW_TRACKING_URI` environment
  variable and never call `set_tracking_uri`; if the variable is missing,
  MLflow silently falls back to `./mlflow.db` in the current directory and
  you get "dataset not found". Check with
  `uv run python -c "import mlflow; print(mlflow.get_tracking_uri())"`.
- `mlflow server` does **not** read `MLFLOW_TRACKING_URI`; it needs
  `--backend-store-uri` (or `MLFLOW_BACKEND_STORE_URI`).
- Runs write artifacts under `./mlruns` relative to the cwd (gitignored).
- `mlflow.parentRunId` tags hold the parent's run **id**, not its name;
  look the parent up by name first (the notebook does this).
- Dataset expectations become `Expectation` assessments on each trace and
  MLflow rejects `None` values, so `dataset.records._mlflow_record` omits
  unresolved markets' `outcome_occurred` instead of emitting `None`.
- `sqlalchemy<2.1` is pinned: mlflow 3.17's SQL store imports a symbol
  SQLAlchemy 2.1 removed.

## Conventions

Implement the minimum that satisfies the request (YAGNI): no extra flags,
fallbacks, configurability or "while I'm here" extras; remove features that
turn out unused.

Avoid closures. A callable that needs configuration is a module-level
function with explicit parameters bound via `functools.partial`, or a small
`@dataclass` with `__call__`.

Prefer `@dataclass` over `TypedDict`/dicts for structured data.

One expression per line: don't nest a call inside another call's arguments.
Assign the inner result to a named variable on its own line first, so
`markets = fetch_markets(...)` then `records = mlflow_records_from_pandas(markets)`,
not `mlflow_records_from_pandas(fetch_markets(...))`.

In the notebooks (`analysis/`), one statement per cell: each assignment, function
definition or expression gets its own cell so intermediates can be inspected
and re-run alone. The notebooks' kernel is this project's `.venv`; their cwd
is the project root (`jupyter.notebookFileRoot` in `.vscode/settings.json`),
so `benchmark.*` and `analysis.*` import from there. Restart the kernel after
changing that setting.
