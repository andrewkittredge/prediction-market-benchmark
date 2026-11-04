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
  - `dataset/` — building the evaluation dataset: `build.py` runs
    `snowflake_markets.fetch_markets()` (the Snowpark query, as a pandas
    frame) → MLflow evaluation records (`records.py`) → an MLflow evaluation
    dataset (re-running merges into the existing one).
  - `evaluate.py` — `mlflow.genai.evaluate` sweep: one parent run per
    invocation, one child run per `--model` × `--prompt`, all on the same
    fixed-seed sample. `TRAINING_DATA_CUTOFF` there is hand-maintained.
  - `predict_agent.py` — the `predict_fn`: a local Claude Code agent via
    `claude-agent-sdk`, no tools. Model and prompt are bound with
    `functools.partial`.
  - `prompt.py` — `SYSTEM_PROMPTS` (the keys are what `--prompt` accepts)
    and the user-prompt builder. `parse_probability.py` pulls the number out
    of the reply. `scorers.py` holds the MLflow scorers.
  - `config.py` — `EXPERIMENT_NAME`, `DATASET_NAME` (the scripts' command-line
    defaults) and `SNOWFLAKE_CONNECTION`.
- `analysis/` — notebooks and their helpers. `run_traces.py` pulls a
  finished sweep back out of MLflow (the parent's child runs, their traces,
  and `load_trace_df`, one row per forecast); `brier_score.ipynb` loads a
  sweep with it into `trace_df` and regresses error on months past the
  model's training cutoff.

It depends on the `prediction-markets` library from the sibling checkout
`../prediction-markets` (editable path dependency in `pyproject.toml`) for the
Snowflake engine and models; that repo's `CLAUDE.md` documents the data.

## Commands

Run from this folder. The agent runs on the Claude CLI's own login unless
`ANTHROPIC_API_KEY` is set (the gitignored `.env` here, a copy of the one at
the sibling repo's root, passed with `--env-file .env`). Snowflake credentials come
from the `[prediction-markets]` section of `~/.snowflake/connections.toml`
(`config.SNOWFLAKE_CONNECTION`), not from the environment.

```bash
uv sync
uv run --env-file .env build-dataset --dataset-name smoke
uv run --env-file .env evaluate --dataset-name smoke --max-rows 2
uv run --env-file .env evaluate --dataset-name forecasting-v2 --model claude-haiku-4-5 --prompt no-cutoff
uv run mlflow server --port 5000 --backend-store-uri "$MLFLOW_TRACKING_URI"
```

**Don't launch real evaluation runs unprompted.** Every row is a paid API
call (there is no stub model). To check the harness, smoke-test with
`--max-rows 2` on the `smoke` dataset and hand the full command to the user.
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

In the notebooks (`analysis/`), one statement per cell: each assignment, function
definition or expression gets its own cell so intermediates can be inspected
and re-run alone. The notebooks' kernel is this project's `.venv`; their cwd
is the project root (`jupyter.notebookFileRoot` in `.vscode/settings.json`),
so `benchmark.*` and `analysis.*` import from there. Restart the kernel after
changing that setting.
