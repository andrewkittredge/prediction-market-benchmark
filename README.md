# LLM Prediction Market Forecasting

Measure how well LLM's forecast prediction market (Kalshi/Polymarket) prices and outcomes.

Uses the MLFlow library.

## Dataset

https://app.snowflake.com/marketplace/providers/GZT1Z3TUXVI/Prediction%20Market%20Data

## Experiments

Each experiment fixes its dataset, prompt and scorers ([benchmark/experiment.py](benchmark/experiment.py)); pick one with `--experiment`.

- **`price`** — the model estimates the price a market opened at, scored by squared error against that price (and against the outcome, where the market has resolved). Runs on all markets. Analysis: [analysis/brier_score.ipynb](analysis/brier_score.ipynb), Brier scores by category and model.
- **`outcome-confidence`** — the model names the outcome the market resolves to (its `resolution` label: yes/no, up/down, a candidate…) and states its confidence, scored by accuracy and Brier score. Runs on resolved markets only. Analysis: [analysis/calibration.ipynb](analysis/calibration.ipynb), the expected calibration error of the stated confidence.

## Commands

```bash
uv sync

uv run build-dataset --experiment outcome-confidence

# one --model per model; a date after it evaluates that model only on markets
# that opened after the date (here each model's training data cutoff)
uv run evaluate --experiment outcome-confidence \
  --model claude-opus-5-5 2026-06-30 \
  --model claude-haiku-4-5 2025-07-31

# browse datasets, evaluation runs and traces at http://localhost:5000; the
# server doesn't read MLFLOW_TRACKING_URI, so name the store explicitly
uv run mlflow server --port 5000 --backend-store-uri "$MLFLOW_TRACKING_URI"
```
