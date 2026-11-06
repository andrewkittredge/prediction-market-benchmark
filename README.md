# LLM Prediction Market Forecasting

Measure how well LLM's forecast prediction market (Kalshi/Polymarket) prices and outcomes.

Uses the MLFlow library.

## Dataset

https://app.snowflake.com/marketplace/providers/GZT1Z3TUXVI/Prediction%20Market%20Data

## Experiments

Each experiment fixes its dataset, prompt and scorers ([benchmark/experiment.py](benchmark/experiment.py)); pick one with `--experiment`.

- **`price`** — the model estimates the price a market opened at, scored by squared error against that price (and against the outcome, where the market has resolved). Runs on all markets. Analysis: [analysis/brier_score.ipynb](analysis/brier_score.ipynb), Brier scores by category and model.
- **`outcome-confidence`** — the model makes a yes/no call on whether the outcome occurs and states its confidence, scored by accuracy and Brier score. Runs on resolved markets only. Analysis: [analysis/calibration.ipynb](analysis/calibration.ipynb), the expected calibration error of the stated confidence.

## Commands

```bash
uv sync

uv run build-dataset --experiment outcome-confidence

uv run evaluate --experiment outcome-confidence \
  --models claude-opus-5-5 claude-haiku-4-5 \
  --run-name outcome-confidence

# browse datasets, evaluation runs and traces at http://localhost:5000
uv run mlflow server
```
