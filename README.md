# LLM forecasting benchmark

Measure how well LLM's forecast prediction market (Kalshi/Polymarket) prices and outcomes.

Uses the MLFlow library.

## Dataset

https://app.snowflake.com/marketplace/providers/GZT1Z3TUXVI/Prediction%20Market%20Data

## Commands

```bash
uv sync

uv run build-dataset --resolved

uv run evaluate --dataset-name prediction-markets-resolved --models claude-opus-5-5 claude-haiku-4-5 --prompt outcome-cutoff --run-name outcome-confidence

# browse datasets, evaluation runs and traces at http://localhost:5000
uv run mlflow server
```

## Analysis

[analysis/calibration.ipynb](analysis/calibration.ipynb) computes the expected calibration error of the model's stated confidence.

