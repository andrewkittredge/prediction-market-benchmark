# LLM Prediction Market Forecasting

Measure how well LLM's forecast prediction market (Kalshi/Polymarket) prices and outcomes.

Uses the MLFlow library.

## Dataset

https://app.snowflake.com/marketplace/providers/GZT1Z3TUXVI/Prediction%20Market%20Data

## Commands

```bash
uv sync

# --experiment is price (estimate the market's opening price) or
# outcome-confidence (a yes/no call on resolved markets); it fixes the
# dataset, prompt and scorers
uv run build-dataset --experiment outcome-confidence

uv run evaluate --experiment outcome-confidence \
  --models claude-opus-5-5 claude-haiku-4-5 \
  --run-name outcome-confidence

# browse datasets, evaluation runs and traces at http://localhost:5000
uv run mlflow server
```

## Analysis

[analysis/calibration.ipynb](analysis/calibration.ipynb) computes the expected calibration error of the model's stated confidence.

