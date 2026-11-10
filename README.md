# LLM Prediction Market Forecasting

Benchmark LLM model forecasting of prediction markets (Polymarket/Kalshi) outcomes.

After Lukas Nel's [Do Large Language Models Know What They Don’t
Know?
Evaluating Epistemic Calibration via Prediction
Markets](https://arxiv.org/pdf/2512.16030)

Uses MLFlow to prompt models to predict the outcome of prediction markets. 

## Dataset

https://app.snowflake.com/marketplace/providers/GZT1Z3TUXVI/Prediction%20Market%20Data

## Experiments

Each experiment fixes its dataset, prompt and scorers ([benchmark/experiment.py](benchmark/experiment.py)); pick one with `--experiment`.

- **`outcome-confidence`** — the model names the outcome the market resolves to (its `resolution` label: yes/no, up/down, a candidate…) and states its confidence, scored by accuracy and Brier score.  Analysis: [analysis/calibration.ipynb](analysis/calibration.ipynb).

## Commands

```bash
uv sync

# Create the MLflow dataset.
uv run build-dataset --experiment outcome-confidence

# one --model per model; a date after it evaluates that model only on markets
# that opened after the date (here each model's training data cutoff)
uv run evaluate --experiment outcome-confidence \
  --model claude-opus-5-5 2026-06-30 \
  --model claude-haiku-4-5 2025-07-31 \
  --model claude-fable-5-1 2026-06-30

# browse datasets, evaluation runs and traces  
uv run mlflow server
```
