"""
The benchmark's experiments: each fixes the evaluation dataset it runs on,
the prompt shown to the model and the scorers applied to the answers. The
keys of EXPERIMENTS are what `--experiment` accepts in build-dataset and
evaluate.
"""

from dataclasses import dataclass

from mlflow.genai.scorers import Scorer

from benchmark.prompt import (
    OUTCOME_QUESTION,
    OUTCOME_SYSTEM_PROMPT,
    PRICE_QUESTION,
    PRICE_SYSTEM_PROMPT,
)
from benchmark.scorers import outcome_correct, squared_error, squared_error_vs_outcome


@dataclass(frozen=True)
class Experiment:
    # MLflow evaluation dataset name
    dataset_name: str
    # build the dataset from only the markets that have resolved
    resolved_only: bool
    system_prompt: str
    # closes the user prompt built by prompt.build_forecast_prompt
    question: str
    scorers: list[Scorer]


EXPERIMENTS = {
    # estimate the market's opening price, scored against that price and,
    # where the market has since resolved, against what happened
    "price": Experiment(
        dataset_name="prediction-markets",
        resolved_only=False,
        system_prompt=PRICE_SYSTEM_PROMPT,
        question=PRICE_QUESTION,
        scorers=[squared_error, squared_error_vs_outcome],
    ),
    # a yes/no call on whether the outcome occurs, on resolved markets only
    "outcome-confidence": Experiment(
        dataset_name="prediction-markets-resolved",
        resolved_only=True,
        system_prompt=OUTCOME_SYSTEM_PROMPT,
        question=OUTCOME_QUESTION,
        scorers=[outcome_correct, squared_error_vs_outcome],
    ),
}
