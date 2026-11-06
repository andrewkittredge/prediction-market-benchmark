"""
Builds the LLM forecasting benchmark from STANDARDIZED_MARKETS.

One row per (exchange, event_ticker, outcome) in the politics / companies /
finance / elections / economics categories, excluding share-price markets,
frozen at its first observed price: the model is asked for P(outcome) as of
that moment and scored against the market's own price and, where the market
has since resolved, against whether the outcome occurred. The outcome
experiment's dataset keeps only the markets that have resolved, each with its
open, settlement time and resolution, so every row can be scored against the
outcome.

    uv run build-dataset --experiment price
    uv run build-dataset --experiment outcome-confidence

Registers (or merges into) the experiment's MLflow evaluation dataset
(experiment.EXPERIMENTS). MLflow evaluation datasets need a SQL tracking
store, hence the sqlite default.
"""

import argparse
import logging

import mlflow
from mlflow.genai.datasets import create_dataset, get_dataset

from benchmark.dataset.records import mlflow_records_from_pandas
from benchmark.dataset.snowflake_markets import fetch_markets
from benchmark.experiment import EXPERIMENTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENTS,
        help="the experiment (benchmark.experiment.EXPERIMENTS) whose dataset to build",
    )
    args = parser.parse_args()
    experiment = EXPERIMENTS[args.experiment]
    dataset_name = experiment.dataset_name

    markets = fetch_markets(resolved_only=experiment.resolved_only)
    records = mlflow_records_from_pandas(markets)
    logger.info("%d records for dataset %r", len(records), dataset_name)

    try:
        dataset = get_dataset(name=dataset_name)
    except mlflow.MlflowException:
        dataset = create_dataset(name=dataset_name)
    dataset.merge_records(records)


if __name__ == "__main__":
    main()
