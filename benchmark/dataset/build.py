"""
Builds the LLM forecasting benchmark from STANDARDIZED_MARKETS.

One row per (exchange, event_ticker, outcome) in the politics / companies /
finance / elections / economics categories, excluding share-price markets,
frozen at its first observed price: the model is asked for P(outcome) as of
that moment and scored against the market's own price and, where the market
has since resolved, against whether the outcome occurred. `--resolved` keeps
only the markets that have resolved, each with its open, settlement time and
resolution, so every row can be scored against the outcome.

    uv run build-dataset
    uv run build-dataset --resolved
    uv run build-dataset --dataset-name smoke

Registers (or merges into) an MLflow evaluation dataset of the given name
(default: config.DATASET_NAME, or config.RESOLVED_DATASET_NAME with
--resolved). MLflow evaluation datasets need a SQL tracking store, hence the
sqlite default.
"""

import argparse
import logging

import mlflow
from mlflow.genai.datasets import create_dataset, get_dataset

from benchmark.config import DATASET_NAME, RESOLVED_DATASET_NAME
from benchmark.dataset.records import mlflow_records_from_pandas
from benchmark.dataset.snowflake_markets import fetch_markets

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--resolved",
        action="store_true",
        help="only markets that have resolved (default dataset name: "
        f"{RESOLVED_DATASET_NAME})",
    )
    parser.add_argument(
        "--dataset-name",
        help=f"MLflow evaluation dataset name (default: {DATASET_NAME})",
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name or (
        RESOLVED_DATASET_NAME if args.resolved else DATASET_NAME
    )

    markets = fetch_markets(resolved_only=args.resolved)
    records = mlflow_records_from_pandas(markets)
    logger.info("%d records for dataset %r", len(records), dataset_name)

    try:
        dataset = get_dataset(name=dataset_name)
    except mlflow.MlflowException:
        dataset = create_dataset(name=dataset_name)
    dataset.merge_records(records)


if __name__ == "__main__":
    main()
