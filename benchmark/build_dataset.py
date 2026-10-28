"""
Builds the LLM forecasting benchmark from STANDARDIZED_MARKETS.

One row per (exchange, event_ticker, outcome) in the politics / companies /
finance / elections / economics categories, excluding share-price markets,
frozen at its first observed price: the model is asked for P(outcome) as of
that moment and scored against the market's own price and, where the market
has since resolved, against whether the outcome occurred.

    cd scripts/benchmark
    uv run python -m benchmark.build_dataset --name smoke --exchange kalshi
    uv run python -m benchmark.build_dataset --name forecasting-v2

Registers (or merges into) an MLflow evaluation dataset of the given name.
MLflow evaluation datasets need a SQL tracking store, hence the sqlite default.
"""

import argparse
import logging
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import text
from sqlmodel import Session

from benchmark.config import EXPERIMENT_NAME
from benchmark.records import BenchmarkRecord
from prediction_markets.database.engine import create_db_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# PRICE_TIME is TIMESTAMP_NTZ (UTC) while MARKET_CLOSE is TIMESTAMP_TZ, so the
# close is normalised to UTC to match.
#
# RESOLUTION is the label of the outcome the market settled on ('yes'/'no' on
# Kalshi, 'Yes'/'No' or a candidate name on Polymarket) — a market attribute
# repeated on every price row, NULL or '' while unresolved. The record's
# outcome occurred when it matches OUTCOME (case-insensitively); unresolved
# markets get NULL and are skipped by the Brier scorer.
_QUERY_TEMPLATE = r"""
with first_price as (
    select
        market as exchange,
        event_ticker,
        event_category,
        event_title,
        market_title,
        market_description,
        convert_timezone('UTC', market_close)::timestamp_ntz as market_close,
        outcome,
        price_time as as_of,
        price as market_probability,
        iff(
            nullif(trim(resolution), '') is null,
            null,
            iff(lower(trim(resolution)) = lower(trim(outcome)), 1, 0)
        ) as outcome_occurred,
        hash(market, event_ticker, outcome) as record_id
    from standardized_markets
    where event_category in (
            'politics', 'companies', 'finance', 'elections', 'economics'
        )
        and std_event_type not in ('share price')
        and price is not null
        and outcome is not null
        and price_time is not null
    qualify row_number() over (
        partition by market, event_ticker, outcome
        order by price_time
    ) = 1
)
select *
from first_price
order by exchange, event_ticker, outcome
"""


def fetch_records(
    session: Session,
) -> list[BenchmarkRecord]:
    rows = session.execute(text(_QUERY_TEMPLATE)).all()
    return [BenchmarkRecord.from_row(row) for row in rows]


def register_with_mlflow(records: list[BenchmarkRecord], name: str, tags: dict) -> str:
    import mlflow
    from mlflow.genai.datasets import create_dataset, get_dataset, set_dataset_tags

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("mlflow tracking uri %s", mlflow.get_tracking_uri())
    try:
        dataset = get_dataset(name=name)
        logger.info("mlflow dataset %r exists; merging records into it", name)
    except Exception:
        dataset = create_dataset(
            name=name, experiment_id=experiment.experiment_id, tags=tags
        )
        logger.info("created mlflow dataset %r (%s)", name, dataset.dataset_id)
    else:
        set_dataset_tags(dataset_id=dataset.dataset_id, tags=tags)
    dataset.merge_records([r.to_mlflow_record() for r in records])
    return dataset.dataset_id


def summarize(records: list[BenchmarkRecord]) -> str:
    counts = Counter(r.exchange for r in records)
    resolved = Counter(r.exchange for r in records if r.outcome_occurred is not None)
    lines = [f"{len(records)} records"]
    for exchange, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n:6d}  {exchange}  ({resolved[exchange]} resolved)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="forecasting-v2")

    args = parser.parse_args()

    with Session(create_db_engine()) as session:
        records = fetch_records(session)
    if not records:
        raise SystemExit("query returned no markets")
    logger.info("%s", summarize(records))

    tags = {
        "categories": "politics,companies,finance,elections,economics",
        "snapshot": "first_observed_price",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_records": str(len(records)),
    }
    dataset_id = register_with_mlflow(records, args.name, tags)
    logger.info("mlflow dataset %r id=%s", args.name, dataset_id)


if __name__ == "__main__":
    main()
