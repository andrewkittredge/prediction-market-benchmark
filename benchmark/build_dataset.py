"""
Builds the LLM forecasting benchmark from STANDARDIZED_MARKETS.

One row per (exchange, event_ticker, outcome) in the politics / companies /
finance / elections / economics categories, excluding share-price markets,
frozen at its first observed price: the model is asked for P(outcome) as of
that moment and scored against the market's own price and, where the market
has since resolved, against whether the outcome occurred.

    # from the project root (`benchmark` is not installed, so `-m` needs the
    # root on sys.path)
    uv run --env-file .env python -m benchmark.build_dataset --name smoke
    uv run python -m benchmark.build_dataset --name forecasting-v2

Registers (or merges into) an MLflow evaluation dataset of the given name.
MLflow evaluation datasets need a SQL tracking store, hence the sqlite default.
"""

import argparse
import logging

from mlflow.genai.datasets import get_dataset
from snowflake.snowpark.session import Session

from benchmark.records import mlflow_records_from_pandas

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


EXPERIMENT_NAME = "forecasting-v3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="forecasting-v2")

    args = parser.parse_args()

    with Session.builder.create() as session:
        snowpark_df = session.sql(_QUERY_TEMPLATE)
        local_pandas_df = snowpark_df.to_pandas()
    records = mlflow_records_from_pandas(local_pandas_df)

    dataset = get_dataset(name=args.name)
    dataset.merge_records(records)


if __name__ == "__main__":
    main()
