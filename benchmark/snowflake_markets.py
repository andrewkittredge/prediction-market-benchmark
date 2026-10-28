"""
The Snowflake side of the benchmark: the STANDARDIZED_MARKETS query that
freezes every (exchange, event_ticker, outcome) at its first observed price,
run through Snowpark and returned as a pandas frame.
"""

import pandas as pd
from snowflake.snowpark.session import Session

from benchmark.config import SNOWFLAKE_CONNECTION

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


def fetch_markets() -> pd.DataFrame:
    # credentials come from ~/.snowflake/connections.toml, not the environment
    with Session.builder.config(
        "connection_name", SNOWFLAKE_CONNECTION
    ).create() as session:
        # the filtering and first-price windowing run on Snowflake compute
        return session.sql(_QUERY_TEMPLATE).to_pandas()
