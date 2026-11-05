"""
The Snowflake side of the benchmark: the STANDARDIZED_MARKETS query that
freezes every (exchange, event_ticker, outcome) at its first observed price,
run through Snowpark and returned as a pandas frame. `resolved_only` keeps
just the markets that have since resolved.
"""

import pandas as pd
from snowflake.snowpark.session import Session

from benchmark.config import SNOWFLAKE_CONNECTION

# PRICE_TIME is TIMESTAMP_NTZ (UTC) while MARKET_START, MARKET_CLOSE and
# CLOSED_TIME are TIMESTAMP_TZ, so those are normalised to UTC to match.
#
# RESOLUTION is the label of the outcome the market settled on ('yes'/'no' on
# Kalshi, 'Yes'/'No' or a candidate name on Polymarket), NULL or '' while
# unresolved, and CLOSED_TIME is when the market settled. Both are market
# attributes but can differ between a market's price rows (Polymarket rows
# predating the resolution lack it), so they are taken over the whole
# market, as is MARKET_START. The record's outcome occurred when the
# resolution matches OUTCOME (case-insensitively); unresolved markets get
# NULL and are skipped by the outcome scorers.
_QUERY_TEMPLATE = r"""
with markets as (
    select
        std_event_type,
        market as exchange,
        event_ticker,
        event_category,
        event_title,
        market_title,
        market_description,
        min(convert_timezone('UTC', market_start)::timestamp_ntz) over (
            partition by market, event_ticker, outcome
        ) as market_start,
        convert_timezone('UTC', market_close)::timestamp_ntz as market_close,
        outcome,
        price_time as as_of,
        price as market_probability,
        max(nullif(trim(resolution), '')) over (
            partition by market, event_ticker, outcome
        ) as resolution,
        max(convert_timezone('UTC', closed_time)::timestamp_ntz) over (
            partition by market, event_ticker, outcome
        ) as resolved_at,
        hash(market, event_ticker, outcome) as record_id
    from standardized_markets
    where event_category in (
            'politics', 'companies', 'finance', 'elections', 'economics'
        )
        and coalesce(std_event_type, '') <> 'share price' // exclude share price markets, not very interesting
        and price is not null
        and outcome is not null
        and price_time is not null
),
first_price as (
    select
        *,
        iff(
            resolution is null,
            null,
            iff(lower(resolution) = lower(trim(outcome)), 1, 0)
        ) as outcome_occurred
    from markets
    qualify row_number() over (
        partition by exchange, event_ticker, outcome
        order by as_of
    ) = 1
)
select *
from first_price
where {where}
order by exchange, event_ticker, outcome
"""


def fetch_markets(resolved_only: bool) -> pd.DataFrame:
    query = _QUERY_TEMPLATE.format(
        where="resolution is not null" if resolved_only else "true"
    )
    # credentials come from ~/.snowflake/connections.toml, not the environment
    with Session.builder.config(
        "connection_name", SNOWFLAKE_CONNECTION
    ).create() as session:
        # the filtering and first-price windowing run on Snowflake compute
        return session.sql(query).to_pandas()
