"""
Runs an LLM against the forecasting benchmark with mlflow.genai.evaluate.

    cd prediction-market-benchmark
    set ANTHROPIC_API_KEY=...
    uv run python -m benchmark.evaluate --dataset-name smoke --max-rows 10
    uv run python -m benchmark.evaluate --dataset-name forecasting-v2 --model claude-opus-5-5 claude-haiku-4-5

    # forecast with a local Claude Code agent (claude-agent-sdk) instead of
    # the Messages API; see predict_agent.py
    uv run python -m benchmark.evaluate --dataset-name smoke --max-rows 10 --agent

Each invocation is one parent MLflow run with a nested child run per
(model, system prompt) pair — every --model crossed with every --prompt
(default: all of benchmark.prompt.SYSTEM_PROMPTS) — all over the same sample
of rows.

Results land in the MLflow experiment config.EXPERIMENT_NAME at the
tracking URI; browse them with
    uv run mlflow-ui
"""

import argparse
import itertools
import logging
from datetime import date, datetime
from functools import partial

import mlflow

# Work around an import deadlock in mlflow 3.17: the first scorer call imports
# mlflow.server.jobs.utils on a worker thread (via telemetry), which fires a
# post-import hook that loads most of mlflow.genai, while predict threads are
# lazily importing the token-cost modules; the threads then wait on each
# other's import locks forever. Importing it once here, single-threaded,
# before evaluate() starts its thread pools avoids the race.
import mlflow.server.jobs.utils  # noqa: E402,F401
import pandas as pd
from mlflow.genai.datasets import get_dataset

from benchmark.config import DATASET_NAME, EXPERIMENT_NAME
from benchmark.model_cutoffs import TRAINING_DATA_CUTOFF
from benchmark.predict_agent import predict_with_agent

# from benchmark.predict_api import MAX_TOKENS, predict_api
from benchmark.prompt import SYSTEM_PROMPTS
from benchmark.scorers import squared_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def after_cutoff(data: pd.DataFrame, cutoff: date) -> pd.Series:
    """Mask of rows whose inputs.as_of falls after `cutoff` (exclusive)."""
    as_of = pd.to_datetime(data["inputs"].map(lambda inputs: inputs["as_of"]))
    return as_of > pd.Timestamp(cutoff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset-name", default=DATASET_NAME, help="MLflow evaluation dataset name"
    )
    parser.add_argument(
        "--experiment-name", default=EXPERIMENT_NAME, help="MLflow experiment name"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=TRAINING_DATA_CUTOFF,
        metavar="MODEL",
        help="Anthropic model id(s) to sweep; default %(default)s",
        default=["claude-opus-4-6"],
    )
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=list(SYSTEM_PROMPTS),
        choices=SYSTEM_PROMPTS,
        metavar="PROMPT",
        help="system prompt(s) from benchmark.prompt.SYSTEM_PROMPTS to sweep; "
        "default all (%(default)s)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of rows to evaluate",
    )
    parser.add_argument(
        "--cutoff",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="only evaluate rows whose as_of is after this date",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="forecast with a local Claude Code agent (no tools) "
        "instead of a single Messages API call",
    )

    parser.add_argument("--run-name")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    data = get_dataset(name=args.dataset_name).to_df()
    if args.cutoff is not None:
        data = data[after_cutoff(data, args.cutoff)]
    if args.max_rows is not None:
        data = data.sample(n=min(args.max_rows, len(data)), random_state=0)
    run_name = args.run_name or f"sweep-{datetime.now():%Y%m%d-%H%M}"
    # a parent run per invocation holding the shared settings, with one child
    # run per (model, system prompt) pair, all on the same sample, so models
    # and prompts can be compared trace-for-trace
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "dataset": args.dataset_name,
                "max_rows": args.max_rows or "all",
                "cutoff": args.cutoff or "none",
                "backend": "agent" if args.agent else "api",
            }
        )
        for model_name, system_prompt in itertools.product(args.models, args.prompt):
            if args.agent:
                predict_fn = partial(predict_with_agent, model_name, system_prompt)
            else:
                pass
                # predict_fn = partial(predict_api, model_name, MAX_TOKENS, system_prompt)
            with mlflow.start_run(
                run_name=f"{model_name}-{system_prompt}", nested=True
            ):
                mlflow.set_active_model(name=model_name)
                mlflow.set_tag("training_data_cutoff", TRAINING_DATA_CUTOFF[model_name])
                mlflow.log_params({"system_prompt": system_prompt})
                mlflow.genai.evaluate(
                    data=data,
                    predict_fn=predict_fn,
                    scorers=[squared_error],
                )


if __name__ == "__main__":
    main()
