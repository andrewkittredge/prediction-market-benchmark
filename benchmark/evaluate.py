"""
Runs an LLM against the forecasting benchmark with mlflow.genai.evaluate.

    cd prediction-market-benchmark
    uv run evaluate --experiment outcome-confidence --max-rows 10
    uv run evaluate --experiment price --models claude-opus-5-5 claude-haiku-4-5

--experiment (benchmark.experiment.EXPERIMENTS) fixes the dataset, the prompt
and the scorers. Forecasts come from a local Claude Code agent
(claude-agent-sdk); see predict_agent.py for what it is logged in as.

Each invocation is one parent MLflow run with a nested child run per
--models entry, all over the same sample of rows.

Results land in the MLflow experiment config.EXPERIMENT_NAME at the
tracking URI; browse them with
    uv run mlflow-ui
"""

import argparse
import logging
from datetime import date, datetime
from functools import partial

import mlflow
import mlflow.genai

# Work around an import deadlock in mlflow 3.17: the first scorer call imports
# mlflow.server.jobs.utils on a worker thread (via telemetry), which fires a
# post-import hook that loads most of mlflow.genai, while predict threads are
# lazily importing the token-cost modules; the threads then wait on each
# other's import locks forever. Importing it once here, single-threaded,
# before evaluate() starts its thread pools avoids the race.
import mlflow.server.jobs.utils  # noqa: E402,F401
import pandas as pd
from mlflow.genai.datasets import get_dataset

from benchmark.config import EXPERIMENT_NAME
from benchmark.experiment import EXPERIMENTS
from benchmark.predict_agent import predict_with_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
# the SDK logs which `claude` binary it runs at INFO on every call
logging.getLogger("claude_agent_sdk").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def after_cutoff(data: pd.DataFrame, cutoff: date) -> pd.Series:
    """Mask of rows whose inputs.as_of falls after `cutoff` (exclusive)."""
    as_of = pd.to_datetime(data["inputs"].map(lambda inputs: inputs["as_of"]))
    return as_of > pd.Timestamp(cutoff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])  # pyright: ignore[reportOptionalMemberAccess]
    parser.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENTS,
        help="the experiment (benchmark.experiment.EXPERIMENTS) to run: its "
        "dataset, prompt and scorers",
    )
    parser.add_argument(
        "--experiment-name", default=EXPERIMENT_NAME, help="MLflow experiment name"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Anthropic model id(s) to sweep; default %(default)s",
        default=["claude-opus-4-6"],
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
    parser.add_argument("--run-name")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    experiment = EXPERIMENTS[args.experiment]
    dataset = get_dataset(name=experiment.dataset_name)
    data = dataset.to_df()
    if cutoff := args.cutoff:
        data = data[after_cutoff(data, cutoff)]
    if max_rows := args.max_rows:
        data = data.sample(n=min(max_rows, len(data)), random_state=0)
    run_name = args.run_name or f"sweep-{datetime.now():%Y%m%d-%H%M}"
    # a parent run per invocation holding the shared settings, with one child
    # run per model, all on the same sample, so models can be compared
    # trace-for-trace
    with mlflow.start_run(run_name=run_name):
        for model_name in args.models:
            predict_fn = partial(predict_with_agent, model_name, args.experiment)
            with mlflow.start_run(
                run_name=f"{model_name}-{args.experiment}", nested=True
            ):
                mlflow.set_active_model(name=model_name)
                mlflow.log_params({"experiment": args.experiment})
                mlflow.genai.evaluate(
                    data=data,
                    predict_fn=predict_fn,
                    scorers=experiment.scorers,
                )


if __name__ == "__main__":
    main()
