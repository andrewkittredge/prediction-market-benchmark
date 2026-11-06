"""
Runs an LLM against the forecasting benchmark with mlflow.genai.evaluate.

    cd prediction-market-benchmark
    uv run evaluate --experiment outcome-confidence --max-rows 10
    uv run evaluate --experiment price --model claude-opus-5-5 --model claude-haiku-4-5
    uv run evaluate --experiment price --model claude-opus-5-5 2026-06-30 --model claude-haiku-4-5 2025-07-31

--experiment (benchmark.experiment.EXPERIMENTS) fixes the dataset, the prompt
and the scorers. Forecasts come from a local Claude Code agent
(claude-agent-sdk); see predict_agent.py for what it is logged in as.

Each invocation is one parent MLflow run with a nested child run per
--model argument. A model given with a date (--model MODEL YYYY-MM-DD) is
only evaluated on rows whose as_of is after that date (its training cutoff,
say); models with the same cutoff, or none, get the same sample of rows.

Results land in the MLflow experiment config.EXPERIMENT_NAME at the
tracking URI; browse them with
    uv run mlflow-ui
"""

import argparse
import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    # only rows whose as_of is after this date are evaluated; None for all
    cutoff: date | None


DEFAULT_MODEL = ModelSpec(name="claude-opus-4-6", cutoff=None)


class AppendModelSpec(argparse.Action):
    """Collects each `--model MODEL [YYYY-MM-DD]` into a list of ModelSpec."""

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > 2:
            parser.error(f"{option_string} takes a model and at most one date")
        cutoff = None
        if len(values) == 2:
            try:
                cutoff = date.fromisoformat(values[1])
            except ValueError:
                parser.error(f"{option_string}: invalid date {values[1]!r}")
        model = ModelSpec(name=values[0], cutoff=cutoff)
        models = getattr(namespace, self.dest) or []
        setattr(namespace, self.dest, [*models, model])


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
        "--model",
        dest="models",
        nargs="+",
        action=AppendModelSpec,
        metavar=("MODEL", "YYYY-MM-DD"),
        help="an Anthropic model id to sweep, optionally followed by a cutoff "
        "date: the model is then only evaluated on rows whose as_of is after "
        f"it. Repeat for each model; default {DEFAULT_MODEL.name}",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of rows to evaluate",
    )
    parser.add_argument("--run-name")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    experiment = EXPERIMENTS[args.experiment]
    dataset = get_dataset(name=experiment.dataset_name)
    data = dataset.to_df()
    run_name = args.run_name or f"sweep-{datetime.now():%Y%m%d-%H%M}"
    # a parent run per invocation holding the shared settings, with one child
    # run per model; the sample is seeded, so models with the same cutoff get
    # the same rows and can be compared trace-for-trace
    with mlflow.start_run(run_name=run_name):
        for model in args.models or [DEFAULT_MODEL]:
            model_data = data
            if model.cutoff:
                model_data = data[after_cutoff(data, model.cutoff)]
            if max_rows := args.max_rows:
                sample_size = min(max_rows, len(model_data))
                model_data = model_data.sample(n=sample_size, random_state=0)
            predict_fn = partial(predict_with_agent, model.name, args.experiment)
            with mlflow.start_run(
                run_name=f"{model.name}-{args.experiment}", nested=True
            ):
                mlflow.set_active_model(name=model.name)
                mlflow.log_params(
                    {"experiment": args.experiment, "cutoff": model.cutoff}
                )
                mlflow.genai.evaluate(
                    data=model_data,
                    predict_fn=predict_fn,
                    scorers=experiment.scorers,
                )


if __name__ == "__main__":
    main()
