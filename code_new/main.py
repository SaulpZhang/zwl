import argparse
import dataclasses
from funsearch.implementation import config as config_lib
from funsearch.implementation import funsearch
import dataset
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import record_wandb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [thread:%(threadName)s/%(thread)d] - %(message)s')


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)

    if not os.getenv("LLM_API_KEY"):
        raise ValueError("LLM_API_KEY is missing. Please configure it in code_new/.env.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FunSearch with optional program clustering.")
    parser.add_argument(
        "--cluster_mode",
        nargs="?",
        default=1,
        type=int,
        choices=[0, 1],
        help="0: run original FunSearch (score signature), 1: run clustering FunSearch (embedding signature).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _load_env()

    base_config = config_lib.Config()
    use_program_clustering = args.cluster_mode == 1
    programs_database_config = dataclasses.replace(
        base_config.programs_database,
        use_program_clustering=use_program_clustering,
    )
    run_config = dataclasses.replace(
        base_config,
        programs_database=programs_database_config,
    )

    logging.info(
        "Starting mode: %s",
        "clustering" if use_program_clustering else "original",
    )

    run = record_wandb.get_wandb_run()

    try:
        with open("templete/bin_online.txt", "r") as f:
            specification = f.read()

        or3 = dataset.get_dataset_or3()

        logging.info("Starting funsearch...")
        funsearch.main(specification=specification, inputs=[or3], config=run_config)
    finally:
        if run is not None:
            run.finish()
    