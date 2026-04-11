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


if __name__ == "__main__":
    _load_env()
    run = record_wandb.get_wandb_run()

    try:
        with open("templete/bin_online.txt", "r") as f:
            specification = f.read()

        or3 = dataset.get_dataset_or3()

        logging.info("Starting funsearch...")
        funsearch.main(specification=specification, inputs=[or3], config=config_lib.Config())
    finally:
        if run is not None:
            run.finish()
    