import multiprocessing as mp
import os
import uuid

import wandb


run = None


def get_wandb_run():
    global run
    if run is not None:
        return run

    # Avoid initializing wandb in spawned worker processes.
    if mp.current_process().name != "MainProcess":
        return None

    wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()
    if wandb_api_key and wandb_api_key != "your_wandb_api_key_here":
        wandb.login(key=wandb_api_key)

    wandb_entity = os.getenv("WANDB_ENTITY", "").strip()
    wandb_project = os.getenv("WANDB_PROJECT", "funsearch").strip() or "funsearch"
    wandb_run_name = os.getenv("WANDB_RUN_NAME", "funsearch_new_qwen3_coder_30b").strip()

    run_id = f"funsearch_{uuid.uuid4().hex[:10]}"
    run = wandb.init(
        id=run_id,
        entity=wandb_entity or None,
        project=wandb_project,
        name=wandb_run_name,
    )
    return run


def log_metrics(metrics: dict) -> None:
    current_run = get_wandb_run()
    if current_run is not None:
        current_run.log(metrics)
