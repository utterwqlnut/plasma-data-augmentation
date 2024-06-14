import glob
import os
import wandb
import socket
import subprocess
import shutil
import numpy as np
import random
import torch

def set_up_wandb(seed, parsed_args):
    """Set up wandb for logging.

    Args:
        model (PlasmaTransformer): The model to be trained.
        training_args (TrainingArguments): The training arguments.
        seed (int): The random seed.

    Returns:
        None
    """
    wandb.init(project="testing dhruva_ttd",
               entity="timetodisrupt",)

    if not check_wandb_connection():
        os.environ["WANDB_MODE"] = "offline"
    sync_offline_wandb_runs()

    # zip model.config and training_args into a single dictionary
    wandb_config = {**parsed_args}
    wandb.config.update(wandb_config)
    wandb.log({"seed": seed})
    # os.environ["WANDB_LOG_MODEL"] = "end"

    return

def check_wandb_connection(host="api.wandb.ai", port=443, timeout=5):
    """Check if the machine is connected to wandb."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            return True
    except socket.error:
        return False


def sync_offline_wandb_runs():
    """Sync offline wandb runs with the wandb server. Deletes old runs."""
    local_runs = glob.glob("wandb/offline-run-*")
    if local_runs and check_wandb_connection():
        for run_dir in local_runs:
            subprocess.run(["wandb", "sync", run_dir])
            # Uncomment the following line to remove the synced run directory
            shutil.rmtree(run_dir)
    return

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False