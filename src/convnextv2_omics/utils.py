"""Run metadata, output protection, deterministic seeds, and checkpoints."""

import hashlib
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

from . import __version__


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output / "run.log", encoding="utf-8"),
        ],
        force=True,
    )


def save_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def environment_info() -> dict:
    import timm

    return {
        "package": __version__,
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "timm": timm.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def state_to_cpu(state: dict) -> dict:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def average_state(average: dict, current: dict, count: int) -> dict:
    current = state_to_cpu(current)
    if average is None:
        return current
    for key, value in current.items():
        if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
            average[key] = value
        elif value.dtype.is_floating_point:
            average[key].mul_((count - 1) / count).add_(value, alpha=1 / count)
        else:
            average[key] = value
    return average


def protocol_fingerprint(frame, folds: list) -> str:
    value = {
        "labels": list(zip(frame.ID.tolist(), frame.label.tolist())),
        "folds": [
            {key: fold[key] for key in ("train_ids", "val_ids", "selected_features")}
            for fold in folds
        ],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def load_checkpoint(path: Path) -> dict:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1:
        raise ValueError(
            "Expected a release-format checkpoint (model.pt from train_fulltrain.py)."
        )
    if len(bundle["features"]) != bundle["model_config"]["radiomics_dim"]:
        raise ValueError("Checkpoint feature dimension mismatch.")
    return bundle
