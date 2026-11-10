"""Portable JSON configuration; relative paths are relative to the config file."""

import argparse
import copy
import inspect
import json
from pathlib import Path

from .model import build_model

COHORTS = ("train", "internal", "external1", "external2")


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected config schema_version=1.")
    for key in ("parameters", "paths", "cohorts"):
        if key not in config:
            raise ValueError(f"Missing config section: {key}")
    for name in COHORTS:
        if name not in config["cohorts"]:
            raise ValueError(f"Missing cohort: {name}")
    config = copy.deepcopy(config)
    for key, value in config["paths"].items():
        config["paths"][key] = str((path.parent / value).resolve())
    for cohort in config["cohorts"].values():
        for key, value in cohort.items():
            cohort[key] = str((path.parent / value).resolve())
    params = config["parameters"]
    if params["num_classes"] != 2 or params["num_centers"] % 2 != 1:
        raise ValueError(
            "This release requires binary classification and odd num_centers."
        )
    if params["batch_size"] < 1 or params["num_workers"] < 0:
        raise ValueError("Invalid batch_size or num_workers.")
    if params["epochs"] < 1 or params["patience"] < 1:
        raise ValueError("CV epochs and patience must be positive.")
    if params["threshold_mode"] != "constrained_balanced":
        raise ValueError("This release locks the constrained_balanced OOF rule.")
    if params["eval_tta_mode"] != "none":
        raise ValueError("The released evaluation protocol uses eval_tta_mode=none.")
    if (
        not 0 <= params["weight_avg_start_ratio"] <= 1
        or params["weight_avg_min_count"] < 1
    ):
        raise ValueError("Invalid tail weight averaging settings.")
    return config


def training_args(config: dict, epochs: int, *, pretrained: bool = True):
    params = dict(config["parameters"])
    params["epochs"] = epochs
    params["pretrained"] = bool(params["pretrained"] and pretrained)
    params["pretrained_ckpt"] = config["paths"]["pretrained"]
    return argparse.Namespace(**params)


def model_config(config: dict, *, radiomics_dim: int = 45) -> dict:
    accepted = set(inspect.signature(build_model).parameters)
    params = {
        key: value for key, value in config["parameters"].items() if key in accepted
    }
    params.update(radiomics_dim=radiomics_dim, pretrained=False)
    return params


def preprocessing_config(config: dict) -> dict:
    names = (
        "input_size",
        "normalize_mode",
        "crop_margin",
        "context_margin",
        "num_centers",
        "center_step",
        "hu_min",
        "hu_max",
    )
    return {name: config["parameters"][name] for name in names}


def validate_input_paths(config: dict) -> None:
    required = [config["paths"]["protocol"]]
    if config["parameters"]["pretrained"]:
        required.append(config["paths"]["pretrained"])
    for cohort in config["cohorts"].values():
        required.extend(cohort.values())
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing input paths:\n" + "\n".join(missing))
