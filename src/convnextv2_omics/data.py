"""Fixed-fold loading and train-only radiomics preprocessing."""

import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ._dataset import DualContextDataset
from .config import COHORTS
from .tables import (
    _resolve_volume_file,
    feature_matrix,
    load_table,
    normalize_id,
    worker_initializer,
)


def load_cohorts(config: dict) -> dict:
    return {
        name: load_table(Path(config["cohorts"][name]["table"])) for name in COHORTS
    }


def read_protocol(path: Path, train: pd.DataFrame) -> list:
    """Read existing folds only; never generate or reassign a split."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    folds = [raw[f"Fold_{i}"] for i in range(1, 6)]
    pool = set(train["ID"])
    validation_ids = []
    reference_features = folds[0]["selected_features"]
    if len(reference_features) != 45 or len(set(reference_features)) != 45:
        raise ValueError(
            "The fixed protocol must contain 45 distinct radiomics features."
        )
    for i, fold in enumerate(folds, 1):
        for key in ("train_ids", "val_ids"):
            fold[key] = [normalize_id(value) for value in fold[key]]
            if len(fold[key]) != len(set(fold[key])):
                raise ValueError(f"Duplicate {key} in fold {i}.")
        training, validation = set(fold["train_ids"]), set(fold["val_ids"])
        if (
            not training
            or not validation
            or training & validation
            or training | validation != pool
        ):
            raise ValueError(
                f"Fold {i} is not an exact disjoint partition of training IDs."
            )
        if fold["selected_features"] != reference_features:
            raise ValueError(
                "Feature names and their CV order must be fixed across all folds."
            )
        train_patients = set(train.loc[train.ID.isin(training), "patientID"])
        val_patients = set(train.loc[train.ID.isin(validation), "patientID"])
        if train_patients & val_patients:
            raise ValueError(f"Patient overlap within fold {i}.")
        validation_ids.extend(validation)
        scaler_name = Path(str(fold["scaler_file"]).replace("\\", "/")).name
        fold["scaler_path"] = str(path.parent / scaler_name)
        feature_matrix(train, fold["selected_features"])
    if len(validation_ids) != len(pool) or set(validation_ids) != pool:
        raise ValueError(
            "OOF validation IDs must cover each training sample exactly once."
        )
    return folds


def load_fold_scaler(fold: dict) -> StandardScaler:
    """Only load trusted, researcher-provided protocol artifacts."""
    scaler = joblib.load(fold["scaler_path"])
    if not isinstance(scaler, StandardScaler) or scaler.n_features_in_ != 45:
        raise ValueError("Expected a fitted 45-feature StandardScaler.")
    return scaler


def dataset(
    config: dict,
    frame: pd.DataFrame,
    cohort: str,
    features: list,
    scaler: StandardScaler,
    *,
    augment: bool = False,
) -> DualContextDataset:
    params = config["parameters"]
    paths = config["cohorts"][cohort]
    hu = (params["hu_min"], params["hu_max"])
    return DualContextDataset(
        df=frame,
        image_dir=Path(paths["images"]),
        mask_dir=Path(paths["masks"]),
        selected_features=features,
        scaler=scaler,
        split="train" if cohort == "train" else cohort,
        apply_aug=augment,
        aug_profile=params.get("aug_profile", "light"),
        hu_clip=None if None in hu else hu,
        normalize_mode=params["normalize_mode"],
        target_size=(params["input_size"], params["input_size"]),
        crop_margin=params["crop_margin"],
        context_margin=params["context_margin"],
        num_centers=params["num_centers"],
        center_step=params["center_step"],
        seed=params.get("seed", 42),
    )


def loader(config: dict, ds: DualContextDataset, *, shuffle: bool = False, seed=None):
    params = config["parameters"]
    seed = params.get("seed", 42) if seed is None else seed
    generator = torch.Generator().manual_seed(seed)
    workers = params["num_workers"]
    return DataLoader(
        ds,
        batch_size=params["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
        worker_init_fn=worker_initializer(seed) if workers > 0 else None,
        generator=generator,
    )


def fulltrain_features(folds: list) -> list:
    return sorted(folds[0]["selected_features"])


def prepare_fulltrain(config: dict, frames: dict, folds: list):
    features = fulltrain_features(folds)
    scaler = clone(load_fold_scaler(folds[0]))
    scaler.fit(feature_matrix(frames["train"], features))
    datasets = {
        name: dataset(config, frames[name], name, features, scaler) for name in COHORTS
    }
    train_ds = dataset(
        config,
        frames["train"],
        "train",
        features,
        scaler,
        augment=config["parameters"]["apply_aug"],
    )
    return features, scaler, datasets, train_ds


def validate_data(config: dict, *, check_volumes: bool = True) -> dict:
    frames = load_cohorts(config)
    folds = read_protocol(Path(config["paths"]["protocol"]), frames["train"])
    for fold in folds:
        load_fold_scaler(fold)
    report = {}
    for name, frame in frames.items():
        feature_matrix(frame, folds[0]["selected_features"])
        if frame["label"].nunique() != 2:
            raise ValueError(f"{name} must contain both classes for AUC.")
        if check_volumes:
            root = config["cohorts"][name]
            for sample_id in frame.ID:
                _resolve_volume_file(sample_id, Path(root["images"]), False)
                _resolve_volume_file(sample_id, Path(root["masks"]), True)
        report[name] = {
            "n": len(frame),
            "positive": int(frame.label.sum()),
            "negative": int((frame.label == 0).sum()),
        }
    report["folds"] = 5
    report["features"] = len(folds[0]["selected_features"])
    return report
