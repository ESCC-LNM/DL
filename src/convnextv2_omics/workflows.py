"""Five-fold OOF fitting and full-train-once orchestration."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (
    COHORTS,
    model_config,
    preprocessing_config,
    training_args,
    validate_input_paths,
)
from .data import (
    dataset,
    load_cohorts,
    load_fold_scaler,
    loader,
    prepare_fulltrain,
    read_protocol,
    validate_data,
)
from .engine import Trainer
from .metrics import calculate_metrics, prediction_dict_to_frame, select_threshold
from .tables import scaler_to_dict
from .utils import (
    average_state,
    environment_info,
    protocol_fingerprint,
    save_json,
    set_seed,
    setup_logging,
    state_to_cpu,
)

log = logging.getLogger(__name__)


def resolve_epochs(requested: int, cv_summary: dict) -> tuple:
    if requested < 0:
        raise ValueError("--epochs must be >= 0.")
    if requested > 0:
        return requested, "manual"
    best = [int(fold["best_epoch"]) for fold in cv_summary["fold_results"]]
    if len(best) != 5 or min(best) < 1:
        raise ValueError("Automatic epochs require five positive CV best epochs.")
    return max(1, int(round(float(np.percentile(best, 75))))), "cv_p75"


def lock_threshold(oof: pd.DataFrame, train: pd.DataFrame, config: dict) -> tuple:
    if oof.ID.duplicated().any() or set(oof.ID) != set(train.ID):
        raise ValueError("OOF must cover each training ID exactly once.")
    expected = train.set_index("ID").label
    if not oof.set_index("ID").label.eq(expected.reindex(oof.ID)).all():
        raise ValueError("OOF labels differ from the supplied training table.")
    params = config["parameters"]
    return select_threshold(
        oof.label.to_numpy(),
        oof.prob.to_numpy(),
        params["threshold_min_sens"],
        params["threshold_min_spec"],
    )


def predict_frame(trainer: Trainer, data_loader) -> pd.DataFrame:
    return prediction_dict_to_frame(trainer.predict(data_loader))


def save_bundle(
    path: Path,
    config: dict,
    model_state: dict,
    features: list,
    scaler,
    *,
    epochs: int,
    threshold: float,
    source: str,
) -> None:
    """Store tensors and plain metadata; no paths or patient predictions."""
    bundle = {
        "format_version": 1,
        "model_name": "convnextv2_atto_omics_prior",
        "model_config": model_config(config, radiomics_dim=len(features)),
        "model": state_to_cpu(model_state),
        "amp": bool(config["parameters"]["amp"]),
        "features": list(features),
        "scaler": scaler_to_dict(scaler),
        "preprocessing": preprocessing_config(config),
        "threshold": float(threshold),
        "threshold_source": source,
        "epochs": epochs,
        "environment": environment_info(),
    }
    torch.save(bundle, path)


def train_cv(config: dict, device: str, output: Path, *, check_only=False) -> dict:
    validate_input_paths(config)
    report = validate_data(config)
    if check_only:
        return report
    setup_logging(output)
    params = config["parameters"]
    set_seed(params["seed"])
    frames = load_cohorts(config)
    pool = frames["train"]
    folds = read_protocol(Path(config["paths"]["protocol"]), pool)
    oof_frames, fold_results = [], []
    save_json(output / "run_config.json", config)
    save_json(output / "environment.json", environment_info())
    for index, fold in enumerate(folds, 1):
        fold_dir = output / f"fold_{index}"
        fold_dir.mkdir()
        features = fold["selected_features"]
        scaler = load_fold_scaler(fold)
        training = pool.loc[pool.ID.isin(fold["train_ids"])].copy()
        validation = pool.loc[pool.ID.isin(fold["val_ids"])].copy()
        train_ds = dataset(
            config,
            training,
            "train",
            features,
            scaler,
            augment=params["apply_aug"],
        )
        train_loader = loader(config, train_ds, shuffle=True)
        val_loader = loader(
            config, dataset(config, validation, "train", features, scaler)
        )
        train_ds[0]
        args = training_args(config, params["epochs"])
        trainer = Trainer(args, torch.device(device), len(features))
        best_auc, best_epoch, best_state, stale = -1.0, 0, None, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            stats = trainer.train_one_epoch(train_loader, epoch)
            val_frame = predict_frame(trainer, val_loader)
            auc = calculate_metrics(val_frame.label, val_frame.prob, 0.5)["AUC"]
            history.append({"epoch": epoch, **stats, "val_auc": auc})
            log.info(
                "Fold %d epoch %d | loss=%.6f | val_auc=%.4f",
                index,
                epoch,
                stats["loss"],
                auc,
            )
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                best_state = state_to_cpu(trainer.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        trainer.model.load_state_dict(best_state)
        val_frame = predict_frame(trainer, val_loader)
        val_frame["fold"] = index
        oof_frames.append(val_frame)
        val_frame.to_csv(fold_dir / "val_predictions.csv", index=False)
        training_eval = loader(
            config,
            dataset(config, training, "train", features, scaler),
            seed=params["seed"] + index - 1 + 1000,
        )
        predict_frame(trainer, training_eval).to_csv(
            fold_dir / "train_predictions.csv", index=False
        )
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        save_bundle(
            fold_dir / "model.pt",
            config,
            best_state,
            features,
            scaler,
            epochs=best_epoch,
            threshold=0.5,
            source="not_locked_fold_checkpoint",
        )
        fold_results.append(
            {"fold": index, "best_epoch": best_epoch, "val_auc": best_auc}
        )
    oof = (
        pd.concat(oof_frames, ignore_index=True)
        .sort_values("ID")
        .reset_index(drop=True)
    )
    threshold, metrics = lock_threshold(oof, pool, config)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    summary = {
        "model_name": "convnextv2_atto_omics_prior",
        "fold_results": fold_results,
        "oof_threshold": threshold,
        "oof_metrics": metrics,
        "threshold_source": "cv_oof",
        "protocol_fingerprint": protocol_fingerprint(pool, folds),
        "parameters": params,
    }
    save_json(output / "cv_results.json", summary)
    return summary


def train_fulltrain(
    config: dict,
    device: str,
    output: Path,
    requested_epochs: int,
    cv_dir: Path,
    *,
    check_only=False,
) -> dict:
    validate_input_paths(config)
    report = validate_data(config)
    summary = json.loads((cv_dir / "cv_results.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(
        cv_dir / "oof_predictions.csv",
        dtype={"ID": str, "patientID": str},
        float_precision="round_trip",
    )
    frames = load_cohorts(config)
    folds = read_protocol(Path(config["paths"]["protocol"]), frames["train"])
    fingerprint = protocol_fingerprint(frames["train"], folds)
    if summary["protocol_fingerprint"] != fingerprint:
        raise ValueError(
            "CV results were produced with a different training pool or fold protocol."
        )
    if summary["parameters"] != config["parameters"]:
        raise ValueError(
            "CV and full-train fixed parameters differ; use the same config."
        )
    epochs, epoch_source = resolve_epochs(requested_epochs, summary)
    threshold, oof_metrics = lock_threshold(oof, frames["train"], config)
    report.update(epochs=epochs, epoch_source=epoch_source, oof_threshold=threshold)
    if check_only:
        return report
    setup_logging(output)
    params = config["parameters"]
    set_seed(params["seed"])
    features, scaler, datasets, train_ds = prepare_fulltrain(config, frames, folds)
    train_loader = loader(config, train_ds, shuffle=True)
    train_ds[0]
    args = training_args(config, epochs)
    trainer = Trainer(args, torch.device(device), len(features))
    average, count = None, 0
    start = max(1, int(round(epochs * params["weight_avg_start_ratio"])))
    history = []
    save_json(output / "run_config.json", config)
    for epoch in range(1, epochs + 1):
        stats = trainer.train_one_epoch(train_loader, epoch)
        history.append({"epoch": epoch, **stats})
        if params["weight_avg"] and epoch >= start:
            count += 1
            average = average_state(average, trainer.model.state_dict(), count)
        log.info(
            "Epoch %d/%d | loss=%.6f | lr=%.3e",
            epoch,
            epochs,
            stats["loss"],
            stats["lr"],
        )
    state = state_to_cpu(trainer.model.state_dict())
    state_source = "last"
    if average is not None and count >= params["weight_avg_min_count"]:
        state, state_source = average, f"tail_weight_avg({count})"
    trainer.model.load_state_dict(state)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    metrics = {}
    for cohort in COHORTS:
        seed = params["seed"] + 1000 if cohort == "train" else params["seed"]
        frame = predict_frame(trainer, loader(config, datasets[cohort], seed=seed))
        frame.to_csv(output / f"{cohort}_predictions.csv", index=False)
        metrics[cohort] = calculate_metrics(frame.label, frame.prob, threshold)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    save_bundle(
        output / "model.pt",
        config,
        state,
        features,
        scaler,
        epochs=epochs,
        threshold=threshold,
        source="cv_oof",
    )
    result = {
        "model_name": "convnextv2_atto_omics_prior",
        "run_mode": "full_train_single_run",
        "final_epochs": epochs,
        "epoch_source": epoch_source,
        "state_source": state_source,
        "threshold": threshold,
        "threshold_source": "cv_oof",
        "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "metrics": {"oof": oof_metrics, **metrics},
        "protocol_fingerprint": fingerprint,
        "environment": environment_info(),
    }
    save_json(output / "metrics.json", result)
    return result
