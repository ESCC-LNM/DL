"""Command-line entry points for the single released model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import dataset, loader, validate_data
from .metrics import calculate_metrics
from .model import build_model
from .tables import load_table, scaler_from_dict
from .utils import load_checkpoint, save_json
from .workflows import train_cv, train_fulltrain

DEFAULT_CONFIG = Path("configs/default.json")


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def cv_main() -> None:
    parser = common_parser("Train five fixed folds and create OOF predictions.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output_dir or Path(config["paths"]["cv_output"])
    result = train_cv(config, args.device, output, check_only=args.check_only)
    print(json.dumps(result, indent=2))


def fulltrain_main() -> None:
    parser = common_parser(
        "Train once on the full training pool; lock the OOF threshold."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="0: use CV P75 best epoch; N>0: train exactly N epochs.",
    )
    parser.add_argument("--cv-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 0:
        parser.error("--epochs must be >= 0.")
    config = load_config(args.config)
    result = train_fulltrain(
        config,
        args.device,
        args.output_dir or Path(config["paths"]["fulltrain_output"]),
        args.epochs,
        args.cv_dir or Path(config["paths"]["cv_output"]),
        check_only=args.check_only,
    )
    print(json.dumps(result, indent=2))


def validate_main() -> None:
    parser = common_parser(
        "Validate supplied cohorts, fixed folds, features, and files."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(validate_data(config), indent=2))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate or predict one cohort using an exported full-train checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="Predict without outcome labels; metrics and label columns are omitted.",
    )
    args = parser.parse_args()
    bundle = load_checkpoint(args.checkpoint)
    if bundle["threshold_source"] != "cv_oof":
        raise ValueError(
            "Evaluation requires a final checkpoint with a locked OOF threshold."
        )
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    table = load_table(args.table, require_labels=not args.unlabeled)
    config = {
        "parameters": {
            **bundle["preprocessing"],
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": 42,
        },
        "cohorts": {
            "inference": {"images": str(args.images), "masks": str(args.masks)}
        },
    }
    scaler = scaler_from_dict(bundle["scaler"])
    ds = dataset(config, table, "inference", bundle["features"], scaler)
    model = build_model(**bundle["model_config"]).to(args.device).eval()
    model.load_state_dict(bundle["model"], strict=True)
    probabilities, ids, patient_ids, labels = [], [], [], []
    device = torch.device(args.device)
    with torch.no_grad():
        for batch in loader(config, ds):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda" and bundle.get("amp", True),
            ):
                out = model(
                    batch["tumor_bag"].to(device),
                    batch["context_bag"].to(device),
                    batch["rad_features"].to(device),
                )
                prob = torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
            if not np.isfinite(prob).all():
                raise FloatingPointError("Non-finite probabilities during inference.")
            probabilities.extend(prob.tolist())
            ids.extend(batch["id"])
            patient_ids.extend(batch["patientID"])
            labels.extend(batch["label"].tolist())
    frame = pd.DataFrame(
        {
            "ID": ids,
            "patientID": patient_ids,
            "label": labels,
            "prob": probabilities,
        }
    )
    threshold = bundle["threshold"]
    metrics = (
        None
        if args.unlabeled
        else calculate_metrics(frame.label, frame.prob, threshold)
    )
    if args.unlabeled:
        frame = frame.drop(columns="label")
    frame["prediction"] = (frame.prob >= threshold).astype(int)
    args.output_dir.mkdir(parents=True)
    frame.to_csv(args.output_dir / "predictions.csv", index=False)
    result = {
        "threshold": threshold,
        "threshold_source": "cv_oof",
        "n": len(frame),
        "metrics": metrics,
    }
    save_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2))
