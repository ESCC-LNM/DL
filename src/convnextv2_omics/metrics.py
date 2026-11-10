"""Binary metrics and the locked OOF operating-threshold rule."""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def calculate_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    y_pred = (y_prob >= float(threshold)).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sens = spec = 0.0
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "AUC": auc,
        "ACC": acc,
        "F1": f1,
        "Sensitivity": sens,
        "Specificity": spec,
        "Threshold": float(threshold),
    }


def prediction_dict_to_frame(pred: Dict[str, np.ndarray]) -> pd.DataFrame:
    data = {
        "ID": pred["ids"].astype(str),
        "patientID": pred["patient_ids"].astype(str),
        "label": pred["labels"].astype(int),
        "prob": pred["probs"].astype(float),
    }
    n = len(data["ID"])
    for key, value in pred.items():
        if key in {"ids", "patient_ids", "labels", "probs"}:
            continue
        arr = np.asarray(value)
        if arr.ndim == 1 and len(arr) == n:
            data[key] = arr.astype(float)
    return pd.DataFrame(data)


def summarize_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"AUC={metrics['AUC']:.4f} | ACC={metrics['ACC']:.4f} | F1={metrics['F1']:.4f} | "
        f"Sens={metrics['Sensitivity']:.4f} | Spec={metrics['Specificity']:.4f} | Thr={metrics['Threshold']:.4f}"
    )


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_sensitivity: float = 0.75,
    min_specificity: float = 0.70,
) -> tuple:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("OOF threshold selection requires both binary classes.")
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must be equally sized 1D arrays.")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0).all() and (probabilities <= 1).all()
    ):
        raise ValueError("Probabilities must be finite values in [0, 1].")
    unique = np.unique(probabilities)
    mids = (unique[:-1] + unique[1:]) * 0.5
    candidates = np.unique(np.concatenate(([0.0], mids, unique, [1.0])))
    positive = np.sort(probabilities[labels == 1])
    negative = np.sort(probabilities[labels == 0])
    sens = (len(positive) - np.searchsorted(positive, candidates, side="left")) / len(
        positive
    )
    spec = np.searchsorted(negative, candidates, side="left") / len(negative)
    sd = np.maximum(0, min_sensitivity - sens)
    pd = np.maximum(0, min_specificity - spec)
    scores = np.where(
        (sd <= 1e-12) & (pd <= 1e-12),
        1000.0 + sens + spec,
        -(sd + pd) * 100.0 + sens + spec,
    )
    threshold = float(candidates[np.argmax(scores)])
    return threshold, calculate_metrics(labels, probabilities, threshold)
