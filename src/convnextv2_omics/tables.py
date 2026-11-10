"""Feature tables, portable scaler state, and exact volume matching."""

import random
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = ("ID", "id", "Patient_ID", "patientID", "patient_id", "PatientID")
PATIENT_COLUMNS = ("patientID", "Patient_ID", "patient_id", "PatientID")
VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")


def normalize_id(value: Any) -> str:
    """Normalize numeric IDs in the same way as the research pipeline."""
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("\ufeff", "")
    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def load_table(path: Path, *, require_labels: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    frame.columns = frame.columns.str.strip().str.replace("\ufeff", "", regex=False)
    id_col = next((c for c in ID_COLUMNS if c in frame), None)
    if id_col is None:
        raise ValueError(f"No sample ID column in {path}")
    patient_col = next((c for c in PATIENT_COLUMNS if c in frame), id_col)
    ids = frame[id_col].map(normalize_id)
    patient_ids = frame[patient_col].map(normalize_id)
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"Empty or duplicate sample IDs in {path}")
    frame["ID"] = ids
    frame["patientID"] = patient_ids
    if "label" not in frame:
        if require_labels:
            raise ValueError(f"No binary label column in {path}")
        frame["label"] = 0
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"Labels must be exactly 0 or 1 in {path}")
    frame["label"] = labels.astype(int)
    if frame.empty:
        raise ValueError(f"Empty table: {path}")
    return frame


def _ensure_selected_features_exist(
    df: pd.DataFrame, selected_features: Sequence[str], cohort_name: str
) -> pd.DataFrame:
    missing = set(selected_features) - set(df.columns)
    if missing:
        raise ValueError(f"[{cohort_name}] Missing features: {sorted(missing)[:10]}")
    return df.copy()


def feature_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    _ensure_selected_features_exist(frame, features, "radiomics")
    matrix = frame.loc[:, list(features)].apply(pd.to_numeric, errors="raise")
    array = matrix.to_numpy(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Selected radiomics features must all be finite.")
    return array


class RadiomicsFeatureProcessor:
    """Apply a scaler fitted exclusively on the corresponding training cohort."""

    def __init__(self, selected_features: Sequence[str], scaler: StandardScaler):
        self.selected_features = list(selected_features)
        self.scaler = scaler

    def transform_dataframe(self, df: pd.DataFrame, split: str) -> np.ndarray:
        return self.scaler.transform(feature_matrix(df, self.selected_features)).astype(
            np.float32
        )


def scaler_to_dict(scaler: StandardScaler) -> dict:
    """Convert StandardScaler to JSON-safe values without pickling objects."""
    if not isinstance(scaler, StandardScaler):
        raise TypeError("The fixed protocol requires a StandardScaler.")
    return {
        "class": "StandardScaler",
        "with_mean": bool(scaler.with_mean),
        "with_std": bool(scaler.with_std),
        "mean": None if scaler.mean_ is None else scaler.mean_.tolist(),
        "scale": None if scaler.scale_ is None else scaler.scale_.tolist(),
        "var": None if scaler.var_ is None else scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": np.asarray(scaler.n_samples_seen_).tolist(),
    }


def scaler_from_dict(state: dict) -> StandardScaler:
    if state.get("class") != "StandardScaler":
        raise ValueError("Only a StandardScaler state is supported.")
    scaler = StandardScaler(with_mean=state["with_mean"], with_std=state["with_std"])
    for field, key in (("mean_", "mean"), ("scale_", "scale"), ("var_", "var")):
        value = state[key]
        setattr(
            scaler, field, None if value is None else np.asarray(value, dtype=float)
        )
    scaler.n_features_in_ = int(state["n_features_in"])
    scaler.n_samples_seen_ = np.asarray(state["n_samples_seen"])
    return scaler


def _read_volume(path: Path) -> np.ndarray:
    try:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
    except RuntimeError:
        import nibabel as nib

        array = np.asanyarray(nib.load(str(path)).dataobj).transpose(2, 1, 0)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume: {path}")
    return array.astype(np.float32)


def _resolve_volume_file(sample_id: str, directory: Path, is_mask: bool) -> Path:
    names = []
    for ext in VOLUME_EXTENSIONS:
        names.append(f"{sample_id}{ext}")
        if is_mask:
            names.extend(
                f"{sample_id}{suffix}{ext}" for suffix in ("_mask", "-mask", "_seg")
            )
    matches = [directory / name for name in names if (directory / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one exact volume match for {sample_id} in {directory}; "
            f"found {len(matches)}. Prefix matching is deliberately not used."
        )
    return matches[0]


def seed_worker(worker_id: int, *, seed: int) -> None:
    """Top-level, picklable worker initializer for Windows spawn."""
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    import torch

    torch.manual_seed(worker_seed)


def worker_initializer(seed: int):
    return partial(seed_worker, seed=seed)
