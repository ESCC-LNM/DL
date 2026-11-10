"""Dataset implementation shared by CV, full-train, and inference."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    _apply_bag_aug,
    _extract_dual_context_25d_bags,
    _sample_bag_aug_params,
)
from .tables import (
    RadiomicsFeatureProcessor,
    _ensure_selected_features_exist,
    _read_volume,
    _resolve_volume_file,
)


class DualContextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        mask_dir: Path,
        selected_features: Sequence[str],
        scaler: Any,
        feature_processor: Optional[RadiomicsFeatureProcessor] = None,
        split: str = "train",
        apply_aug: bool = False,
        aug_profile: str = "light",
        hu_clip: Optional[Tuple[float, float]] = None,
        normalize_mode: str = "zscore",
        target_size: Optional[Tuple[int, int]] = (48, 48),
        crop_margin: int = 6,
        context_margin: int = 18,
        num_centers: int = 7,
        center_step: int = 1,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.selected_features = [str(x).strip() for x in selected_features]
        self.scaler = scaler
        self.feature_processor = feature_processor
        self.split = str(split)
        self.apply_aug = bool(apply_aug and split == "train")
        self.aug_profile = str(aug_profile).lower()
        self.hu_clip = hu_clip
        self.normalize_mode = str(normalize_mode)
        self.target_size = target_size
        self.crop_margin = int(crop_margin)
        self.context_margin = int(context_margin)
        self.num_centers = int(num_centers)
        self.center_step = int(center_step)
        self.seed = int(seed)

        if self.num_centers <= 0 or self.num_centers % 2 == 0:
            raise ValueError("num_centers must be a positive odd integer.")
        if self.center_step <= 0:
            raise ValueError("center_step must be >= 1.")
        if self.scaler is None and self.feature_processor is None:
            raise ValueError("A fitted training scaler is required.")
        if len(self.selected_features) == 0:
            raise ValueError("selected_features must not be empty.")

        self.df = _ensure_selected_features_exist(
            self.df, self.selected_features, cohort_name=self.split
        )
        self.rad_features = self._prepare_radiomics_matrix(self.df)
        self.rad_dim = int(self.rad_features.shape[1])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sample_id = str(row["ID"])
        patient_id = str(row["patientID"])
        label = int(row["label"])

        image_path = _resolve_volume_file(sample_id, self.image_dir, is_mask=False)
        mask_path = _resolve_volume_file(sample_id, self.mask_dir, is_mask=True)

        image = _read_volume(image_path)
        mask = _read_volume(mask_path)
        tumor_bag, context_bag, max_slice, center_indices = (
            _extract_dual_context_25d_bags(
                image=image,
                mask=mask,
                hu_clip=self.hu_clip,
                normalize_mode=self.normalize_mode,
                target_size=self.target_size,
                crop_margin=self.crop_margin,
                context_margin=self.context_margin,
                num_centers=self.num_centers,
                center_step=self.center_step,
            )
        )

        if self.apply_aug:
            aug_params = _sample_bag_aug_params(
                profile=self.aug_profile, h=tumor_bag.shape[-2], w=tumor_bag.shape[-1]
            )
            tumor_bag = _apply_bag_aug(
                tumor_bag, normalize_mode=self.normalize_mode, aug_params=aug_params
            )
            context_bag = _apply_bag_aug(
                context_bag, normalize_mode=self.normalize_mode, aug_params=aug_params
            )

        return {
            "tumor_bag": torch.from_numpy(tumor_bag.astype(np.float32, copy=False)),
            "context_bag": torch.from_numpy(context_bag.astype(np.float32, copy=False)),
            "rad_features": torch.from_numpy(self.rad_features[idx]),
            "label": torch.tensor(label, dtype=torch.long),
            "id": sample_id,
            "sample_key": sample_id,
            "patientID": patient_id,
            "max_slice": torch.tensor(max_slice, dtype=torch.long),
            "center_indices": torch.tensor(center_indices, dtype=torch.long),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }

    def _prepare_radiomics_matrix(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_processor is not None:
            return self.feature_processor.transform_dataframe(df, split=self.split)
        x = df.loc[:, self.selected_features].apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)
        if x.isna().any().any():
            bad_cols = x.columns[x.isna().any()].tolist()
            raise ValueError(
                f"[{self.split}] non-finite radiomics values in selected features: {bad_cols[:10]}"
            )
        x_np = x.to_numpy(dtype=np.float32)

        scaler_n_features = getattr(self.scaler, "n_features_in_", None)
        if scaler_n_features is not None and int(scaler_n_features) != int(
            x_np.shape[1]
        ):
            raise ValueError(
                f"Scaler and selected feature dimensions differ: scaler={scaler_n_features}, features={x_np.shape[1]}"
            )
        if not hasattr(self.scaler, "transform"):
            raise TypeError("The scaler must implement transform().")
        x_scaled = self.scaler.transform(x_np)
        return np.asarray(x_scaled, dtype=np.float32)
