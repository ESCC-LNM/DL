"""ROI-based 2.5D extraction and synchronized dual-view augmentation."""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


def _find_max_slice_from_mask(mask: np.ndarray) -> int:
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D; got: {mask.shape}")
    areas = (mask > 0).reshape(mask.shape[0], -1).sum(axis=1)
    if np.all(areas == 0):
        return int(mask.shape[0] // 2)
    return int(np.argmax(areas))


def _build_center_indices(
    max_z: int, depth: int, num_centers: int, center_step: int
) -> List[int]:
    half = num_centers // 2
    centers = []
    for offset in range(-half, half + 1):
        z = max_z + offset * center_step
        z = max(0, min(depth - 1, z))
        centers.append(int(z))
    return centers


def _three_slice_indices(center_z: int, depth: int) -> List[int]:
    return [
        max(0, min(depth - 1, center_z - 1)),
        max(0, min(depth - 1, center_z)),
        max(0, min(depth - 1, center_z + 1)),
    ]


def _get_bbox2d(mask2d: np.ndarray, margin: int = 6) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask2d > 0)
    h, w = mask2d.shape
    if len(xs) == 0 or len(ys) == 0:
        return 0, h, 0, w
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin + 1)
    return y0, y1, x0, x1


def _resize_stack(x: np.ndarray, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    if target_size is None:
        return x.astype(np.float32, copy=False)
    th, tw = int(target_size[0]), int(target_size[1])
    if x.shape[1] == th and x.shape[2] == tw:
        return x.astype(np.float32, copy=False)
    zoom_h = th / max(1, x.shape[1])
    zoom_w = tw / max(1, x.shape[2])
    x = ndimage.zoom(x, (1.0, zoom_h, zoom_w), order=1)
    return x.astype(np.float32, copy=False)


def _normalize_stack(
    x: np.ndarray, hu_clip: Optional[Tuple[float, float]], normalize_mode: str
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if hu_clip is not None:
        x = np.clip(x, hu_clip[0], hu_clip[1])
    mode = str(normalize_mode).lower()
    if mode == "none":
        return x
    if mode == "zscore":
        mu = float(x.mean())
        sigma = float(x.std())
        if sigma < 1e-6:
            sigma = 1.0
        x = (x - mu) / sigma
        return np.clip(x, -5.0, 5.0).astype(np.float32)
    if mode == "minmax":
        vmin, vmax = float(x.min()), float(x.max())
        if vmax - vmin < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - vmin) / (vmax - vmin)).astype(np.float32)
    raise ValueError(f"Unsupported normalization: {normalize_mode}")


def _crop_stack(stack: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return stack[:, y0:y1, x0:x1]


def _extract_dual_context_25d_bags(
    image: np.ndarray,
    mask: np.ndarray,
    hu_clip: Optional[Tuple[float, float]] = None,
    normalize_mode: str = "zscore",
    target_size: Optional[Tuple[int, int]] = (48, 48),
    crop_margin: int = 6,
    context_margin: int = 18,
    num_centers: int = 7,
    center_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int, List[int]]:
    if image.shape != mask.shape:
        raise ValueError(f"Image and mask shapes differ: {image.shape} vs {mask.shape}")
    if image.ndim != 3:
        raise ValueError(f"Image must be 3D; got: {image.shape}")
    max_slice = _find_max_slice_from_mask(mask)
    centers = _build_center_indices(
        max_slice, image.shape[0], num_centers=num_centers, center_step=center_step
    )

    tumor_bag: List[np.ndarray] = []
    context_bag: List[np.ndarray] = []
    for center in centers:
        tri = _three_slice_indices(center, image.shape[0])
        stack = image[tri]
        center_mask = (mask[center] > 0).astype(np.uint8)
        tumor_bbox = _get_bbox2d(center_mask, margin=crop_margin)
        context_bbox = _get_bbox2d(center_mask, margin=context_margin)
        tumor = _crop_stack(stack, tumor_bbox)
        context = _crop_stack(stack, context_bbox)
        tumor = _resize_stack(tumor, target_size=target_size)
        context = _resize_stack(context, target_size=target_size)
        tumor = _normalize_stack(tumor, hu_clip=hu_clip, normalize_mode=normalize_mode)
        context = _normalize_stack(
            context, hu_clip=hu_clip, normalize_mode=normalize_mode
        )
        tumor_bag.append(tumor)
        context_bag.append(context)
    tumor_bag_np = np.stack(tumor_bag, axis=0).astype(np.float32, copy=False)
    context_bag_np = np.stack(context_bag, axis=0).astype(np.float32, copy=False)
    return tumor_bag_np, context_bag_np, int(max_slice), centers


def _sample_bag_aug_params(profile: str, h: int, w: int) -> Dict[str, float]:
    profile = str(profile).lower()
    if profile == "none":
        return {
            "rotate": 0.0,
            "shift_y": 0.0,
            "shift_x": 0.0,
            "contrast": 1.0,
            "bias": 0.0,
            "noise_std": 0.0,
            "flip": 0.0,
        }
    params = {
        "rotate": 0.0,
        "shift_y": 0.0,
        "shift_x": 0.0,
        "contrast": 1.0,
        "bias": 0.0,
        "noise_std": 0.0,
        "flip": 0.0,
    }
    if random.random() < 0.35:
        params["rotate"] = (
            random.uniform(-8.0, 8.0)
            if profile == "light"
            else random.uniform(-12.0, 12.0)
        )
    if random.random() < 0.25:
        shift_ratio = 0.05 if profile == "light" else 0.08
        params["shift_y"] = random.uniform(-shift_ratio * h, shift_ratio * h)
        params["shift_x"] = random.uniform(-shift_ratio * w, shift_ratio * w)
    if random.random() < 0.15:
        params["contrast"] = (
            random.uniform(0.95, 1.05)
            if profile == "light"
            else random.uniform(0.90, 1.10)
        )
    if random.random() < 0.10:
        params["bias"] = (
            random.uniform(-0.08, 0.08)
            if profile == "light"
            else random.uniform(-0.15, 0.15)
        )
    if random.random() < 0.15:
        params["noise_std"] = (
            random.uniform(0.008, 0.02)
            if profile == "light"
            else random.uniform(0.01, 0.05)
        )
    if profile == "strong" and random.random() < 0.25:
        params["flip"] = 1.0
    return params


def _clip_normalized_range(x: np.ndarray, normalize_mode: str) -> np.ndarray:
    mode = str(normalize_mode).lower()
    if mode == "minmax":
        return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    if mode == "zscore":
        return np.clip(x, -5.0, 5.0).astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)


def _apply_aug_to_stack(
    x: np.ndarray, aug_params: Dict[str, float], normalize_mode: str
) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    angle = float(aug_params.get("rotate", 0.0))
    if abs(angle) > 1e-8:
        for i in range(x.shape[0]):
            x[i] = ndimage.rotate(
                x[i], angle=angle, reshape=False, order=1, mode="nearest"
            )
    shift_y = float(aug_params.get("shift_y", 0.0))
    shift_x = float(aug_params.get("shift_x", 0.0))
    if abs(shift_y) > 1e-8 or abs(shift_x) > 1e-8:
        for i in range(x.shape[0]):
            x[i] = ndimage.shift(
                x[i], shift=(shift_y, shift_x), order=1, mode="nearest"
            )
    contrast = float(aug_params.get("contrast", 1.0))
    if abs(contrast - 1.0) > 1e-8:
        x_mean = x.mean()
        x = x_mean + contrast * (x - x_mean)
    bias = float(aug_params.get("bias", 0.0))
    if abs(bias) > 1e-8:
        x = x + bias
    noise_std = float(aug_params.get("noise_std", 0.0))
    if noise_std > 1e-8:
        x = x + np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    if float(aug_params.get("flip", 0.0)) > 0.5:
        x = np.flip(x, axis=2).copy()
    return _clip_normalized_range(x, normalize_mode=normalize_mode)


def _apply_bag_aug(
    bag: np.ndarray, normalize_mode: str, aug_params: Dict[str, float]
) -> np.ndarray:
    out = []
    for i in range(bag.shape[0]):
        out.append(
            _apply_aug_to_stack(
                bag[i], aug_params=aug_params, normalize_mode=normalize_mode
            )
        )
    return np.stack(out, axis=0).astype(np.float32, copy=False)
