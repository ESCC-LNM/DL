"""Shared optimization and inference implementation for this model only."""

import argparse
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .model import build_model

log = logging.getLogger(__name__)


def build_optimizer_param_groups(
    model: nn.Module,
    base_weight_decay: float,
    backbone_lr_scale: float,
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, bool, float], List[nn.Parameter]] = {}
    group_numels: Dict[Tuple[str, bool, float], int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        block = "backbone" if name.startswith("image_backbone.") else "head"
        lr_scale = float(backbone_lr_scale) if block == "backbone" else 1.0
        no_decay = (
            (param.ndim <= 1)
            or name.endswith(".bias")
            or ("norm" in name.lower())
            or ("bn" in name.lower())
        )
        key = (block, no_decay, lr_scale)
        grouped.setdefault(key, []).append(param)
        group_numels[key] = group_numels.get(key, 0) + int(param.numel())

    param_groups: List[Dict] = []
    summaries: List[Dict[str, object]] = []
    for (block, no_decay, lr_scale), params in grouped.items():
        wd = 0.0 if no_decay else float(base_weight_decay)
        param_groups.append(
            {
                "params": params,
                "lr_scale": lr_scale,
                "group_name": f"{block}_{'nodecay' if no_decay else 'decay'}",
                "weight_decay": wd,
            }
        )
        summaries.append(
            {
                "block": block,
                "no_decay": no_decay,
                "lr_scale": lr_scale,
                "numel": group_numels[(block, no_decay, lr_scale)],
                "weight_decay": wd,
            }
        )
    return param_groups, summaries


class Trainer:
    def __init__(
        self, args: argparse.Namespace, device: torch.device, radiomics_dim: int
    ):
        self.args = args
        self.device = device
        self.model = build_model(
            radiomics_dim=int(radiomics_dim),
            num_classes=args.num_classes,
            rad_hidden_dim_1=args.rad_hidden_dim_1,
            rad_hidden_dim_2=args.rad_hidden_dim_2,
            prior_dim=args.prior_dim,
            num_prior_groups=args.num_prior_groups,
            image_dropout=args.image_dropout,
            prior_dropout=args.prior_dropout,
            shared_dim=args.shared_dim,
            group_temperature=args.group_temperature,
            prior_confidence_power=args.prior_confidence_power,
            mil_attn_dim=args.mil_attn_dim,
            mil_dropout=args.mil_dropout,
            tumor_mil_topk=args.tumor_mil_topk,
            context_mil_topk=args.context_mil_topk,
            tumor_mil_temperature=args.tumor_mil_temperature,
            context_mil_temperature=args.context_mil_temperature,
            mil_prior_conditioning=args.mil_prior_conditioning,
            pretrained=args.pretrained,
            local_pretrained_ckpt=args.pretrained_ckpt,
            freeze_image_backbone=False,
        ).to(device)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(args.label_smoothing)
        )
        param_groups, summaries = build_optimizer_param_groups(
            self.model, args.weight_decay, args.backbone_lr_scale
        )
        self.optimizer = AdamW(
            param_groups if param_groups else self.model.parameters(),
            lr=args.lr,
            weight_decay=0.0,
        )
        self.use_amp = bool(args.amp and device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if int(args.freeze_backbone_epochs) > 0:
            self.model.freeze_image_backbone()
            self.backbone_frozen = True
        else:
            self.backbone_frozen = False

        self.best_state: Optional[Dict[str, object]] = None
        self.best_val_auc = -1.0
        self.best_selection_score = -1.0
        self.best_epoch = -1
        self.patience_counter = 0
        self.current_base_lr = float(args.lr)
        log.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        log.info(
            f"LR groups: backbone={float(args.lr) * float(args.backbone_lr_scale):.2e}, head={float(args.lr):.2e}"
        )
        for group_meta in summaries:
            log.info(
                "Optimizer group: "
                f"{group_meta['block']} | no_decay={group_meta['no_decay']} | lr_scale={group_meta['lr_scale']:.2f} | "
                f"wd={group_meta['weight_decay']:.1e} | params={int(group_meta['numel']):,}"
            )
        log.info("Train class weighting: disabled")
        log.info(
            f"Label smoothing={float(args.label_smoothing):.3f} | "
            f"freeze_backbone_epochs={int(args.freeze_backbone_epochs)} | "
            f"group_temperature={float(args.group_temperature):.2f} | "
            f"prior_conf_power={float(args.prior_confidence_power):.2f} | "
            f"radiomics_standardization=train_scaler_only | "
            f"eval_tta={str(args.eval_tta_mode)} | ensemble={str(args.ensemble_mode)} | "
            f"ensemble_topk={int(args.ensemble_topk)}"
        )

    def _autocast_context(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            )
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    def _keep_backbone_eval_when_frozen(self) -> None:
        if self.backbone_frozen:
            self.model.image_backbone.eval()

    def _ensure_finite(
        self,
        tensor: torch.Tensor,
        name: str,
        epoch: int,
        batch_idx: int,
        batch_ids: Optional[List[str]] = None,
    ) -> None:
        if torch.isfinite(tensor).all():
            return
        sample_ids = batch_ids[:4] if batch_ids else []
        raise FloatingPointError(
            f"Non-finite tensor detected: {name} at epoch={epoch}, batch={batch_idx}, sample_ids={sample_ids}"
        )

    def _build_eval_views(
        self,
        tumor_bag: torch.Tensor,
        context_bag: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        mode = str(getattr(self.args, "eval_tta_mode", "none")).lower()
        views: List[Tuple[torch.Tensor, torch.Tensor]] = [(tumor_bag, context_bag)]
        if mode in {"hflip", "flip2"}:
            views.append(
                (torch.flip(tumor_bag, dims=[-1]), torch.flip(context_bag, dims=[-1]))
            )
        if mode == "flip2":
            views.append(
                (torch.flip(tumor_bag, dims=[-2]), torch.flip(context_bag, dims=[-2]))
            )
        return views

    @torch.no_grad()
    def _forward_eval(
        self,
        tumor_bag: torch.Tensor,
        context_bag: torch.Tensor,
        rad_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs: List[Dict[str, torch.Tensor]] = []
        for tumor_view, context_view in self._build_eval_views(tumor_bag, context_bag):
            with self._autocast_context():
                outputs.append(self.model(tumor_view, context_view, rad_features))
        if len(outputs) == 1:
            return outputs[0]

        merged: Dict[str, torch.Tensor] = {}
        for key, value in outputs[0].items():
            if torch.is_tensor(value) and value.is_floating_point():
                merged[key] = torch.stack([out[key] for out in outputs], dim=0).mean(
                    dim=0
                )
            else:
                merged[key] = value
        return merged

    def _set_epoch_lr(self, epoch: int) -> None:
        base_lr = float(self.args.lr)
        min_lr = float(self.args.min_lr)
        warmup_epochs = int(self.args.warmup_epochs)
        total_epochs = int(self.args.epochs)
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            lr = base_lr * float(epoch) / float(warmup_epochs)
        else:
            if total_epochs <= warmup_epochs:
                lr = base_lr
            else:
                progress = (epoch - warmup_epochs) / float(
                    max(1, total_epochs - warmup_epochs)
                )
                progress = min(max(progress, 0.0), 1.0)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
        self.current_base_lr = lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr * float(pg.get("lr_scale", 1.0))

    def maybe_unfreeze_backbone(self, epoch: int) -> None:
        if self.backbone_frozen and epoch > int(self.args.freeze_backbone_epochs):
            self.model.unfreeze_image_backbone()
            self.backbone_frozen = False
            log.info(f"Epoch {epoch}: Image backbone unfrozen.")

            self.optimizer.zero_grad(set_to_none=True)

    def _compute_losses(
        self, out: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        ce = self.criterion(out["logits"], labels)
        align = (
            1.0
            - F.cosine_similarity(
                F.normalize(out["image_proj"], dim=1),
                F.normalize(out["prior"], dim=1),
                dim=1,
            ).mean()
        )
        group_ent = out["group_entropy"].mean()
        total = (
            ce
            + float(self.args.align_lambda) * align
            + float(self.args.group_entropy_lambda) * group_ent
        )
        return {
            "loss": total,
            "ce": ce,
            "align": align,
            "group_ent": group_ent,
        }

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self._keep_backbone_eval_when_frozen()
        self._set_epoch_lr(epoch)
        self.maybe_unfreeze_backbone(epoch)
        self._keep_backbone_eval_when_frozen()

        total = {
            k: 0.0
            for k in [
                "loss",
                "ce",
                "align",
                "group_ent",
                "t_prompt",
                "c_prompt",
                "t_film",
                "c_film",
                "prior_conf",
                "tumor_prior_mil_strength",
                "context_prior_mil_strength",
            ]
        }
        n = 0
        for batch_idx, batch in enumerate(train_loader):
            tumor_bag = batch["tumor_bag"].to(self.device, non_blocking=True)
            context_bag = batch["context_bag"].to(self.device, non_blocking=True)
            rad_features = batch["rad_features"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            batch_ids = [str(x) for x in batch["id"]]

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                out = self.model(tumor_bag, context_bag, rad_features)
                losses = self._compute_losses(out, labels)

            self._ensure_finite(
                out["logits"],
                "logits",
                epoch=epoch,
                batch_idx=batch_idx,
                batch_ids=batch_ids,
            )
            self._ensure_finite(
                out["prior"],
                "prior",
                epoch=epoch,
                batch_idx=batch_idx,
                batch_ids=batch_ids,
            )
            self._ensure_finite(
                out["image_feat"],
                "image_feat",
                epoch=epoch,
                batch_idx=batch_idx,
                batch_ids=batch_ids,
            )
            self._ensure_finite(
                losses["loss"],
                "loss",
                epoch=epoch,
                batch_idx=batch_idx,
                batch_ids=batch_ids,
            )

            self.scaler.scale(losses["loss"]).backward()
            if self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float(self.args.grad_clip)
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in ["loss", "ce", "align", "group_ent"]:
                total[k] += float(losses[k].item())
            total["t_prompt"] += float(out["t_prompt"].mean().item())
            total["c_prompt"] += float(out["c_prompt"].mean().item())
            total["t_film"] += float(out["t_film"].mean().item())
            total["c_film"] += float(out["c_film"].mean().item())
            total["prior_conf"] += float(out["prior_confidence"].mean().item())
            total["tumor_prior_mil_strength"] += float(
                out["tumor_prior_mil_strength"].mean().item()
            )
            total["context_prior_mil_strength"] += float(
                out["context_prior_mil_strength"].mean().item()
            )
            n += 1

        total = {k: v / max(1, n) for k, v in total.items()}
        total["lr"] = self.current_base_lr
        return total

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        self.model.eval()
        all_ids, all_patient_ids, all_labels, all_probs = [], [], [], []
        stats = {
            k: []
            for k in [
                "group_ent",
                "group_max",
                "prior_conf",
                "t_prompt",
                "c_prompt",
                "t_film",
                "c_film",
                "tumor_peak",
                "context_peak",
                "tumor_prior_mil_strength",
                "context_prior_mil_strength",
            ]
        }
        for batch in loader:
            tumor_bag = batch["tumor_bag"].to(self.device, non_blocking=True)
            context_bag = batch["context_bag"].to(self.device, non_blocking=True)
            rad_features = batch["rad_features"].to(self.device, non_blocking=True)
            logits_out = self._forward_eval(tumor_bag, context_bag, rad_features)
            batch_ids = [str(x) for x in batch["id"]]
            if not torch.isfinite(logits_out["logits"]).all():
                raise FloatingPointError(
                    f"Non-finite logits during eval, sample_ids={batch_ids[:4]}"
                )
            probs = (
                torch.softmax(logits_out["logits"], dim=1)[:, 1].detach().cpu().numpy()
            )
            if not np.isfinite(probs).all():
                raise FloatingPointError(
                    f"Non-finite probabilities during eval, sample_ids={batch_ids[:4]}"
                )
            labels = batch["label"].cpu().numpy()
            ids = [str(x) for x in batch["id"]]
            patient_ids = [str(x) for x in batch["patientID"]]

            all_ids.extend(ids)
            all_patient_ids.extend(patient_ids)
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

            stats["group_ent"].extend(
                logits_out["group_entropy"]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )
            stats["group_max"].extend(
                logits_out["group_weights"]
                .max(dim=1)
                .values.detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )
            stats["prior_conf"].extend(
                logits_out["prior_confidence"]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )
            stats["t_prompt"].extend(
                logits_out["t_prompt"].detach().cpu().numpy().astype(float).tolist()
            )
            stats["c_prompt"].extend(
                logits_out["c_prompt"].detach().cpu().numpy().astype(float).tolist()
            )
            stats["t_film"].extend(
                logits_out["t_film"].detach().cpu().numpy().astype(float).tolist()
            )
            stats["c_film"].extend(
                logits_out["c_film"].detach().cpu().numpy().astype(float).tolist()
            )
            stats["tumor_peak"].extend(
                logits_out["tumor_peak"].detach().cpu().numpy().astype(float).tolist()
            )
            stats["context_peak"].extend(
                logits_out["context_peak"].detach().cpu().numpy().astype(float).tolist()
            )
            stats["tumor_prior_mil_strength"].extend(
                logits_out["tumor_prior_mil_strength"]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )
            stats["context_prior_mil_strength"].extend(
                logits_out["context_prior_mil_strength"]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )

        pred = {
            "ids": np.asarray(all_ids),
            "patient_ids": np.asarray(all_patient_ids),
            "labels": np.asarray(all_labels, dtype=int),
            "probs": np.asarray(all_probs, dtype=float),
        }
        for k, v in stats.items():
            pred[k] = np.asarray(v, dtype=float)
        return pred
