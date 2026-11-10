"""Model definition; state-dict keys match the research implementation."""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class TopKSparseGatedAttentionMIL(nn.Module):
    """Sparse MIL with optional radiomics-prior-conditioned attention.

    The pooled values remain image instance features. The radiomics prior only
    changes which 2.5D instances receive attention, so this is not late fusion.
    """

    def __init__(
        self,
        input_dim: int,
        attn_dim: int = 128,
        dropout: float = 0.0,
        topk: int = 3,
        temperature: float = 1.0,
        prior_dim: Optional[int] = None,
        use_prior_conditioning: bool = True,
    ):
        super().__init__()
        self.attn_v = nn.Linear(input_dim, attn_dim)
        self.attn_u = nn.Linear(input_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.use_prior_conditioning = bool(use_prior_conditioning)
        if self.use_prior_conditioning:
            if prior_dim is None or int(prior_dim) <= 0:
                raise ValueError(
                    "prior_dim must be positive when prior conditioning is enabled"
                )
            self.prior_proj = nn.Linear(int(prior_dim), int(attn_dim), bias=False)
        else:
            self.prior_proj = None
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.topk = int(topk)
        self.temperature = float(max(temperature, 1e-3))
        self._init_weights()
        if self.prior_proj is not None:
            nn.init.zeros_(self.prior_proj.weight)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _resolve_k(self, n_instances: int) -> int:
        if self.topk <= 0:
            return int(n_instances)
        return int(max(1, min(n_instances, self.topk)))

    def forward(
        self,
        x: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(
                f"MIL expects [B,N,D] image features, got {tuple(x.shape)}"
            )

        v_logits = self.attn_v(x)
        u_logits = self.attn_u(x)
        conditioning_strength = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if self.use_prior_conditioning:
            if prior is None:
                raise ValueError("Radiomics prior is required by prior-conditioned MIL")
            if prior.ndim != 2 or prior.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Prior must be [B,P] with matching batch; got x={tuple(x.shape)}, prior={tuple(prior.shape)}"
                )
            prior_bias = self.prior_proj(prior).unsqueeze(1)
            v_logits = v_logits + prior_bias
            u_logits = u_logits + prior_bias
            conditioning_strength = prior_bias.squeeze(1).abs().mean(dim=1)

        a = torch.tanh(v_logits) * torch.sigmoid(u_logits)
        a = self.dropout(a)
        logits = self.attn_w(a).squeeze(-1) / self.temperature
        dense_weights = torch.softmax(logits, dim=1)

        n_instances = int(x.shape[1])
        k = self._resolve_k(n_instances)
        topk_vals, topk_idx = torch.topk(
            dense_weights, k=k, dim=1, largest=True, sorted=True
        )
        sparse_weights = torch.zeros_like(dense_weights)
        sparse_weights.scatter_(1, topk_idx, topk_vals)
        sparse_weights = sparse_weights / sparse_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

        pooled = torch.sum(sparse_weights.unsqueeze(-1) * x, dim=1)
        stats = {
            "dense_weights": dense_weights,
            "sparse_weights": sparse_weights,
            "topk_mass": topk_vals.sum(dim=1),
            "selected_frac": torch.full(
                (x.shape[0],),
                float(k) / float(n_instances),
                device=x.device,
                dtype=x.dtype,
            ),
            "prior_conditioning_strength": conditioning_strength,
        }
        return pooled, sparse_weights, stats


class RadiomicsPriorEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        prior_dim: int = 128,
        num_groups: int = 4,
        dropout: float = 0.2,
        group_temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim_1 = int(hidden_dim_1)
        self.hidden_dim_2 = int(hidden_dim_2)
        self.prior_dim = int(prior_dim)
        self.num_groups = int(num_groups)
        self.group_temperature = float(max(group_temperature, 1e-3))
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )
        self.group_logits = nn.Linear(self.hidden_dim_2, self.num_groups)
        self.group_tokens = nn.Linear(
            self.hidden_dim_2, self.num_groups * self.prior_dim
        )
        self.group_prototypes = nn.Parameter(
            torch.randn(self.num_groups, self.prior_dim) * 0.02
        )
        self.prior_refine = nn.Sequential(
            nn.LayerNorm(self.prior_dim),
            nn.Linear(self.prior_dim, self.prior_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x.float())
        logits = self.group_logits(h)
        weights = torch.softmax(logits / self.group_temperature, dim=1)
        token_delta = self.group_tokens(h).view(-1, self.num_groups, self.prior_dim)
        tokens = torch.tanh(token_delta + self.group_prototypes.unsqueeze(0))
        prior = torch.sum(weights.unsqueeze(-1) * tokens, dim=1)
        prior = self.prior_refine(prior) + prior
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1)
        entropy_max = math.log(max(self.num_groups, 2))
        confidence = (1.0 - entropy / max(entropy_max, 1e-6)).clamp(0.0, 1.0)
        return {
            "prior": prior,
            "group_weights": weights,
            "group_entropy": entropy,
            "group_confidence": confidence,
            "group_tokens": tokens,
        }


class OmicsPriorPromptBlock(nn.Module):
    def __init__(
        self,
        prior_dim: int,
        feat_dim: int,
        prompt_dropout: float = 0.1,
        confidence_power: float = 1.0,
    ):
        super().__init__()
        self.confidence_power = float(max(confidence_power, 1.0))
        self.prompt = nn.Linear(int(prior_dim), int(feat_dim))
        self.gamma = nn.Linear(int(prior_dim), int(feat_dim))
        self.beta = nn.Linear(int(prior_dim), int(feat_dim))
        self.prompt_scale = nn.Sequential(nn.Linear(int(prior_dim), 1), nn.Sigmoid())
        self.film_scale = nn.Sequential(nn.Linear(int(prior_dim), 1), nn.Sigmoid())
        self.dropout = (
            nn.Dropout(float(prompt_dropout)) if prompt_dropout > 0 else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        feats: torch.Tensor,
        prior: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if confidence is None:
            confidence = torch.ones(
                (feats.shape[0], 1), device=feats.device, dtype=feats.dtype
            )
        elif confidence.ndim == 1:
            confidence = confidence.unsqueeze(1)
        confidence = confidence.clamp(0.0, 1.0)
        confidence_strength = confidence.pow(self.confidence_power)
        prompt = self.prompt(prior)
        gamma_delta = 0.25 * torch.tanh(self.gamma(prior))
        beta = 0.10 * torch.tanh(self.beta(prior))
        p_scale = 0.20 * self.prompt_scale(prior) * confidence_strength
        f_scale = 0.20 * self.film_scale(prior) * confidence_strength

        prompt_term = p_scale.unsqueeze(1) * prompt.unsqueeze(1)
        film_term = f_scale.unsqueeze(1) * (
            gamma_delta.unsqueeze(1) * feats + beta.unsqueeze(1)
        )
        out = feats + self.dropout(prompt_term + film_term)
        stats = {
            "prompt_scale": p_scale.squeeze(1),
            "film_scale": f_scale.squeeze(1),
            "gamma_abs": gamma_delta.abs().mean(dim=1),
            "beta_abs": beta.abs().mean(dim=1),
        }
        return out, stats


class ConvNeXtV2OmicsPrior(nn.Module):
    def __init__(
        self,
        radiomics_dim: int,
        num_classes: int = 2,
        rad_hidden_dim_1: int = 128,
        rad_hidden_dim_2: int = 64,
        prior_dim: int = 128,
        num_prior_groups: int = 4,
        image_dropout: float = 0.2,
        prior_dropout: float = 0.2,
        shared_dim: int = 128,
        group_temperature: float = 0.85,
        prior_confidence_power: float = 1.25,
        mil_attn_dim: int = 128,
        mil_dropout: float = 0.1,
        tumor_mil_topk: int = 2,
        context_mil_topk: int = 4,
        tumor_mil_temperature: float = 0.70,
        context_mil_temperature: float = 0.90,
        mil_prior_conditioning: bool = True,
        pretrained: bool = True,
        local_pretrained_ckpt: Optional[str] = None,
        freeze_image_backbone: bool = False,
    ):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError(
                "Install timm to construct the ConvNeXt V2 backbone."
            ) from e

        self.model_name = "convnextv2_atto.fcmae_ft_in1k"
        self.pretrained = bool(pretrained)
        if self.pretrained and local_pretrained_ckpt is None:
            raise ValueError("pretrained=True requires a local ImageNet checkpoint.")

        self.image_backbone = timm.create_model(
            self.model_name, pretrained=False, num_classes=0, global_pool=""
        )
        self.image_feature_dim = int(getattr(self.image_backbone, "num_features", 168))
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.instance_post = nn.Sequential(
            nn.LayerNorm(self.image_feature_dim), nn.Dropout(p=float(image_dropout))
        )

        self.prior_encoder = RadiomicsPriorEncoder(
            input_dim=int(radiomics_dim),
            hidden_dim_1=int(rad_hidden_dim_1),
            hidden_dim_2=int(rad_hidden_dim_2),
            prior_dim=int(prior_dim),
            num_groups=int(num_prior_groups),
            dropout=float(prior_dropout),
            group_temperature=float(group_temperature),
        )
        self.tumor_prompt = OmicsPriorPromptBlock(
            int(prior_dim),
            self.image_feature_dim,
            prompt_dropout=float(prior_dropout),
            confidence_power=float(prior_confidence_power),
        )
        self.context_prompt = OmicsPriorPromptBlock(
            int(prior_dim),
            self.image_feature_dim,
            prompt_dropout=float(prior_dropout),
            confidence_power=float(prior_confidence_power),
        )

        self.mil_prior_conditioning = bool(mil_prior_conditioning)
        self.tumor_mil = TopKSparseGatedAttentionMIL(
            self.image_feature_dim,
            int(mil_attn_dim),
            float(mil_dropout),
            int(tumor_mil_topk),
            float(tumor_mil_temperature),
            prior_dim=int(prior_dim),
            use_prior_conditioning=self.mil_prior_conditioning,
        )
        self.context_mil = TopKSparseGatedAttentionMIL(
            self.image_feature_dim,
            int(mil_attn_dim),
            float(mil_dropout),
            int(context_mil_topk),
            float(context_mil_temperature),
            prior_dim=int(prior_dim),
            use_prior_conditioning=self.mil_prior_conditioning,
        )

        self.image_context_fuser = nn.Sequential(
            nn.LayerNorm(self.image_feature_dim * 2),
            nn.Dropout(p=float(image_dropout)),
            nn.Linear(self.image_feature_dim * 2, int(shared_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(image_dropout)),
        )
        self.image_proj = nn.Sequential(
            nn.LayerNorm(int(shared_dim)), nn.Linear(int(shared_dim), int(prior_dim))
        )
        self.image_aux_head = nn.Sequential(
            nn.LayerNorm(int(shared_dim)),
            nn.Dropout(p=float(image_dropout)),
            nn.Linear(int(shared_dim), int(num_classes)),
        )
        self._init_head()

        self.loaded_local_pretrain = False
        self.local_missing_keys = []
        self.local_unexpected_keys = []
        if self.pretrained:
            self._load_local_pretrain(local_pretrained_ckpt)
        if freeze_image_backbone:
            self.freeze_image_backbone()

    def _init_head(self) -> None:
        for mod in [self.image_context_fuser, self.image_proj, self.image_aux_head]:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def _clean_state_dict_key(self, key: str) -> str:
        out = key
        for p in ["module.", "model.", "backbone."]:
            if out.startswith(p):
                out = out[len(p) :]
        return out

    def _load_local_pretrain(self, local_pretrained_ckpt: Optional[str]) -> None:
        ckpt_path = Path(str(local_pretrained_ckpt))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        filtered_state = {}
        for k, v in state_dict.items():
            k = self._clean_state_dict_key(k)
            kl = k.lower()
            if (
                kl.startswith("head")
                or kl.startswith("classifier")
                or kl.startswith("fc")
            ):
                continue
            if ".head" in kl or ".classifier" in kl or ".fc" in kl:
                continue
            filtered_state[k] = v
        incompatible = self.image_backbone.load_state_dict(filtered_state, strict=False)
        self.local_missing_keys = list(getattr(incompatible, "missing_keys", []))
        self.local_unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
        self.loaded_local_pretrain = True

    def freeze_image_backbone(self) -> None:
        for p in self.image_backbone.parameters():
            p.requires_grad = False

    def unfreeze_image_backbone(self) -> None:
        for p in self.image_backbone.parameters():
            p.requires_grad = True

    def _extract_instance_features(self, bag: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = bag.shape
        x = bag.reshape(b * n, c, h, w)
        if hasattr(self.image_backbone, "forward_features"):
            feats = self.image_backbone.forward_features(x)
        else:
            feats = self.image_backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if feats.ndim != 4:
            raise ValueError(
                f"Unexpected ConvNeXt V2 feature shape: {tuple(feats.shape)}"
            )
        feats = torch.nan_to_num(feats, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
            -1e4, 1e4
        )
        feats = self.image_pool(feats).flatten(1)
        feats = self.instance_post(feats)
        feats = feats.view(b, n, -1)
        return feats

    def forward(
        self,
        tumor_bag: torch.Tensor,
        context_bag: torch.Tensor,
        rad_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        prior_out = self.prior_encoder(rad_features)
        prior = prior_out["prior"]
        group_weights = prior_out["group_weights"]
        group_entropy = prior_out["group_entropy"]
        prior_confidence = prior_out["group_confidence"]

        tumor_feats = self._extract_instance_features(tumor_bag)
        context_feats = self._extract_instance_features(context_bag)

        tumor_feats, tumor_prompt_stats = self.tumor_prompt(
            tumor_feats, prior, confidence=prior_confidence
        )
        context_feats, context_prompt_stats = self.context_prompt(
            context_feats, prior, confidence=prior_confidence
        )

        mil_prior = prior if self.mil_prior_conditioning else None
        tumor_pooled, tumor_attn, tumor_stats = self.tumor_mil(tumor_feats, mil_prior)
        context_pooled, context_attn, context_stats = self.context_mil(
            context_feats, mil_prior
        )
        image_feat = self.image_context_fuser(
            torch.cat([tumor_pooled, context_pooled], dim=1)
        )

        image_logits = self.image_aux_head(image_feat)
        image_proj = self.image_proj(image_feat)

        return {
            "logits": image_logits,
            "image_logits": image_logits,
            "image_feat": image_feat,
            "image_proj": image_proj,
            "prior": prior,
            "group_weights": group_weights,
            "group_entropy": group_entropy,
            "prior_confidence": prior_confidence,
            "tumor_attn": tumor_attn,
            "context_attn": context_attn,
            "tumor_peak": tumor_attn.max(dim=1).values,
            "context_peak": context_attn.max(dim=1).values,
            "tumor_topk_mass": tumor_stats["topk_mass"],
            "context_topk_mass": context_stats["topk_mass"],
            "tumor_selected_frac": tumor_stats["selected_frac"],
            "context_selected_frac": context_stats["selected_frac"],
            "tumor_prior_mil_strength": tumor_stats["prior_conditioning_strength"],
            "context_prior_mil_strength": context_stats["prior_conditioning_strength"],
            "tumor_entropy": -(
                tumor_attn.clamp_min(1e-8) * tumor_attn.clamp_min(1e-8).log()
            ).sum(dim=1),
            "context_entropy": -(
                context_attn.clamp_min(1e-8) * context_attn.clamp_min(1e-8).log()
            ).sum(dim=1),
            "t_prompt": tumor_prompt_stats["prompt_scale"],
            "c_prompt": context_prompt_stats["prompt_scale"],
            "t_film": tumor_prompt_stats["film_scale"],
            "c_film": context_prompt_stats["film_scale"],
        }


def build_model(
    radiomics_dim: int,
    num_classes: int = 2,
    rad_hidden_dim_1: int = 128,
    rad_hidden_dim_2: int = 64,
    prior_dim: int = 128,
    num_prior_groups: int = 4,
    image_dropout: float = 0.2,
    prior_dropout: float = 0.2,
    shared_dim: int = 128,
    group_temperature: float = 0.85,
    prior_confidence_power: float = 1.25,
    mil_attn_dim: int = 128,
    mil_dropout: float = 0.1,
    tumor_mil_topk: int = 2,
    context_mil_topk: int = 4,
    tumor_mil_temperature: float = 0.70,
    context_mil_temperature: float = 0.90,
    mil_prior_conditioning: bool = True,
    pretrained: bool = True,
    local_pretrained_ckpt: Optional[str] = None,
    freeze_image_backbone: bool = False,
) -> ConvNeXtV2OmicsPrior:
    return ConvNeXtV2OmicsPrior(
        radiomics_dim=int(radiomics_dim),
        num_classes=int(num_classes),
        rad_hidden_dim_1=int(rad_hidden_dim_1),
        rad_hidden_dim_2=int(rad_hidden_dim_2),
        prior_dim=int(prior_dim),
        num_prior_groups=int(num_prior_groups),
        image_dropout=float(image_dropout),
        prior_dropout=float(prior_dropout),
        shared_dim=int(shared_dim),
        group_temperature=float(group_temperature),
        prior_confidence_power=float(prior_confidence_power),
        mil_attn_dim=int(mil_attn_dim),
        mil_dropout=float(mil_dropout),
        tumor_mil_topk=int(tumor_mil_topk),
        context_mil_topk=int(context_mil_topk),
        tumor_mil_temperature=float(tumor_mil_temperature),
        context_mil_temperature=float(context_mil_temperature),
        mil_prior_conditioning=bool(mil_prior_conditioning),
        pretrained=bool(pretrained),
        local_pretrained_ckpt=local_pretrained_ckpt,
        freeze_image_backbone=bool(freeze_image_backbone),
    )
