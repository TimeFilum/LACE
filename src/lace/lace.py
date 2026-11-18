"""
LACE model composition.

Architecture
------------
    x
      |
      v
    Frozen TSFM backbone
      |
      | backbone-specific latent extraction
      v
    z0 [B, D]
      |
      | shared representation normalization
      v
    z [B, D]
      |
      +----------------+----------------+ ...
      |                |
      v                v
     P_r              P_s
      |                |
      v                v
     W_r              W_s
      |                |
      v                v
    y_hat_r          y_hat_s

For each resolution r:

    y_hat_r = W_r P_r z + b_r

with effective mapping:

    M_r = W_r P_r

This file performs no loss calculation and no hierarchy aggregation.
"""

from typing import Dict, List

import torch
from torch import nn

from lace.adapter import ResolutionAdapter
from lace.head import ForecastHead
from lace.hierarchy import TemporalHierarchy
from lace.lace_config import LACEConfig


class LACE(nn.Module):
    """LACE joint multi-resolution forecasting model."""

    def __init__(
        self,
        context_len: int,
        pred_len: int,
        factors: List[int],
        adapter_dim: int = 128,
        representation_norm: str = "batch",
        reduction: str = "sum",
        backbone_name: str = "Salesforce/moirai-1.0-R-small",
    ):
        super().__init__()

        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.adapter_dim = int(adapter_dim)

        # ==========================================================
        # 1. Frozen TSFM representation
        # ==========================================================
        backbone_key = backbone_name.lower()

        if "timesfm-2" in backbone_key or "timesfm2" in backbone_key:
            from lace.backbone_timesfm2 import FrozenTimesFM2Backbone

            self.backbone = FrozenTimesFM2Backbone(
                context_len=self.context_len,
                pred_len=self.pred_len,
                model_name=backbone_name,
            )

        elif "moirai-2" in backbone_key or "moirai2" in backbone_key:
            from lace.backbone_moirai2 import FrozenMoirai2Backbone

            self.backbone = FrozenMoirai2Backbone(
                context_len=self.context_len,
                pred_len=self.pred_len,
                model_name=backbone_name,
            )

        elif "moirai-1" in backbone_key or "moirai1" in backbone_key:
            from lace.backbone import FrozenMoiraiBackbone

            self.backbone = FrozenMoiraiBackbone(
                context_len=self.context_len,
                pred_len=self.pred_len,
                model_name=backbone_name,
            )

        elif "chronos-2" in backbone_key or "chronos2" in backbone_key:
            from lace.backbone_chronos2 import FrozenChronos2Backbone

            self.backbone = FrozenChronos2Backbone(
                context_len=self.context_len,
                pred_len=self.pred_len,
                model_name=backbone_name,
            )
        else:
            raise ValueError(
                "Unsupported LACE backbone: "
                f"{backbone_name!r}. "
                "Currently supported: Moirai-1.0, Moirai-2.0, TimesFM-2.0, Chronos-2.0."
            )

        self.backbone_name = backbone_name
        self.latent_dim = self.backbone.latent_dim

        # ==========================================================
        # 2. Dynamic temporal hierarchy
        # ==========================================================
        self.hierarchy = TemporalHierarchy(
            factors=factors,
            pred_len=self.pred_len,
            reduction=reduction,
        )

        self.factors = self.hierarchy.factors
        self.target_lengths = self.hierarchy.target_lengths

        for factor, length in zip(self.factors, self.target_lengths):
            if length <= 0:
                raise ValueError(
                    f"Factor {factor} produces zero forecast length "
                    f"for pred_len={self.pred_len}."
                )

        # ==========================================================
        # 3. Shared representation normalization
        # ==========================================================
        representation_norm = representation_norm.lower()

        if representation_norm == "batch":
            # affine=False keeps the explicit branch mapping as W_r P_r.
            self.representation_norm = nn.BatchNorm1d(
                self.latent_dim,
                affine=False,
            )
        elif representation_norm == "layer":
            self.representation_norm = nn.LayerNorm(
                self.latent_dim,
                elementwise_affine=False,
            )
        elif representation_norm == "none":
            self.representation_norm = nn.Identity()
        else:
            raise ValueError(
                "representation_norm must be 'batch', 'layer', or 'none', "
                f"got {representation_norm!r}."
            )

        self.representation_norm_type = representation_norm

        # ==========================================================
        # 4. Resolution-specific adapters P_r
        # ==========================================================
        self.adapters = nn.ModuleDict({
            str(factor): ResolutionAdapter(
                input_dim=self.latent_dim,
                adapter_dim=self.adapter_dim,
            )
            for factor in self.factors
        })

        # ==========================================================
        # 5. Resolution-specific forecast heads W_r, b_r
        # ==========================================================
        self.heads = nn.ModuleDict({
            str(factor): ForecastHead(
                adapter_dim=self.adapter_dim,
                output_len=output_len,
            )
            for factor, output_len in zip(
                self.factors,
                self.target_lengths,
            )
        })

    # ==============================================================
    # Shared representation
    # ==============================================================

    def encode_pre_norm(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Frozen TSFM -> pooled shared representation.

        Parameters
        ----------
        x_enc : Tensor [B, context_len, 1]

        Returns
        -------
        z0 : Tensor [B, latent_dim]
            Frozen backbone representation used by all
            resolution-specific LACE branches.
        """
        return self.backbone(x_enc)

    def normalize_representation(
        self,
        z0: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured shared representation normalization."""
        if z0.ndim != 2:
            raise ValueError(
                f"Expected representation [B,D], got {z0.shape}."
            )

        if z0.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"got {z0.shape[-1]}."
            )

        return self.representation_norm(z0)

    def encode(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Encode a raw history into the normalized shared representation."""
        z0 = self.encode_pre_norm(x_enc)
        return self.normalize_representation(z0)

    # ==============================================================
    # Forward from a precomputed representation
    # ==============================================================

    def forward_from_representation(
        self,
        z0: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Forecast all resolutions from a pooled pre-normalization
        representation z0 [B, latent_dim].

        This is a convenience path for diagnostics or optional future
        precomputation; normal training uses forward().
        """
        z = self.normalize_representation(z0)

        return {
            factor: self.forecast_scale(z, factor)
            for factor in self.factors
        }

    # ==============================================================
    # Standard full forward
    # ==============================================================

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ) -> Dict[int, torch.Tensor]:
        """Forecast all configured resolutions from the raw input history."""
        z0 = self.encode_pre_norm(x_enc)
        return self.forward_from_representation(z0)

    # ==============================================================
    # Resolution branch
    # ==============================================================

    def forecast_scale(
        self,
        z: torch.Tensor,
        factor: int,
    ) -> torch.Tensor:
        """Forecast one resolution: z -> P_r -> W_r -> y_hat_r."""
        factor = int(factor)

        if factor not in self.factors:
            raise ValueError(f"Unknown hierarchy factor {factor}.")

        key = str(factor)
        z_r = self.adapters[key](z)
        return self.heads[key](z_r)

    # ==============================================================
    # Explicit structural mappings
    # ==============================================================

    def effective_mappings(self) -> Dict[int, torch.Tensor]:
        """
        Return M_r = W_r P_r for every resolution.

        The returned tensors remain attached to the computation graph so
        structural coherence regularization updates both P_r and W_r.
        """
        mappings = {}

        for factor in self.factors:
            key = str(factor)
            P_r = self.adapters[key].weight
            W_r = self.heads[key].weight
            mappings[factor] = W_r @ P_r

        return mappings

    def effective_biases(self) -> Dict[int, torch.Tensor]:
        """Return b_r for every resolution."""
        return {
            factor: self.heads[str(factor)].bias
            for factor in self.factors
        }


class Model(LACE):
    """THUML-compatible entry point for LACE."""

    def __init__(self, configs):
        cfg = LACEConfig.from_namespace(configs)

        super().__init__(
            context_len=cfg.context_len,
            pred_len=cfg.pred_len,
            factors=list(cfg.factors),
            adapter_dim=cfg.adapter_dim,
            representation_norm=cfg.representation_norm,
            reduction=cfg.reduction,
            backbone_name=cfg.backbone_name,
        )
