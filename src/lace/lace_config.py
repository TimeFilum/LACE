"""
Configuration for LACE architecture, temporal hierarchy, and training objectives.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class LACEConfig:
    # ==============================================================
    # 1. Forecast setup
    # ==============================================================

    context_len: int = 512
    pred_len: int = 24

    # Temporal factors relative to the finest resolution.
    #
    # Example:
    #   [1, 3, 12, 24]
    #
    # means:
    #   base -> 3x -> 12x -> 24x
    factors: Tuple[int, ...] = (1, 3, 12, 24)

    # ==============================================================
    # 2. Frozen TSFM
    # ==============================================================

    backbone_name: str = "Salesforce/moirai-1.0-R-small"

    # ==============================================================
    # 3. Resolution adaptation
    # ==============================================================

    adapter_dim: int = 128

    # Shared representation normalization before branching:
    #
    #   "batch" : BatchNorm1d, default
    #   "layer" : LayerNorm
    #   "none"  : normalization ablation
    representation_norm: str = "batch"

    # ==============================================================
    # 4. Hierarchy
    # ==============================================================

    # "sum"  -> additive hierarchy
    # "mean" -> averaging hierarchy
    reduction: str = "sum"

    # ==============================================================
    # 5. Accuracy-loss weighting
    # ==============================================================

    # Optional resolution-specific weights.
    #
    # Empty dictionary means all resolutions receive weight 1.
    scale_weights: Dict[int, float] = field(
        default_factory=dict
    )

    # ==============================================================
    # 6. Coherence objective / ablation
    # ==============================================================

    # "none"
    #     accuracy only
    #
    # "structural"
    #     LACE:
    #         B M_r ~= M_s
    #         B b_r ~= b_s
    coherence_mode: str = "structural"

    # Relative weight:
    #
    # L_total = L_accuracy + lambda_coherence * L_coherence
    lambda_coherence: float = 1.0

    # Relative weight of the bias component inside structural loss.
    structural_bias_weight: float = 1.0

    # ==============================================================
    # Validation
    # ==============================================================

    def __post_init__(self):

        self.factors = tuple(
            sorted(int(f) for f in self.factors)
        )

        if self.context_len <= 0:
            raise ValueError(
                "context_len must be positive."
            )

        if self.pred_len <= 0:
            raise ValueError(
                "pred_len must be positive."
            )

        if self.adapter_dim <= 0:
            raise ValueError(
                "adapter_dim must be positive."
            )

        if not self.factors:
            raise ValueError(
                "At least one hierarchy factor is required."
            )

        if any(f <= 0 for f in self.factors):
            raise ValueError(
                f"Invalid hierarchy factors: {self.factors}."
            )

        if len(set(self.factors)) != len(self.factors):
            raise ValueError(
                f"Duplicate hierarchy factors: {self.factors}."
            )

        if self.representation_norm not in {
            "batch",
            "layer",
            "none",
        }:
            raise ValueError(
                "representation_norm must be "
                "'batch', 'layer', or 'none'."
            )

        if self.reduction not in {
            "sum",
            "mean",
        }:
            raise ValueError(
                "reduction must be 'sum' or 'mean'."
            )

        if self.coherence_mode not in {
            "none",
            "structural",
        }:
            raise ValueError(
                "coherence_mode must be "
                "'none' or 'structural'."
            )

        if self.lambda_coherence < 0:
            raise ValueError(
                "lambda_coherence cannot be negative."
            )

        if self.structural_bias_weight < 0:
            raise ValueError(
                "structural_bias_weight cannot be negative."
            )

    # ==============================================================
    # Helpers
    # ==============================================================

    def weight_for(self, factor: int) -> float:
        """
        Accuracy-loss weight for one resolution.

        Defaults to 1.0 when no explicit weight was provided.
        """
        return float(
            self.scale_weights.get(
                int(factor),
                1.0,
            )
        )

    @classmethod
    def from_namespace(cls, configs):
        """
        Construct LACEConfig from the existing THUML-style argparse
        configuration object.

        This keeps LACE compatible with the current training pipeline.
        """

        return cls(
            context_len=int(configs.seq_len),
            pred_len=int(configs.pred_len),

            factors=tuple(
                getattr(
                    configs,
                    "lace_factors",
                    [1, 3, 12, 288],
                )
            ),

            backbone_name=getattr(
                configs,
                "lace_backbone",
                "Salesforce/moirai-1.0-R-small",
            ),

            adapter_dim=int(
                getattr(
                    configs,
                    "lace_adapter_dim",
                    128,
                )
            ),

            representation_norm=getattr(
                configs,
                "lace_representation_norm",
                "batch",
            ),

            reduction=getattr(
                configs,
                "lace_reduction",
                "sum",
            ),

            coherence_mode=getattr(
                configs,
                "lace_coherence_mode",
                "structural",
            ),

            lambda_coherence=float(
                getattr(
                    configs,
                    "lace_lambda",
                    1.0,
                )
            ),

            structural_bias_weight=float(
                getattr(
                    configs,
                    "lace_bias_weight",
                    1.0,
                )
            ),
        )
