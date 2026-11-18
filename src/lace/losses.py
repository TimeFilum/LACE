"""
Loss functions for LACE.

This module contains two objectives:

1. Accuracy loss
       standardized forecasting error at each resolution.

2. Structural coherence loss
       B M_r ~= M_s
       B b_r ~= b_s
       where M_r = W_r P_r.

No model architecture is defined here.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from lace.hierarchy import TemporalHierarchy


# ==================================================================
# Utilities
# ==================================================================

def _as_scalar_tensor(value, reference: torch.Tensor):
    """
    Convert a float/scalar statistic to a tensor matching the
    prediction device and dtype.
    """
    return torch.as_tensor(
        value,
        dtype=reference.dtype,
        device=reference.device,
    )


def _normalized_frobenius(x: torch.Tensor) -> torch.Tensor:
    """
    Frobenius/L2 norm normalized by sqrt(number of elements).

    This prevents larger mapping matrices from automatically producing
    larger structural penalties merely because they contain more
    parameters.
    """
    if x.numel() == 0:
        raise ValueError("Cannot compute norm of an empty tensor.")

    return torch.linalg.vector_norm(x) / math.sqrt(x.numel())


# ==================================================================
# 1. Forecast accuracy
# ==================================================================

def accuracy_loss(
    outputs: Dict[int, torch.Tensor],
    targets: Dict[int, torch.Tensor],
    factors,
    mus: Dict[int, float],
    stds: Dict[int, float],
    weights: Optional[Dict[int, float]] = None,
    eps: float = 1e-8,
):
    """
    Standardized multi-resolution forecasting loss.

    For resolution r:

        y_norm = (y - mu_r) / sigma_r

        L_r = MSE(y_hat_norm, y_true_norm)

    Parameters
    ----------
    outputs
        factor -> prediction [B, L_r]

    targets
        factor -> target [B, L_r]

    factors
        Active hierarchy factors.

    mus, stds
        Training-set statistics keyed by factor.

    weights
        Optional forecasting-loss weight per resolution.

    Returns
    -------
    total_loss : Tensor

    details : dict
        Individual resolution losses.
    """

    total = None
    details = {}

    for factor in factors:

        if factor not in outputs:
            raise KeyError(
                f"Missing prediction for factor {factor}."
            )

        if factor not in targets:
            raise KeyError(
                f"Missing target for factor {factor}."
            )

        if factor not in mus or factor not in stds:
            raise KeyError(
                f"Missing statistics for factor {factor}."
            )

        y_hat = outputs[factor]
        y_true = targets[factor]

        if y_hat.shape != y_true.shape:
            raise ValueError(
                f"Prediction/target mismatch at factor {factor}: "
                f"{y_hat.shape} vs {y_true.shape}."
            )

        mu = _as_scalar_tensor(
            mus[factor],
            y_hat,
        )

        std = _as_scalar_tensor(
            stds[factor],
            y_hat,
        )

        if torch.any(std <= 0):
            raise ValueError(
                f"Standard deviation must be positive "
                f"for factor {factor}."
            )

        y_hat_n = (y_hat - mu) / (std + eps)
        y_true_n = (y_true - mu) / (std + eps)

        loss_r = F.mse_loss(
            y_hat_n,
            y_true_n,
        )

        weight = (
            1.0
            if weights is None
            else float(weights.get(factor, 1.0))
        )

        weighted = weight * loss_r

        total = (
            weighted
            if total is None
            else total + weighted
        )

        details[f"acc_{factor}"] = loss_r.detach().item()

    if total is None:
        raise RuntimeError(
            "No accuracy losses were computed."
        )

    return total, details


# ==================================================================
# 2. Structural coherence
# ==================================================================

def structural_coherence_loss(
    mappings: Dict[int, torch.Tensor],
    biases: Dict[int, torch.Tensor],
    hierarchy: TemporalHierarchy,
    bias_weight: float = 1.0,
):
    """
    Structural regularization for LACE.

    Each resolution has:

        M_r = W_r P_r

    and:

        y_hat_r = M_r z + b_r

    For every adjacent hierarchy pair r -> s, enforce:

        B_{r->s} M_r ~= M_s
        B_{r->s} b_r ~= b_s

    Unlike output coherence, this objective is independent of the
    current batch predictions.

    Parameters
    ----------
    mappings
        factor -> M_r [L_r, D]

    biases
        factor -> b_r [L_r]

    hierarchy
        Temporal hierarchy operator.

    bias_weight
        Relative contribution of the bias regularizer.
    """

    total = None
    details = {}

    for fine, coarse in hierarchy.adjacent_pairs:

        if fine not in mappings or coarse not in mappings:
            raise KeyError(
                f"Missing mappings for hierarchy pair "
                f"{fine} -> {coarse}."
            )

        if fine not in biases or coarse not in biases:
            raise KeyError(
                f"Missing biases for hierarchy pair "
                f"{fine} -> {coarse}."
            )

        M_fine = mappings[fine]
        M_coarse = mappings[coarse]

        b_fine = biases[fine]
        b_coarse = biases[coarse]

        # ----------------------------------------------------------
        # B M_r
        # ----------------------------------------------------------
        aggregated_M = hierarchy.aggregate_mapping(
            M_fine,
            fine_factor=fine,
            coarse_factor=coarse,
        )

        # ----------------------------------------------------------
        # B b_r
        # ----------------------------------------------------------
        aggregated_b = hierarchy.aggregate_bias(
            b_fine,
            fine_factor=fine,
            coarse_factor=coarse,
        )

        if aggregated_M.shape != M_coarse.shape:
            raise ValueError(
                f"Mapping shape mismatch for {fine}->{coarse}: "
                f"{aggregated_M.shape} vs {M_coarse.shape}."
            )

        if aggregated_b.shape != b_coarse.shape:
            raise ValueError(
                f"Bias shape mismatch for {fine}->{coarse}: "
                f"{aggregated_b.shape} vs {b_coarse.shape}."
            )

        # ----------------------------------------------------------
        # Normalized effective-mapping mismatch
        # ----------------------------------------------------------
        map_loss = _normalized_frobenius(
            aggregated_M - M_coarse
        )

        bias_loss = _normalized_frobenius(
            aggregated_b - b_coarse
        )

        pair_loss = (
            map_loss
            + float(bias_weight) * bias_loss
        )

        total = (
            pair_loss
            if total is None
            else total + pair_loss
        )

        details[
            f"struct_map_{fine}_{coarse}"
        ] = map_loss.detach().item()

        details[
            f"struct_bias_{fine}_{coarse}"
        ] = bias_loss.detach().item()

        details[
            f"struct_total_{fine}_{coarse}"
        ] = pair_loss.detach().item()

    if total is None:
        reference = next(iter(mappings.values()))
        total = reference.new_tensor(0.0)

    return total, details
