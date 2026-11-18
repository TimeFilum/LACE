"""Apply WLS-Struct or OLS reconciliation to independent supervised forecasts."""
# src/evaluator/reconcile_baselines.py

import os
import argparse
import numpy as np

from src.config import EVALUATION_RESULTS_PATH


# ============================================================
# DATASET HIERARCHIES
# ============================================================

DATASETS = {

    "c_traffic": {
        "base_resolution": "10m",
        "resolutions": ["10m", "1h", "6h", "1d"],
        "factors": [1, 6, 36, 144],
        "horizons": [144, 24, 4, 1],
    },

    "k_traffic": {
        "base_resolution": "5m",
        "resolutions": ["5m", "15m", "1h", "1d"],
        "factors": [1, 3, 12, 288],
        "horizons": [288, 96, 24, 1],
    },

    "ETTm1_HUFL": {
        "base_resolution": "15m",
        "resolutions": ["15m", "1h", "6h", "1d"],
        "factors": [1, 4, 24, 96],
        "horizons": [96, 24, 4, 1],
    },

    "road_sensor_62": {
        "base_resolution": "5m",
        "resolutions": ["5m", "15m", "1h", "1d"],
        "factors": [1, 3, 12, 288],
        "horizons": [288, 96, 24, 1],
    },

    "road_sensor_103": {
        "base_resolution": "5m",
        "resolutions": ["5m", "15m", "1h", "1d"],
        "factors": [1, 3, 12, 288],
        "horizons": [288, 96, 24, 1],
    },

    "road_sensor_240": {
        "base_resolution": "5m",
        "resolutions": ["5m", "15m", "1h", "1d"],
        "factors": [1, 3, 12, 288],
        "horizons": [288, 96, 24, 1],
    },
}


# ============================================================
# DIRECT FORECAST PATHS
# ============================================================

def direct_folder(model, dataset, resolution):

    return os.path.join(
        EVALUATION_RESULTS_PATH,
        f"{model}_independent",
        dataset,
        resolution,
        "direct",
    )


def load_direct(model, dataset, resolution):

    folder = direct_folder(
        model,
        dataset,
        resolution,
    )

    pred_path = os.path.join(
        folder,
        "pred.npy",
    )

    true_path = os.path.join(
        folder,
        "true.npy",
    )

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Missing direct prediction:\n{pred_path}"
        )

    if not os.path.exists(true_path):
        raise FileNotFoundError(
            f"Missing direct truth:\n{true_path}"
        )

    pred = np.load(pred_path)
    true = np.load(true_path)

    if pred.ndim == 2:
        pred = pred[..., None]

    if true.ndim == 2:
        true = true[..., None]

    if pred.shape != true.shape:
        raise RuntimeError(
            f"{model} {dataset}/{resolution}: "
            f"pred={pred.shape}, true={true.shape}"
        )

    print(
        f"[LOAD] {model:10s} | "
        f"{dataset:18s} | "
        f"{resolution:4s} | "
        f"{pred.shape}"
    )

    return pred, true, folder


# ============================================================
# ALIGN COMMON DAILY FORECAST ORIGINS
# ============================================================

def align_daily(pred, true, horizon):
    """
    Forecast windows advance one native timestep.

    Since each horizon spans one physical day, taking every
    horizon-th test window produces daily forecast origins.
    """

    return (
        pred[::horizon],
        true[::horizon],
    )


# ============================================================
# TEMPORAL HIERARCHY MATRIX
# ============================================================

def aggregation_matrix(
    base_horizon,
    factor,
):

    if base_horizon % factor != 0:
        raise ValueError(
            f"{base_horizon} not divisible by {factor}"
        )

    coarse_horizon = (
        base_horizon // factor
    )

    A = np.zeros(
        (
            coarse_horizon,
            base_horizon,
        ),
        dtype=np.float64,
    )

    for i in range(coarse_horizon):

        start = i * factor
        end = start + factor

        A[
            i,
            start:end,
        ] = 1.0

    return A


def build_summing_matrix(
    base_horizon,
    factors,
):

    blocks = [

        aggregation_matrix(
            base_horizon,
            factor,
        )

        for factor in factors
    ]

    return np.concatenate(
        blocks,
        axis=0,
    )


# ============================================================
# STACK LEVELS
# ============================================================

def stack_levels(arrays):

    return np.concatenate(
        [
            x[..., 0]
            for x in arrays
        ],
        axis=1,
    )


def split_levels(
    stacked,
    horizons,
):

    outputs = []

    start = 0

    for horizon in horizons:

        end = start + horizon

        outputs.append(
            stacked[
                :,
                start:end,
                None,
            ]
        )

        start = end

    return outputs


# ============================================================
# CHECK TRUTH ALIGNMENT
# ============================================================

def aggregate_fine(
    fine,
    factor,
):

    if factor == 1:
        return fine.copy()

    n, t, c = fine.shape

    if t % factor != 0:
        raise RuntimeError(
            f"Horizon {t} cannot aggregate "
            f"by factor {factor}"
        )

    return (
        fine
        .reshape(
            n,
            t // factor,
            factor,
            c,
        )
        .sum(axis=2)
    )


def check_truth_hierarchy(
    truths,
    factors,
):

    fine = truths[0]

    for direct, factor in zip(
        truths,
        factors,
    ):

        generated = aggregate_fine(
            fine,
            factor,
        )

        max_abs = np.max(
            np.abs(
                generated - direct
            )
        )

        scale = max(
            np.max(np.abs(direct)),
            1.0,
        )

        rel = max_abs / scale

        print(
            f"[TRUTH CHECK] "
            f"factor={factor:<4} | "
            f"max_abs={max_abs:.6g} | "
            f"relative={rel:.3e}"
        )

        if not np.allclose(
            generated,
            direct,
            rtol=1e-5,
            atol=1e-5,
        ):

            raise RuntimeError(
                "Cross-resolution truths are not aligned "
                f"for factor={factor}."
            )


# ============================================================
# RECONCILIATION
# ============================================================

def reconcile(
    base_forecasts,
    S,
    covariance,
):

    W = np.asarray(
        covariance,
        dtype=np.float64,
    )

    # Numerical protection
    scale = max(
        np.trace(W) / W.shape[0],
        1.0,
    )

    W = (
        W
        + 1e-8
        * scale
        * np.eye(W.shape[0])
    )


    # W^-1 S
    WinvS = np.linalg.solve(
        W,
        S,
    )


    # (S' W^-1 S)^-1
    middle_inv = np.linalg.pinv(
        S.T @ WinvS
    )


    # Weighted least-squares projection
    G = (
        middle_inv
        @ WinvS.T
    )


    # Estimate coherent bottom-level vector
    bottom = (
        base_forecasts
        @ G.T
    )


    # Reconstruct all levels
    reconciled = (
        bottom
        @ S.T
    )

    return reconciled


# ============================================================
# STRUCTURAL WLS COVARIANCE
# ============================================================

def structural_covariance(S):
    """
    Structural scaling approximation:

        W_struct = diag(S @ 1)

    Each node variance is scaled by the number of finest-level
    components contributing to that aggregate.

    No forecast residuals are required.
    """

    structural_scale = S @ np.ones(
        S.shape[1],
        dtype=np.float64,
    )

    if np.any(structural_scale <= 0):
        raise RuntimeError(
            "Structural covariance contains non-positive values."
        )

    return np.diag(structural_scale)


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    model,
    dataset,
    method,
):

    cfg = DATASETS[dataset]

    resolutions = cfg["resolutions"]
    factors = cfg["factors"]
    horizons = cfg["horizons"]


    print()
    print("=" * 72)
    print(
        f"{model} | {dataset} | "
        f"{method.upper()}"
    )
    print("=" * 72)


    # --------------------------------------------------------
    # Load four independent forecasts
    # --------------------------------------------------------

    preds = []
    truths = []

    for resolution, horizon in zip(
        resolutions,
        horizons,
    ):

        pred, true, _ = load_direct(
            model,
            dataset,
            resolution,
        )

        pred, true = align_daily(
            pred,
            true,
            horizon,
        )

        preds.append(pred)
        truths.append(true)


    # --------------------------------------------------------
    # Common forecast origins
    # --------------------------------------------------------

    n_days = min(
        x.shape[0]
        for x in preds
    )


    preds = [
        x[:n_days]
        for x in preds
    ]

    truths = [
        x[:n_days]
        for x in truths
    ]


    print(
        f"[ALIGN] common daily origins = {n_days}"
    )


    # --------------------------------------------------------
    # Check ground-truth aggregation before reconciliation.
    # --------------------------------------------------------

    check_truth_hierarchy(
        truths,
        factors,
    )


    # --------------------------------------------------------
    # Build hierarchy matrix
    # --------------------------------------------------------

    base_horizon = horizons[0]

    S = build_summing_matrix(
        base_horizon,
        factors,
    )


    expected_nodes = sum(
        horizons
    )


    if S.shape != (
        expected_nodes,
        base_horizon,
    ):

        raise RuntimeError(
            f"Unexpected S shape: {S.shape}"
        )


    print(
        f"[S] shape={S.shape}"
    )


    base_forecasts = stack_levels(
        preds
    )


    # ========================================================
    # OLS RECONCILIATION
    # ========================================================

    if method == "ols":

        covariance = np.eye(
            expected_nodes,
            dtype=np.float64,
        )

        output_model = f"{model}_ols"
        output_mode = "ols"


    # ========================================================
    # WLS-STRUCT / STRUCTURAL COVARIANCE
    #
    # No residuals required.
    # Uses only the hierarchy structure:
    #
    #     W = diag(S @ 1)
    # ========================================================

    elif method == "mint_struct":

        covariance = structural_covariance(S)

        print(
            "[WLS-STRUCT] covariance:",
            covariance.shape,
        )

        print(
            "[WLS-STRUCT] diagonal range:",
            float(np.diag(covariance).min()),
            "->",
            float(np.diag(covariance).max()),
        )

        output_model = f"{model}_mint_struct"
        output_mode = "mint_struct"


    else:
        raise ValueError(method)


    # --------------------------------------------------------
    # Reconcile the test forecasts
    # --------------------------------------------------------

    reconciled = reconcile(
        base_forecasts,
        S,
        covariance,
    )


    # --------------------------------------------------------
    # Check exact numerical coherence
    # --------------------------------------------------------

    bottom = reconciled[
        :,
        :base_horizon,
    ]

    reconstructed = (
        bottom
        @ S.T
    )


    max_coherence_error = np.max(
        np.abs(
            reconciled
            - reconstructed
        )
    )


    print(
        "[COHERENCE CHECK] "
        f"max error={max_coherence_error:.3e}"
    )


    # --------------------------------------------------------
    # Split into evaluator factors
    # --------------------------------------------------------

    rec_levels = split_levels(
        reconciled,
        horizons,
    )


    # --------------------------------------------------------
    # Save exactly like LACE/Bottom-Up evaluator structure
    # --------------------------------------------------------

    output_folder = os.path.join(
        EVALUATION_RESULTS_PATH,
        output_model,
        dataset,
        cfg["base_resolution"],
        output_mode,
    )


    os.makedirs(
        output_folder,
        exist_ok=True,
    )


    for (
        factor,
        pred_factor,
        true_factor,
    ) in zip(
        factors,
        rec_levels,
        truths,
    ):

        np.save(
            os.path.join(
                output_folder,
                f"pred_factor_{factor}.npy",
            ),
            pred_factor,
        )

        np.save(
            os.path.join(
                output_folder,
                f"true_factor_{factor}.npy",
            ),
            true_factor,
        )


        print(
            f"[SAVE] factor={factor:<4} | "
            f"pred={pred_factor.shape} | "
            f"true={true_factor.shape}"
        )


    print(
        f"[DONE] {output_folder}"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model",
        default="DLinear",
        choices=[
            "DLinear",
            "LSTM",
        ],
    )


    parser.add_argument(
        "--dataset",
        default="all",
        choices=[
            "all",
            *DATASETS.keys(),
        ],
    )


    parser.add_argument(
        "--method",
        default="mint_struct",
        choices=[
            "ols",
            "mint_struct",
        ],
    )


    args = parser.parse_args()


    datasets = (
        DATASETS.keys()
        if args.dataset == "all"
        else [args.dataset]
    )


    for dataset in datasets:

        process_dataset(
            model=args.model,
            dataset=dataset,
            method=args.method,
        )
