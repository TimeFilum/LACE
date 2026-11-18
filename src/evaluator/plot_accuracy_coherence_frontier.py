
"""Plot forecast accuracy against cross-resolution inconsistency.

Compare independent TSFM forecasts, reconciliation controls, A0, and A1.
Both axes show errors, so lower values indicate better results. Accuracy
uses unique resolutions represented by the selected adjacent edges; road
comparisons use 5m->15m and 15m->1h. Save one figure per dataset.

Run from the repository root:
    python -m src.evaluator.plot_accuracy_coherence_frontier --help"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT IMPORTS
# ============================================================

try:
    from src.config import EVALUATION_RESULTS_PATH
except ImportError:
    from config import EVALUATION_RESULTS_PATH

try:
    from src.evaluator.evaluate_lace import (
        get_dataset_config,
        aggregate_sum,
        nmse,
        cross_resolution_consistency,
    )
except ImportError:
    from evaluator.evaluate_lace import (
        get_dataset_config,
        aggregate_sum,
        nmse,
        cross_resolution_consistency,
    )


DATASETS = [
    "c_traffic",
    "k_traffic",
    "ETTm1_HUFL",
    "road_sensor_62",
    "road_sensor_103",
    "road_sensor_240",
]

BACKBONES = {
    "Moirai-1": {
        "phase0": "Moirai1",
        "lace": "LACE_moirai1",
        "a0_wls": "LACE_A0_Moirai1",
    },
    "Moirai-2": {
        "phase0": "Moirai2",
        "lace": "LACE_moirai2",
        "a0_wls": "LACE_A0_Moirai2",
    },
    "Chronos-2": {
        "phase0": "Chronos2",
        "lace": "LACE_chronos2",
        "a0_wls": "LACE_A0_Chronos2",
    },
    "TimesFM-2": {
        "phase0": "TimesFM2",
        "lace": "LACE_timesfm2",
        "a0_wls": "LACE_A0_TimesFM2",
    },
}

METHOD_ORDER = [
    "Independent",
    "Bottom-Up",
    "WLS-Struct",
    "LACE-A0",
    "LACE-A0+WLS",
    "LACE-A1",
]

METHOD_MARKERS = {
    "Independent": "o",
    "Bottom-Up": "s",
    "WLS-Struct": "^",
    "LACE-A0": "D",
    "LACE-A0+WLS": "P",
    "LACE-A1": "X",
}


# ============================================================
# HELPERS
# ============================================================

def resolve_phase0_root(cli_root):
    if cli_root:
        return cli_root

    for module_name in ("src.config", "config"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        root = getattr(module, "RESULT_PHASE0_DIR", None)
        if root:
            return str(root)

    raise RuntimeError(
        "RESULT_PHASE0_DIR was not found in config. "
        "Pass --phase0-root explicitly."
    )


def squeeze(x):
    x = np.asarray(x)

    if x.ndim == 3:
        if x.shape[-1] != 1:
            raise ValueError(f"Expected singleton channel: {x.shape}")
        x = x[..., 0]

    if x.ndim != 2:
        raise ValueError(f"Expected [N,H] or [N,H,1], got {x.shape}")

    return x


def dataset_config(dataset, road_data_root):
    try:
        cfg = get_dataset_config(
            dataset,
            road_data_root=road_data_root,
        )
    except TypeError:
        cfg = get_dataset_config(dataset)

    factor_to_res = cfg["factor_to_res"]
    factors = list(factor_to_res.keys())

    return cfg, factor_to_res, factors


def comparison_pairs(cfg, factors):
    """
    Paper-facing frontier:
      - adjacent hierarchy edges only;
      - road excludes the daily edge, matching current road reporting.
    """
    pairs = list(zip(factors[:-1], factors[1:]))

    if cfg["decision"] == "road_capacity":
        pairs = [
            pair
            for pair in pairs
            if cfg["factor_to_res"][pair[1]] != "1d"
        ]

    return pairs


def selected_factors(pairs):
    out = set()
    for fine, coarse in pairs:
        out.add(fine)
        out.add(coarse)
    return sorted(out)


def align_common(preds, truths):
    n = min(
        min(len(x) for x in preds.values()),
        min(len(x) for x in truths.values()),
    )

    preds = {
        factor: x[:n]
        for factor, x in preds.items()
    }
    truths = {
        factor: x[:n]
        for factor, x in truths.items()
    }

    return preds, truths


def point_metrics(preds, truths, pairs):
    """
    x: mean NMSE over unique factors represented by the frontier edges.
    y: mean C_rs over those edges.
    """
    factors = selected_factors(pairs)

    acc_values = [
        nmse(
            preds[factor],
            truths[factor],
        )
        for factor in factors
    ]

    coh_values = []

    for fine, coarse in pairs:
        ratio = coarse // fine

        agg_pred = aggregate_sum(
            preds[fine],
            ratio,
        )

        direct_pred = preds[coarse]

        coh_values.append(
            cross_resolution_consistency(
                agg_pred,
                direct_pred,
            )
        )

    return (
        float(np.mean(acc_values)),
        float(np.mean(coh_values)),
    )


# ============================================================
# ZERO-SHOT INDEPENDENT
# ============================================================

def load_phase0_independent(
    phase0_root,
    model,
    dataset,
    cfg,
    factor_to_res,
):
    preds = {}
    truths = {}

    daily_stride = int(cfg["daily_stride"])

    for factor, resolution in factor_to_res.items():
        folder = os.path.join(
            phase0_root,
            model,
            dataset,
            resolution,
        )

        pred = squeeze(
            np.load(
                os.path.join(
                    folder,
                    "pred.npy",
                )
            )
        )
        true = squeeze(
            np.load(
                os.path.join(
                    folder,
                    "true.npy",
                )
            )
        )

        # Independent resolution-specific rolling forecast origins.
        native_day_horizon = (
            daily_stride // int(factor)
        )

        preds[factor] = pred[
            ::native_day_horizon
        ]
        truths[factor] = true[
            ::native_day_horizon
        ]

    return align_common(
        preds,
        truths,
    )


# ============================================================
# FACTOR-FILE LOADERS
# ============================================================

def load_factor_folder(
    folder,
    factors,
    stride=None,
):
    preds = {}
    truths = {}

    for factor in factors:
        pred_path = os.path.join(
            folder,
            f"pred_factor_{factor}.npy",
        )
        true_path = os.path.join(
            folder,
            f"true_factor_{factor}.npy",
        )

        if not os.path.isfile(pred_path):
            raise FileNotFoundError(pred_path)
        if not os.path.isfile(true_path):
            raise FileNotFoundError(true_path)

        pred = squeeze(
            np.load(pred_path)
        )
        true = squeeze(
            np.load(true_path)
        )

        if stride is not None:
            pred = pred[::stride]
            true = true[::stride]

        preds[factor] = pred
        truths[factor] = true

    return align_common(
        preds,
        truths,
    )


def load_zero_shot_bottomup(
    phase0_model,
    dataset,
    cfg,
    factors,
):
    folder = os.path.join(
        EVALUATION_RESULTS_PATH,
        f"{phase0_model}_bottomup",
        dataset,
        cfg["base_resolution"],
        "bottomup",
    )

    # BU control arrays are generated from every finest-resolution
    # rolling window, so recover common physical-day origins here.
    return load_factor_folder(
        folder,
        factors,
        stride=int(cfg["daily_stride"]),
    )


def load_zero_shot_wls(
    phase0_model,
    dataset,
    cfg,
    factors,
):
    folder = os.path.join(
        EVALUATION_RESULTS_PATH,
        f"{phase0_model}_mint_struct",
        dataset,
        cfg["base_resolution"],
        "mint_struct",
    )

    # WLS control is already saved after daily-origin alignment.
    return load_factor_folder(
        folder,
        factors,
        stride=None,
    )


def load_lace(
    model_dir,
    dataset,
    cfg,
    factors,
    mode,
):
    folder = os.path.join(
        EVALUATION_RESULTS_PATH,
        model_dir,
        dataset,
        cfg["base_resolution"],
        mode,
    )

    # All LACE heads share the same finest-resolution forecast origins.
    return load_factor_folder(
        folder,
        factors,
        stride=int(cfg["daily_stride"]),
    )


def load_lace_a0_wls(
    alias,
    dataset,
    cfg,
    factors,
):
    folder = os.path.join(
        EVALUATION_RESULTS_PATH,
        f"{alias}_mint_struct",
        dataset,
        cfg["base_resolution"],
        "mint_struct",
    )

    # Already daily-aligned by reconcile_controls.py.
    return load_factor_folder(
        folder,
        factors,
        stride=None,
    )


# ============================================================
# BUILD FRONTIER TABLE
# ============================================================

def build_frontier(
    phase0_root,
    datasets,
    road_data_root,
    allow_missing,
):
    rows = []

    for dataset in datasets:
        cfg, factor_to_res, factors = dataset_config(
            dataset,
            road_data_root,
        )

        pairs = comparison_pairs(
            cfg,
            factors,
        )

        print()
        print("=" * 78)
        print(dataset)
        print(
            "Frontier pairs:",
            [
                (
                    factor_to_res[fine],
                    factor_to_res[coarse],
                )
                for fine, coarse in pairs
            ],
        )
        print("=" * 78)

        for backbone, names in BACKBONES.items():

            loaders = [
                (
                    "Independent",
                    lambda: load_phase0_independent(
                        phase0_root,
                        names["phase0"],
                        dataset,
                        cfg,
                        factor_to_res,
                    ),
                ),
                (
                    "Bottom-Up",
                    lambda: load_zero_shot_bottomup(
                        names["phase0"],
                        dataset,
                        cfg,
                        factors,
                    ),
                ),
                (
                    "WLS-Struct",
                    lambda: load_zero_shot_wls(
                        names["phase0"],
                        dataset,
                        cfg,
                        factors,
                    ),
                ),
                (
                    "LACE-A0",
                    lambda: load_lace(
                        names["lace"],
                        dataset,
                        cfg,
                        factors,
                        "none",
                    ),
                ),
                (
                    "LACE-A0+WLS",
                    lambda: load_lace_a0_wls(
                        names["a0_wls"],
                        dataset,
                        cfg,
                        factors,
                    ),
                ),
                (
                    "LACE-A1",
                    lambda: load_lace(
                        names["lace"],
                        dataset,
                        cfg,
                        factors,
                        "structural",
                    ),
                ),
            ]

            for method, loader in loaders:
                try:
                    preds, truths = loader()

                    accuracy, coherence = point_metrics(
                        preds,
                        truths,
                        pairs,
                    )

                    rows.append({
                        "Dataset": dataset,
                        "Backbone": backbone,
                        "Method": method,
                        "Accuracy_NMSE": accuracy,
                        "Coherence_Crs": coherence,
                    })

                    print(
                        f"{backbone:10s} | "
                        f"{method:12s} | "
                        f"NMSE={accuracy:.6f} | "
                        f"C_rs={coherence:.6f}"
                    )

                except Exception as exc:
                    if not allow_missing:
                        raise

                    print(
                        f"[MISSING] {backbone} | "
                        f"{method} | {exc}"
                    )

    return pd.DataFrame(rows)


# ============================================================
# PLOT
# ============================================================

def plot_dataset(
    df,
    dataset,
    output_dir,
):
    d = df[
        df["Dataset"] == dataset
    ].copy()

    if d.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8.2, 6.0)
    )

    # One trajectory per backbone. Matplotlib's normal cycle handles
    # visual distinction; no explicit colors are hard-coded.
    for backbone in BACKBONES:
        b = d[
            d["Backbone"] == backbone
        ].copy()

        if b.empty:
            continue

        b["Method"] = pd.Categorical(
            b["Method"],
            categories=METHOD_ORDER,
            ordered=True,
        )
        b = b.sort_values(
            "Method"
        )

        # Connect available stages to make the trade-off trajectory visible.
        ax.plot(
            b["Accuracy_NMSE"],
            b["Coherence_Crs"],
            linewidth=1,
            alpha=0.55,
        )

        for _, row in b.iterrows():
            method = str(row["Method"])

            ax.scatter(
                row["Accuracy_NMSE"],
                row["Coherence_Crs"],
                marker=METHOD_MARKERS[method],
                s=72,
                label=f"{backbone} — {method}",
            )

            # Short labels keep the frontier readable.
            short_backbone = {
                "Moirai-1": "M1",
                "Moirai-2": "M2",
                "Chronos-2": "C2",
                "TimesFM-2": "TF2",
            }[backbone]

            short_method = {
                "Independent": "Ind",
                "Bottom-Up": "BU",
                "WLS-Struct": "WLS",
                "LACE-A0": "A0",
                "LACE-A0+WLS": "A0+WLS",
                "LACE-A1": "A1",
            }[method]

            ax.annotate(
                f"{short_backbone}-{short_method}",
                (
                    row["Accuracy_NMSE"],
                    row["Coherence_Crs"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel(
        "Mean NMSE ↓"
    )
    ax.set_ylabel(
        r"Mean cross-resolution inconsistency $C_{r,s}$ ↓"
    )
    ax.set_title(
        f"Accuracy–coherence frontier — {dataset}"
    )
    ax.grid(
        alpha=0.2
    )

    # Points are annotated directly.
    fig.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"{dataset}_accuracy_coherence_frontier.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Saved:", output_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase0-root",
        default=None,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
    )

    parser.add_argument(
        "--road-data-root",
        default="./data/Road",
    )

    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            EVALUATION_RESULTS_PATH,
            "accuracy_coherence_frontier",
        ),
    )

    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Plot completed points and report missing combinations "
            "instead of failing."
        ),
    )

    args = parser.parse_args()

    phase0_root = resolve_phase0_root(
        args.phase0_root
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    frontier = build_frontier(
        phase0_root=phase0_root,
        datasets=args.datasets,
        road_data_root=args.road_data_root,
        allow_missing=args.allow_missing,
    )

    csv_path = os.path.join(
        args.output_dir,
        "accuracy_coherence_frontier_points.csv",
    )

    frontier.to_csv(
        csv_path,
        index=False,
    )

    print()
    print("Saved:", csv_path)

    for dataset in args.datasets:
        plot_dataset(
            frontier,
            dataset,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
