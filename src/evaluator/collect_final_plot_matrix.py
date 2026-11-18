
"""Collect the paper's comparison tables and plotting data.

Accuracy uses each hierarchy resolution once, including road daily forecasts.
Coherence and decision summaries use adjacent edges; road summaries use
5m->15m and 15m->1h. Decision quality is provisioning cost for network traffic,
scheduling error for electricity, and decision accuracy for road traffic.
Positive A1-vs-A0 improvements indicate better results under A1.

Run from the repository root:
    python -m src.evaluator.collect_final_plot_matrix"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    RESULT_PHASE0_DIR,
    EVALUATION_RESULTS_PATH,
)

from src.evaluator.evaluate_lace import (
    get_dataset_config,
    aggregate_sum,
    nmse,
    cross_resolution_consistency,
    evaluate_decision_routes,
    compute_road_threshold,
)


# ============================================================
# EXPERIMENT MATRIX
# ============================================================

DATASETS = [
    "c_traffic",
    "k_traffic",
    "ETTm1_HUFL",
    "road_sensor_62",
    "road_sensor_103",
    "road_sensor_240",
]


DATASET_META = {
    "c_traffic": {
        "Domain": "Network",
        "Hierarchy_group": "Network-10m",
    },
    "k_traffic": {
        "Domain": "Network",
        "Hierarchy_group": "Network-5m",
    },
    "ETTm1_HUFL": {
        "Domain": "Energy",
        "Hierarchy_group": "Energy",
    },
    "road_sensor_62": {
        "Domain": "Road",
        "Hierarchy_group": "Road",
    },
    "road_sensor_103": {
        "Domain": "Road",
        "Hierarchy_group": "Road",
    },
    "road_sensor_240": {
        "Domain": "Road",
        "Hierarchy_group": "Road",
    },
}


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


TSFM_METHODS = [
    "Independent",
    "Bottom-Up",
    "WLS-Struct",
    "LACE-A0",
    "LACE-A0+BU",
    "LACE-A0+WLS",
    "LACE-A1",
]


SUPERVISED_METHODS = {
    ("DLinear", "bottomup"): "DLinear-BU",
    ("DLinear", "mint_struct"): "DLinear-WLS",
    ("LSTM", "bottomup"): "LSTM-BU",
    ("LSTM", "mint_struct"): "LSTM-WLS",
}


METHOD_ORDER = [
    "Independent",
    "Bottom-Up",
    "WLS-Struct",
    "LACE-A0",
    "LACE-A0+BU",
    "LACE-A0+WLS",
    "LACE-A1",
    "DLinear-BU",
    "DLinear-WLS",
    "LSTM-BU",
    "LSTM-WLS",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def squeeze(x):
    x = np.asarray(x)

    if x.ndim == 3:
        if x.shape[-1] != 1:
            raise ValueError(
                f"Expected singleton channel dimension, got {x.shape}"
            )
        x = x[..., 0]

    if x.ndim != 2:
        raise ValueError(
            f"Expected [N,H] or [N,H,1], got {x.shape}"
        )

    return x


def safe_mean(values):
    x = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(dtype=float)

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.mean(x))


def safe_improvement_pct(reference, candidate):
    """
    Lower-is-better percentage improvement.

    Positive = candidate better.
    """

    if not (
        np.isfinite(reference)
        and np.isfinite(candidate)
    ):
        return np.nan

    if np.isclose(reference, 0.0):
        if np.isclose(candidate, 0.0):
            return 0.0
        return np.nan

    return (
        100.0
        * (reference - candidate)
        / abs(reference)
    )


def quality_improvement_pct(
    a0,
    a1,
    direction,
):
    """
    Positive always means A1 better.
    """

    if not (
        np.isfinite(a0)
        and np.isfinite(a1)
    ):
        return np.nan

    if np.isclose(a0, 0.0):
        if np.isclose(a1, 0.0):
            return 0.0
        return np.nan

    if direction == "lower":
        return (
            100.0
            * (a0 - a1)
            / abs(a0)
        )

    if direction == "higher":
        return (
            100.0
            * (a1 - a0)
            / abs(a0)
        )

    raise ValueError(
        f"Unknown quality direction: {direction}"
    )


# ============================================================
# DATASET / HIERARCHY HELPERS
# ============================================================

def dataset_config(
    dataset,
    road_data_root,
):
    try:
        cfg = get_dataset_config(
            dataset,
            road_data_root=road_data_root,
        )
    except TypeError:
        cfg = get_dataset_config(dataset)

    factor_to_res = dict(
        cfg["factor_to_res"]
    )

    factors = sorted(
        factor_to_res.keys()
    )

    return (
        cfg,
        factor_to_res,
        factors,
    )


def hierarchy_string(
    factor_to_res,
    factors,
):
    return " -> ".join(
        factor_to_res[f]
        for f in factors
    )


def canonical_pairs(
    cfg,
    factors,
):
    """
    Coherence / operational comparison edges.

    All datasets:
        adjacent hierarchy edges only.

    Road:
        5m -> 15m
        15m -> 1h

    Daily road edge is excluded here.
    """

    pairs = list(
        zip(
            factors[:-1],
            factors[1:],
        )
    )

    if cfg["decision"] == "road_capacity":
        pairs = [
            (fine, coarse)
            for fine, coarse in pairs
            if cfg["factor_to_res"][coarse]
            != "1d"
        ]

    return pairs


def pair_name(
    factor_to_res,
    fine,
    coarse,
):
    return (
        f"{factor_to_res[fine]}"
        f"->{factor_to_res[coarse]}"
    )


# ============================================================
# ALIGNMENT
# ============================================================

def align_common(
    preds,
    truths,
):
    """
    Final safety alignment across hierarchy levels.
    """

    n = min(
        min(len(x) for x in preds.values()),
        min(len(x) for x in truths.values()),
    )

    preds = {
        factor: arr[:n]
        for factor, arr in preds.items()
    }

    truths = {
        factor: arr[:n]
        for factor, arr in truths.items()
    }

    return preds, truths


# ============================================================
# ARRAY LOADERS
# ============================================================

def load_phase0_independent(
    phase0_model,
    dataset,
    cfg,
    factor_to_res,
):
    """
    Zero-shot Independent.

    Each resolution has its own rolling origins.
    Subsample to common physical-day origins.
    """

    preds = {}
    truths = {}

    daily_stride = int(
        cfg["daily_stride"]
    )

    for factor, resolution in factor_to_res.items():

        folder = os.path.join(
            RESULT_PHASE0_DIR,
            phase0_model,
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

        pred = squeeze(
            np.load(pred_path)
        )

        true = squeeze(
            np.load(true_path)
        )

        native_day_horizon = (
            daily_stride
            // int(factor)
        )

        preds[factor] = (
            pred[
                ::native_day_horizon
            ]
        )

        truths[factor] = (
            true[
                ::native_day_horizon
            ]
        )

    return align_common(
        preds,
        truths,
    )


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

        if not os.path.isfile(
            pred_path
        ):
            raise FileNotFoundError(
                pred_path
            )

        if not os.path.isfile(
            true_path
        ):
            raise FileNotFoundError(
                true_path
            )

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


def load_bottom_up(
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

    # BU arrays originate from every finest-scale rolling window.
    return load_factor_folder(
        folder=folder,
        factors=factors,
        stride=int(
            cfg["daily_stride"]
        ),
    )


def load_wls(
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

    # Reconciliation control already daily aligned.
    return load_factor_folder(
        folder=folder,
        factors=factors,
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

    # LACE predictions share finest-resolution origins.
    return load_factor_folder(
        folder=folder,
        factors=factors,
        stride=int(
            cfg["daily_stride"]
        ),
    )


def load_a0_bottomup(
    alias,
    dataset,
    cfg,
    factors,
):
    folder = os.path.join(
        EVALUATION_RESULTS_PATH,
        f"{alias}_bottomup",
        dataset,
        cfg["base_resolution"],
        "bottomup",
    )

    # A0+BU is generated from every finest-resolution A0 origin,
    # so recover the same physical-day origins used by LACE A0/A1.
    return load_factor_folder(
        folder=folder,
        factors=factors,
        stride=int(
            cfg["daily_stride"]
        ),
    )


def load_a0_wls(
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

    return load_factor_folder(
        folder=folder,
        factors=factors,
        stride=None,
    )


# ============================================================
# DECISION QUALITY STANDARDIZATION
# ============================================================

def decision_quality_spec(dataset):

    if dataset in {
        "c_traffic",
        "k_traffic",
    }:
        return {
            "agg_col":
                "Agg_decision_cost",
            "direct_col":
                "Direct_decision_cost",
            "direction":
                "lower",
            "metric":
                "provisioning_cost",
        }

    if dataset == "ETTm1_HUFL":
        return {
            "agg_col":
                "Agg_decision_error",
            "direct_col":
                "Direct_decision_error",
            "direction":
                "lower",
            "metric":
                "scheduling_error",
        }

    if dataset.startswith(
        "road_sensor_"
    ):
        return {
            "agg_col":
                "Agg_decision_accuracy",
            "direct_col":
                "Direct_decision_accuracy",
            "direction":
                "higher",
            "metric":
                "decision_accuracy",
        }

    raise KeyError(
        f"Unknown dataset: {dataset}"
    )


def extract_quality_from_dict(
    result,
    dataset,
):
    spec = decision_quality_spec(
        dataset
    )

    agg = result.get(
        spec["agg_col"],
        np.nan,
    )

    direct = result.get(
        spec["direct_col"],
        np.nan,
    )

    return (
        agg,
        direct,
        spec["direction"],
        spec["metric"],
    )


# ============================================================
# TSFM FAMILY: RESOLUTION ACCURACY
# ============================================================

def collect_resolution_accuracy(
    backbone,
    dataset,
    method,
    preds,
    truths,
    factor_to_res,
    factors,
):
    """
    IMPORTANT:
    Forecast accuracy uses ALL hierarchy resolutions once.

    Road therefore keeps:
        5m, 15m, 1h, 1d
    """

    meta = DATASET_META[
        dataset
    ]

    rows = []

    for factor in factors:

        rows.append({
            "Dataset":
                dataset,

            "Domain":
                meta["Domain"],

            "Hierarchy_group":
                meta["Hierarchy_group"],

            "Backbone":
                backbone,

            "Method":
                method,

            "Family":
                "TSFM",

            "Factor":
                int(factor),

            "Resolution":
                factor_to_res[factor],

            "NMSE":
                float(
                    nmse(
                        preds[factor],
                        truths[factor],
                    )
                ),
        })

    return rows


# ============================================================
# TSFM FAMILY: PAIRWISE
# ============================================================

def collect_pairwise(
    backbone,
    dataset,
    method,
    preds,
    truths,
    cfg,
    factor_to_res,
    factors,
    pairs,
):
    meta = DATASET_META[
        dataset
    ]

    hierarchy = hierarchy_string(
        factor_to_res,
        factors,
    )

    road_q_cap = np.nan
    road_threshold = None

    if cfg["decision"] == "road_capacity":
        (
            road_q_cap,
            road_threshold,
        ) = compute_road_threshold(
            cfg["decision_params"]
        )

    rows = []

    for fine, coarse in pairs:

        fine_pred = preds[fine]
        fine_true = truths[fine]

        direct_pred = preds[coarse]
        coarse_true = truths[coarse]

        ratio = (
            int(coarse)
            // int(fine)
        )

        agg_pred = aggregate_sum(
            fine_pred,
            ratio,
        )

        if (
            agg_pred.shape
            != direct_pred.shape
        ):
            raise RuntimeError(
                f"{backbone} | {dataset} | {method} | "
                f"{pair_name(factor_to_res, fine, coarse)}: "
                f"aggregate={agg_pred.shape}, "
                f"direct={direct_pred.shape}"
            )

        if (
            direct_pred.shape
            != coarse_true.shape
        ):
            raise RuntimeError(
                f"{backbone} | {dataset} | {method} | "
                f"{factor_to_res[coarse]}: "
                f"pred={direct_pred.shape}, "
                f"true={coarse_true.shape}"
            )

        decision_result = (
            evaluate_decision_routes(
                cfg=cfg,
                fine=fine,
                coarse=coarse,
                agg_pred=agg_pred,
                direct_pred=direct_pred,
                true_coarse=coarse_true,
                road_threshold=road_threshold,
            )
        )

        (
            agg_quality,
            direct_quality,
            quality_direction,
            quality_metric,
        ) = extract_quality_from_dict(
            decision_result,
            dataset,
        )

        e_fine = float(
            nmse(
                fine_pred,
                fine_true,
            )
        )

        e_agg = float(
            nmse(
                agg_pred,
                coarse_true,
            )
        )

        e_dir = float(
            nmse(
                direct_pred,
                coarse_true,
            )
        )

        # CURRENT evaluator helper:
        # denominator = aggregate route.
        c_rs = float(
            cross_resolution_consistency(
                agg_pred,
                direct_pred,
            )
        )

        row = {
            "Dataset":
                dataset,

            "Domain":
                meta["Domain"],

            "Hierarchy_group":
                meta["Hierarchy_group"],

            "Hierarchy":
                hierarchy,

            "Backbone":
                backbone,

            "Method":
                method,

            "Family":
                "TSFM",

            "Pair":
                pair_name(
                    factor_to_res,
                    fine,
                    coarse,
                ),

            "Fine_factor":
                int(fine),

            "Coarse_factor":
                int(coarse),

            "Fine_resolution":
                factor_to_res[fine],

            "Coarse_resolution":
                factor_to_res[coarse],

            "E_fine":
                e_fine,

            "E_agg":
                e_agg,

            "E_dir":
                e_dir,

            "Delta_E":
                e_dir - e_agg,

            "Abs_Delta_E":
                abs(
                    e_dir - e_agg
                ),

            "C_rs":
                c_rs,

            "Decision_mismatch":
                decision_result.get(
                    "Decision_mismatch",
                    np.nan,
                ),

            "Agg_decision_quality":
                agg_quality,

            "Direct_decision_quality":
                direct_quality,

            "Decision_quality_direction":
                quality_direction,

            "Decision_quality_metric":
                quality_metric,
        }

        if cfg["decision"] == "road_capacity":

            row[
                "Q_cap_veh_per_h"
            ] = road_q_cap

            row[
                "Decision_threshold_veh_per_h"
            ] = road_threshold

        rows.append(row)

    return rows


# ============================================================
# TSFM FAMILY COLLECTION
# ============================================================

def collect_tsfm_family(
    road_data_root,
    allow_missing,
):
    pair_rows = []
    accuracy_rows = []
    status_rows = []

    for dataset in DATASETS:

        (
            cfg,
            factor_to_res,
            factors,
        ) = dataset_config(
            dataset,
            road_data_root,
        )

        pairs = canonical_pairs(
            cfg,
            factors,
        )

        print()
        print("=" * 80)
        print(dataset)

        print(
            "Hierarchy:",
            hierarchy_string(
                factor_to_res,
                factors,
            ),
        )

        print(
            "Canonical coherence/decision edges:",
            [
                pair_name(
                    factor_to_res,
                    fine,
                    coarse,
                )
                for fine, coarse
                in pairs
            ],
        )

        print(
            "Accuracy resolutions:",
            [
                factor_to_res[f]
                for f in factors
            ],
        )

        print("=" * 80)

        for backbone, names in BACKBONES.items():

            loaders = [
                (
                    "Independent",
                    lambda: load_phase0_independent(
                        phase0_model=
                            names["phase0"],
                        dataset=dataset,
                        cfg=cfg,
                        factor_to_res=
                            factor_to_res,
                    ),
                ),

                (
                    "Bottom-Up",
                    lambda: load_bottom_up(
                        phase0_model=
                            names["phase0"],
                        dataset=dataset,
                        cfg=cfg,
                        factors=factors,
                    ),
                ),

                (
                    "WLS-Struct",
                    lambda: load_wls(
                        phase0_model=
                            names["phase0"],
                        dataset=dataset,
                        cfg=cfg,
                        factors=factors,
                    ),
                ),

                (
                    "LACE-A0",
                    lambda: load_lace(
                        model_dir=
                            names["lace"],
                        dataset=dataset,
                        cfg=cfg,
                        factors=factors,
                        mode="none",
                    ),
                ),

                (
                    "LACE-A0+BU",
                    lambda: load_a0_bottomup(
                        alias=
                            names["a0_wls"],
                        dataset=dataset,
                        cfg=cfg,
                        factors=factors,
                    ),
                ),

                (
                    "LACE-A0+WLS",
                    lambda: load_a0_wls(
                        alias=
                            names["a0_wls"],
                        dataset=dataset,
                        cfg=cfg,
                        factors=factors,
                    ),
                ),

                (
                    "LACE-A1",
                    lambda: load_lace(
                        model_dir=
                            names["lace"],
                        dataset=dataset,
                        cfg=cfg,
                        factors=factors,
                        mode="structural",
                    ),
                ),
            ]

            for method, loader in loaders:

                try:
                    (
                        preds,
                        truths,
                    ) = loader()

                    accuracy_rows.extend(
                        collect_resolution_accuracy(
                            backbone=
                                backbone,
                            dataset=
                                dataset,
                            method=
                                method,
                            preds=
                                preds,
                            truths=
                                truths,
                            factor_to_res=
                                factor_to_res,
                            factors=
                                factors,
                        )
                    )

                    pair_rows.extend(
                        collect_pairwise(
                            backbone=
                                backbone,
                            dataset=
                                dataset,
                            method=
                                method,
                            preds=
                                preds,
                            truths=
                                truths,
                            cfg=
                                cfg,
                            factor_to_res=
                                factor_to_res,
                            factors=
                                factors,
                            pairs=
                                pairs,
                        )
                    )

                    status_rows.append({
                        "Dataset":
                            dataset,
                        "Backbone":
                            backbone,
                        "Method":
                            method,
                        "Status":
                            "OK",
                        "Message":
                            "",
                    })

                    print(
                        f"[OK] "
                        f"{backbone:10s} | "
                        f"{method}"
                    )

                except Exception as exc:

                    status_rows.append({
                        "Dataset":
                            dataset,
                        "Backbone":
                            backbone,
                        "Method":
                            method,
                        "Status":
                            "MISSING",
                        "Message":
                            str(exc),
                    })

                    if not allow_missing:
                        raise

                    print(
                        f"[MISSING] "
                        f"{backbone} | "
                        f"{dataset} | "
                        f"{method} | "
                        f"{exc}"
                    )

    return (
        pd.DataFrame(
            pair_rows
        ),
        pd.DataFrame(
            accuracy_rows
        ),
        pd.DataFrame(
            status_rows
        ),
    )


# ============================================================
# SUPERVISED BASELINE REFERENCES
# ============================================================

def canonical_pair_names_for_dataset(
    dataset,
    road_data_root,
):
    (
        cfg,
        factor_to_res,
        factors,
    ) = dataset_config(
        dataset,
        road_data_root,
    )

    return {
        pair_name(
            factor_to_res,
            fine,
            coarse,
        )
        for fine, coarse
        in canonical_pairs(
            cfg,
            factors,
        )
    }


def standardize_supervised_pairwise(
    baseline_pairwise,
    road_data_root,
):
    """
    Existing baseline evaluator files are retained.

    Only DLinear/LSTM BU/WLS references are added to the
    final canonical matrix.
    """

    rows = []

    for (
        model,
        baseline,
    ), method in (
        SUPERVISED_METHODS.items()
    ):

        for dataset in DATASETS:

            df = baseline_pairwise[
                (
                    baseline_pairwise[
                        "Forecaster"
                    ]
                    == model
                )
                &
                (
                    baseline_pairwise[
                        "Baseline"
                    ]
                    == baseline
                )
                &
                (
                    baseline_pairwise[
                        "Dataset"
                    ]
                    == dataset
                )
            ].copy()

            if df.empty:
                continue

            valid_pairs = (
                canonical_pair_names_for_dataset(
                    dataset,
                    road_data_root,
                )
            )

            df = df[
                df["Pair"].isin(
                    valid_pairs
                )
            ].copy()

            (
                cfg,
                factor_to_res,
                factors,
            ) = dataset_config(
                dataset,
                road_data_root,
            )

            meta = DATASET_META[
                dataset
            ]

            quality_spec = (
                decision_quality_spec(
                    dataset
                )
            )

            for _, row in df.iterrows():

                agg_quality = row.get(
                    quality_spec[
                        "agg_col"
                    ],
                    np.nan,
                )

                direct_quality = row.get(
                    quality_spec[
                        "direct_col"
                    ],
                    np.nan,
                )

                rows.append({
                    "Dataset":
                        dataset,

                    "Domain":
                        meta["Domain"],

                    "Hierarchy_group":
                        meta["Hierarchy_group"],

                    "Hierarchy":
                        hierarchy_string(
                            factor_to_res,
                            factors,
                        ),

                    "Backbone":
                        model,

                    "Method":
                        method,

                    "Family":
                        "Supervised",

                    "Pair":
                        row["Pair"],

                    "Fine_factor":
                        row[
                            "Fine_factor"
                        ],

                    "Coarse_factor":
                        row[
                            "Coarse_factor"
                        ],

                    "Fine_resolution":
                        str(
                            row["Pair"]
                        ).split(
                            "->"
                        )[0],

                    "Coarse_resolution":
                        str(
                            row["Pair"]
                        ).split(
                            "->"
                        )[1],

                    "E_fine":
                        row.get(
                            "E_fine",
                            np.nan,
                        ),

                    "E_agg":
                        row.get(
                            "E_agg",
                            np.nan,
                        ),

                    "E_dir":
                        row.get(
                            "E_dir",
                            np.nan,
                        ),

                    "Delta_E":
                        row.get(
                            "Delta_E",
                            np.nan,
                        ),

                    "Abs_Delta_E":
                        row.get(
                            "Abs_Delta_E",
                            np.nan,
                        ),

                    "C_rs":
                        row.get(
                            "C_rs",
                            np.nan,
                        ),

                    "Decision_mismatch":
                        row.get(
                            "Decision_mismatch",
                            np.nan,
                        ),

                    "Agg_decision_quality":
                        agg_quality,

                    "Direct_decision_quality":
                        direct_quality,

                    "Decision_quality_direction":
                        quality_spec[
                            "direction"
                        ],

                    "Decision_quality_metric":
                        quality_spec[
                            "metric"
                        ],
                })

    return pd.DataFrame(
        rows
    )


def standardize_supervised_accuracy(
    baseline_accuracy,
    road_data_root,
):
    rows = []

    for (
        model,
        baseline,
    ), method in (
        SUPERVISED_METHODS.items()
    ):

        for dataset in DATASETS:

            df = baseline_accuracy[
                (
                    baseline_accuracy[
                        "Forecaster"
                    ]
                    == model
                )
                &
                (
                    baseline_accuracy[
                        "Baseline"
                    ]
                    == baseline
                )
                &
                (
                    baseline_accuracy[
                        "Dataset"
                    ]
                    == dataset
                )
            ].copy()

            if df.empty:
                continue

            (
                _,
                factor_to_res,
                factors,
            ) = dataset_config(
                dataset,
                road_data_root,
            )

            res_to_factor = {
                resolution: factor
                for factor, resolution
                in factor_to_res.items()
            }

            meta = DATASET_META[
                dataset
            ]

            for _, row in df.iterrows():

                resolution = row.get(
                    "Resolution",
                    None,
                )

                factor = row.get(
                    "Factor",
                    np.nan,
                )

                # Robust fallback if old accuracy CSV has
                # resolution but not factor.
                if (
                    not np.isfinite(
                        pd.to_numeric(
                            pd.Series(
                                [factor]
                            ),
                            errors="coerce",
                        ).iloc[0]
                    )
                    and resolution
                    in res_to_factor
                ):
                    factor = (
                        res_to_factor[
                            resolution
                        ]
                    )

                rows.append({
                    "Dataset":
                        dataset,

                    "Domain":
                        meta["Domain"],

                    "Hierarchy_group":
                        meta[
                            "Hierarchy_group"
                        ],

                    "Backbone":
                        model,

                    "Method":
                        method,

                    "Family":
                        "Supervised",

                    "Factor":
                        factor,

                    "Resolution":
                        resolution,

                    "NMSE":
                        row["NMSE"],
                })

    return pd.DataFrame(
        rows
    )


def load_supervised_references(
    road_data_root,
    allow_missing,
):
    summary_dir = os.path.join(
        EVALUATION_RESULTS_PATH,
        "Baseline_Evaluation",
        "_summary",
    )

    pair_path = os.path.join(
        summary_dir,
        "BASELINE_all_results.csv",
    )

    accuracy_path = os.path.join(
        summary_dir,
        "BASELINE_resolution_accuracy.csv",
    )

    if not (
        os.path.isfile(pair_path)
        and os.path.isfile(
            accuracy_path
        )
    ):
        message = (
            "Baseline summary files missing:\n"
            f"  {pair_path}\n"
            f"  {accuracy_path}"
        )

        if allow_missing:
            print(
                "[WARNING]",
                message,
            )

            return (
                pd.DataFrame(),
                pd.DataFrame(),
            )

        raise FileNotFoundError(
            message
        )

    baseline_pairwise = pd.read_csv(
        pair_path
    )

    baseline_accuracy = pd.read_csv(
        accuracy_path
    )

    pairwise = (
        standardize_supervised_pairwise(
            baseline_pairwise,
            road_data_root,
        )
    )

    accuracy = (
        standardize_supervised_accuracy(
            baseline_accuracy,
            road_data_root,
        )
    )

    return (
        pairwise,
        accuracy,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_pairwise(
    pairwise,
):
    if pairwise.empty:
        raise RuntimeError(
            "FINAL_pairwise would be empty."
        )

    duplicated = pairwise.duplicated(
        subset=[
            "Dataset",
            "Backbone",
            "Method",
            "Pair",
        ],
        keep=False,
    )

    if duplicated.any():
        raise RuntimeError(
            "Duplicate canonical pair rows found:\n"
            + pairwise.loc[
                duplicated,
                [
                    "Dataset",
                    "Backbone",
                    "Method",
                    "Pair",
                ],
            ].to_string(
                index=False
            )
        )


def validate_accuracy(
    accuracy,
):
    if accuracy.empty:
        raise RuntimeError(
            "FINAL_resolution_accuracy would be empty."
        )

    duplicated = accuracy.duplicated(
        subset=[
            "Dataset",
            "Backbone",
            "Method",
            "Factor",
            "Resolution",
        ],
        keep=False,
    )

    if duplicated.any():
        raise RuntimeError(
            "Duplicate resolution accuracy rows found:\n"
            + accuracy.loc[
                duplicated,
                [
                    "Dataset",
                    "Backbone",
                    "Method",
                    "Factor",
                    "Resolution",
                ],
            ].to_string(
                index=False
            )
        )


# ============================================================
# CASE SUMMARY
# ============================================================

def route_level_decision_quality(
    pair_df,
):
    """
    Explicit paper definition:

    mean route-level decision quality over selected edges.

    Each selected edge contributes:
        aggregate-route quality
        direct-route quality
    """

    values = []

    for col in [
        "Agg_decision_quality",
        "Direct_decision_quality",
    ]:
        x = pd.to_numeric(
            pair_df[col],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        x = x[
            np.isfinite(x)
        ]

        values.extend(
            x.tolist()
        )

    if not values:
        return np.nan

    return float(
        np.mean(values)
    )


def build_case_summary(
    pairwise,
    accuracy,
):
    rows = []

    keys = (
        pairwise[
            [
                "Dataset",
                "Backbone",
                "Method",
            ]
        ]
        .drop_duplicates()
    )

    for _, key in keys.iterrows():

        dataset = key[
            "Dataset"
        ]

        backbone = key[
            "Backbone"
        ]

        method = key[
            "Method"
        ]

        pair_df = pairwise[
            (
                pairwise["Dataset"]
                == dataset
            )
            &
            (
                pairwise["Backbone"]
                == backbone
            )
            &
            (
                pairwise["Method"]
                == method
            )
        ].copy()

        acc_df = accuracy[
            (
                accuracy["Dataset"]
                == dataset
            )
            &
            (
                accuracy["Backbone"]
                == backbone
            )
            &
            (
                accuracy["Method"]
                == method
            )
        ].copy()

        if acc_df.empty:
            raise RuntimeError(
                f"No accuracy rows for "
                f"{dataset} | "
                f"{backbone} | "
                f"{method}"
            )

        accuracy_nmse = safe_mean(
            acc_df["NMSE"]
        )

        coherence = safe_mean(
            pair_df["C_rs"]
        )

        mismatch = safe_mean(
            pair_df[
                "Decision_mismatch"
            ]
        )

        quality = (
            route_level_decision_quality(
                pair_df
            )
        )

        directions = (
            pair_df[
                "Decision_quality_direction"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        metrics = (
            pair_df[
                "Decision_quality_metric"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if len(directions) != 1:
            raise RuntimeError(
                f"Decision direction ambiguity: "
                f"{dataset} | {backbone} | "
                f"{method} | {directions}"
            )

        if len(metrics) != 1:
            raise RuntimeError(
                f"Decision metric ambiguity: "
                f"{dataset} | {backbone} | "
                f"{method} | {metrics}"
            )

        meta = DATASET_META[
            dataset
        ]

        rows.append({
            "Dataset":
                dataset,

            "Domain":
                meta["Domain"],

            "Hierarchy_group":
                meta["Hierarchy_group"],

            "Backbone":
                backbone,

            "Method":
                method,

            "Family":
                pair_df[
                    "Family"
                ].iloc[0],

            "Accuracy_NMSE":
                accuracy_nmse,

            "Coherence_Crs":
                coherence,

            "Decision_quality":
                quality,

            "Decision_quality_direction":
                directions[0],

            "Decision_quality_metric":
                metrics[0],

            "Decision_mismatch":
                mismatch,

            "Decision_agreement":
                (
                    1.0 - mismatch
                    if np.isfinite(
                        mismatch
                    )
                    else np.nan
                ),

            "Num_accuracy_resolutions":
                int(
                    acc_df[
                        "Resolution"
                    ].nunique()
                ),

            "Num_canonical_pairs":
                int(
                    pair_df[
                        "Pair"
                    ].nunique()
                ),
        })

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# A1 VS A0
# ============================================================

def build_a1_vs_a0(
    case_summary,
):
    rows = []

    for dataset in DATASETS:

        for backbone in BACKBONES:

            a0 = case_summary[
                (
                    case_summary[
                        "Dataset"
                    ]
                    == dataset
                )
                &
                (
                    case_summary[
                        "Backbone"
                    ]
                    == backbone
                )
                &
                (
                    case_summary[
                        "Method"
                    ]
                    == "LACE-A0"
                )
            ]

            a1 = case_summary[
                (
                    case_summary[
                        "Dataset"
                    ]
                    == dataset
                )
                &
                (
                    case_summary[
                        "Backbone"
                    ]
                    == backbone
                )
                &
                (
                    case_summary[
                        "Method"
                    ]
                    == "LACE-A1"
                )
            ]

            if (
                a0.empty
                or a1.empty
            ):
                continue

            a0 = a0.iloc[0]
            a1 = a1.iloc[0]

            direction = a0[
                "Decision_quality_direction"
            ]

            accuracy_improvement = (
                safe_improvement_pct(
                    a0["Accuracy_NMSE"],
                    a1["Accuracy_NMSE"],
                )
            )

            coherence_improvement = (
                safe_improvement_pct(
                    a0["Coherence_Crs"],
                    a1["Coherence_Crs"],
                )
            )

            mismatch_improvement = (
                safe_improvement_pct(
                    a0[
                        "Decision_mismatch"
                    ],
                    a1[
                        "Decision_mismatch"
                    ],
                )
            )

            quality_improvement = (
                quality_improvement_pct(
                    a0[
                        "Decision_quality"
                    ],
                    a1[
                        "Decision_quality"
                    ],
                    direction,
                )
            )

            meta = DATASET_META[
                dataset
            ]

            rows.append({
                "Dataset":
                    dataset,

                "Domain":
                    meta["Domain"],

                "Hierarchy_group":
                    meta[
                        "Hierarchy_group"
                    ],

                "Backbone":
                    backbone,

                "A0_Accuracy_NMSE":
                    a0[
                        "Accuracy_NMSE"
                    ],

                "A1_Accuracy_NMSE":
                    a1[
                        "Accuracy_NMSE"
                    ],

                "Accuracy_improvement_pct":
                    accuracy_improvement,

                "A0_Coherence_Crs":
                    a0[
                        "Coherence_Crs"
                    ],

                "A1_Coherence_Crs":
                    a1[
                        "Coherence_Crs"
                    ],

                "Coherence_improvement_pct":
                    coherence_improvement,

                "A0_Decision_mismatch":
                    a0[
                        "Decision_mismatch"
                    ],

                "A1_Decision_mismatch":
                    a1[
                        "Decision_mismatch"
                    ],

                "Decision_mismatch_improvement_pct":
                    mismatch_improvement,

                # Same quantity, named for the Figure-1 panel.
                "Decision_agreement_improvement_pct":
                    mismatch_improvement,

                "A0_Decision_quality":
                    a0[
                        "Decision_quality"
                    ],

                "A1_Decision_quality":
                    a1[
                        "Decision_quality"
                    ],

                "Decision_quality_direction":
                    direction,

                "Decision_quality_metric":
                    a0[
                        "Decision_quality_metric"
                    ],

                "Decision_quality_improvement_pct":
                    quality_improvement,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# FRONTIER
# ============================================================

def build_frontier(
    case_summary,
):
    """
    Seven TSFM-family methods only.

    Relative coordinates:
        x = accuracy improvement vs Independent
        y = coherence improvement vs Independent

    Positive/right/up = better.
    """

    tsfm = case_summary[
        case_summary[
            "Method"
        ].isin(
            TSFM_METHODS
        )
    ].copy()

    rows = []

    for (
        dataset,
        backbone,
    ), group in tsfm.groupby(
        [
            "Dataset",
            "Backbone",
        ],
        sort=False,
    ):

        independent = group[
            group["Method"]
            == "Independent"
        ]

        if independent.empty:
            continue

        independent = (
            independent.iloc[0]
        )

        ind_accuracy = (
            independent[
                "Accuracy_NMSE"
            ]
        )

        ind_coherence = (
            independent[
                "Coherence_Crs"
            ]
        )

        for _, row in group.iterrows():

            method = row[
                "Method"
            ]

            if method == "Independent":
                accuracy_change = 0.0
                coherence_change = 0.0
            else:
                accuracy_change = (
                    safe_improvement_pct(
                        ind_accuracy,
                        row[
                            "Accuracy_NMSE"
                        ],
                    )
                )

                coherence_change = (
                    safe_improvement_pct(
                        ind_coherence,
                        row[
                            "Coherence_Crs"
                        ],
                    )
                )

            rows.append({
                "Dataset":
                    dataset,

                "Domain":
                    row["Domain"],

                "Hierarchy_group":
                    row[
                        "Hierarchy_group"
                    ],

                "Backbone":
                    backbone,

                "Method":
                    method,

                "Accuracy_NMSE":
                    row[
                        "Accuracy_NMSE"
                    ],

                "Coherence_Crs":
                    row[
                        "Coherence_Crs"
                    ],

                "Independent_Accuracy_NMSE":
                    ind_accuracy,

                "Independent_Coherence_Crs":
                    ind_coherence,

                "Accuracy_improvement_vs_Independent_pct":
                    accuracy_change,

                "Coherence_improvement_vs_Independent_pct":
                    coherence_change,
            })

    return pd.DataFrame(
        rows
    )


def build_frontier_grouped(
    frontier,
):
    """
    One row per:
        hierarchy group × backbone × method

    Road:
        sensor cases are computed first in FINAL_frontier,
        then sensors are equal-weight averaged here.
    """

    value_cols = [
        "Accuracy_NMSE",
        "Coherence_Crs",
        "Independent_Accuracy_NMSE",
        "Independent_Coherence_Crs",
        "Accuracy_improvement_vs_Independent_pct",
        "Coherence_improvement_vs_Independent_pct",
    ]

    grouped = (
        frontier
        .groupby(
            [
                "Hierarchy_group",
                "Backbone",
                "Method",
            ],
            as_index=False,
            sort=False,
        )[value_cols]
        .mean()
    )

    return grouped


# ============================================================
# SORTING
# ============================================================

def apply_ordering(
    df,
):
    if df.empty:
        return df

    result = df.copy()

    if "Dataset" in result.columns:
        result["Dataset"] = (
            pd.Categorical(
                result["Dataset"],
                categories=DATASETS,
                ordered=True,
            )
        )

    if "Backbone" in result.columns:

        backbone_order = (
            list(
                BACKBONES.keys()
            )
            + [
                "DLinear",
                "LSTM",
            ]
        )

        result["Backbone"] = (
            pd.Categorical(
                result["Backbone"],
                categories=
                    backbone_order,
                ordered=True,
            )
        )

    if "Method" in result.columns:
        result["Method"] = (
            pd.Categorical(
                result["Method"],
                categories=
                    METHOD_ORDER,
                ordered=True,
            )
        )

    sort_cols = [
        col
        for col in [
            "Dataset",
            "Backbone",
            "Method",
            "Fine_factor",
            "Coarse_factor",
            "Factor",
        ]
        if col in result.columns
    ]

    if sort_cols:
        result = (
            result
            .sort_values(
                sort_cols
            )
            .reset_index(
                drop=True
            )
        )

    # Save labels as strings rather than pandas category internals.
    for col in [
        "Dataset",
        "Backbone",
        "Method",
    ]:
        if (
            col in result.columns
            and isinstance(
                result[col].dtype,
                pd.CategoricalDtype,
            )
        ):
            result[col] = (
                result[col]
                .astype(str)
            )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical final result matrices "
            "for LACE paper plotting."
        )
    )

    parser.add_argument(
        "--road-data-root",
        default="./data/Road",
    )

    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            EVALUATION_RESULTS_PATH,
            "Final_Plotting",
        ),
    )

    parser.add_argument(
        "--allow-missing",
        action="store_true",
    )

    parser.add_argument(
        "--skip-supervised",
        action="store_true",
        help=(
            "Skip DLinear/LSTM supervised references. "
            "The seven-method TSFM frontier is unaffected."
        ),
    )

    args = parser.parse_args()

    Path(
        args.output_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("FINAL LACE / LACE PLOTTING COLLECTOR")
    print("=" * 80)

    print(
        "RESULT_PHASE0_DIR:",
        RESULT_PHASE0_DIR,
    )

    print(
        "EVALUATION_RESULTS_PATH:",
        EVALUATION_RESULTS_PATH,
    )

    print(
        "Output:",
        args.output_dir,
    )

    # --------------------------------------------------------
    # SEVEN-METHOD TSFM FAMILY
    # --------------------------------------------------------

    (
        tsfm_pairwise,
        tsfm_accuracy,
        status,
    ) = collect_tsfm_family(
        road_data_root=
            args.road_data_root,
        allow_missing=
            args.allow_missing,
    )

    # --------------------------------------------------------
    # SUPERVISED ABSOLUTE REFERENCES
    # --------------------------------------------------------

    if args.skip_supervised:

        supervised_pairwise = (
            pd.DataFrame()
        )

        supervised_accuracy = (
            pd.DataFrame()
        )

    else:

        (
            supervised_pairwise,
            supervised_accuracy,
        ) = load_supervised_references(
            road_data_root=
                args.road_data_root,
            allow_missing=
                args.allow_missing,
        )

    # --------------------------------------------------------
    # FINAL RAW MATRICES
    # --------------------------------------------------------

    pairwise = pd.concat(
        [
            tsfm_pairwise,
            supervised_pairwise,
        ],
        ignore_index=True,
        sort=False,
    )

    accuracy = pd.concat(
        [
            tsfm_accuracy,
            supervised_accuracy,
        ],
        ignore_index=True,
        sort=False,
    )

    validate_pairwise(
        pairwise
    )

    validate_accuracy(
        accuracy
    )

    pairwise = apply_ordering(
        pairwise
    )

    accuracy = apply_ordering(
        accuracy
    )

    # --------------------------------------------------------
    # CASE SUMMARY
    # --------------------------------------------------------

    case_summary = (
        build_case_summary(
            pairwise,
            accuracy,
        )
    )

    case_summary = (
        apply_ordering(
            case_summary
        )
    )

    # --------------------------------------------------------
    # A1 VS A0
    # --------------------------------------------------------

    a1_vs_a0 = (
        build_a1_vs_a0(
            case_summary
        )
    )

    a1_vs_a0 = (
        apply_ordering(
            a1_vs_a0
        )
    )

    # --------------------------------------------------------
    # FRONTIER
    # --------------------------------------------------------

    frontier = (
        build_frontier(
            case_summary
        )
    )

    frontier = (
        apply_ordering(
            frontier
        )
    )

    frontier_grouped = (
        build_frontier_grouped(
            frontier
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    outputs = {
        "FINAL_pairwise.csv":
            pairwise,

        "FINAL_resolution_accuracy.csv":
            accuracy,

        "FINAL_case_summary.csv":
            case_summary,

        "FINAL_A1_vs_A0.csv":
            a1_vs_a0,

        "FINAL_frontier.csv":
            frontier,

        "FINAL_frontier_grouped.csv":
            frontier_grouped,

        "FINAL_collection_status.csv":
            status,
    }

    for filename, table in (
        outputs.items()
    ):

        path = os.path.join(
            args.output_dir,
            filename,
        )

        table.to_csv(
            path,
            index=False,
        )

        print(
            f"Saved {filename:35s} "
            f"rows={len(table)}"
        )

    # --------------------------------------------------------
    # SANITY REPORT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SANITY CHECK")
    print("=" * 80)

    print(
        "\nTSFM collection status:"
    )

    print(
        status[
            [
                "Dataset",
                "Backbone",
                "Method",
                "Status",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nA1 vs A0 canonical summary:"
    )

    print(
        a1_vs_a0[
            [
                "Dataset",
                "Backbone",
                "Accuracy_improvement_pct",
                "Coherence_improvement_pct",
                "Decision_quality_improvement_pct",
                "Decision_agreement_improvement_pct",
            ]
        ].round(
            3
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("EXPECTED FULL-MATRIX COUNTS")
    print("=" * 80)

    print(
        "Seven-method TSFM frontier rows: "
        "168 "
        "(6 datasets × 4 backbones × 7 methods)"
    )

    print(
        "A1-vs-A0 rows: "
        "24 "
        "(6 datasets × 4 backbones)"
    )

    if len(frontier) != 168:
        print(
            f"[CHECK] FINAL_frontier has "
            f"{len(frontier)} rows, expected 168 "
            f"for a complete matrix."
        )

    if len(a1_vs_a0) != 24:
        print(
            f"[CHECK] FINAL_A1_vs_A0 has "
            f"{len(a1_vs_a0)} rows, expected 24 "
            f"for a complete matrix."
        )

    print()
    print(
        "DONE. These files are now the canonical "
        "inputs for final paper plotting."
    )


if __name__ == "__main__":
    main()
