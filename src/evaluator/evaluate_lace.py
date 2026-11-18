"""Evaluate LACE forecast accuracy, cross-resolution consistency, and decision quality."""
import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.config import EVALUATION_RESULTS_PATH


# ============================================================
# REUSABLE LACE EVALUATOR
# ============================================================
#
# Shared across:
#   - c_traffic
#   - k_traffic
#   - ETTm1_HUFL
#   - road_sensor_<id>
#
# Dataset-specific pieces live only in the configuration /
# decision dispatch below.
#
# Forecast-origin alignment:
# Every head is produced from the SAME base-resolution forecast
# origin. Therefore all factors use one common daily stride:
# the one-day horizon length at the BASE resolution.
# ============================================================


MODE_LABELS = {
    "none": "LACE_A0",
    "structural": "LACE_A1",
}


STATIC_DATASET_CONFIGS = {
    # --------------------------------------------------------
    # Network traffic: 10m -> 1h -> 6h -> 1d
    # --------------------------------------------------------
    "c_traffic": {
        "model_dir": "LACE_moirai1",
        "base_resolution": "10m",
        "factor_to_res": {
            1: "10m",
            6: "1h",
            36: "6h",
            144: "1d",
        },
        "daily_stride": 144,
        "pairs": [
            (1, 6),
            (6, 36),
            (36, 144),
            (1, 36),
            (1, 144),
        ],
        "decision": "network_provisioning",
        "decision_pairs": None,  # all evaluation pairs
        "decision_params": {
            "alpha": 2.0,
        },
        "decision_ylabel": "Provisioning-state disagreement",
        "delta_d_ylabel": "Provisioning-cost advantage",
    },

    # --------------------------------------------------------
    # Network traffic: 5m -> 15m -> 1h -> 1d
    # --------------------------------------------------------
    "k_traffic": {
        "model_dir": "LACE_moirai1",
        "base_resolution": "5m",
        "factor_to_res": {
            1: "5m",
            3: "15m",
            12: "1h",
            288: "1d",
        },
        "daily_stride": 288,
        "pairs": [
            (1, 3),
            (3, 12),
            (12, 288),
            (1, 12),
            (1, 288),
        ],
        "decision": "network_provisioning",
        "decision_pairs": None,  # all evaluation pairs
        "decision_params": {
            "alpha": 2.0,
        },
        "decision_ylabel": "Provisioning-state disagreement",
        "delta_d_ylabel": "Provisioning-cost advantage",
    },

    # --------------------------------------------------------
    # Electricity: 15m -> 1h -> 6h -> 1d
    # --------------------------------------------------------
    "ETTm1_HUFL": {
        "model_dir": "LACE_moirai1",
        "base_resolution": "15m",
        "factor_to_res": {
            1: "15m",
            4: "1h",
            24: "6h",
            96: "1d",
        },
        "daily_stride": 96,
        "pairs": [
            (1, 4),
            (4, 24),
            (24, 96),
            (1, 24),
            (1, 96),
        ],
        "decision": "energy_scheduling",
        "decision_pairs": None,  # all evaluation pairs
        "decision_params": {},
        "decision_ylabel": "Scheduling disagreement",
        "delta_d_ylabel": "Decision route advantage",
    },
}


# ============================================================
# DATASET CONFIGURATION
# ============================================================

def make_road_config(dataset, road_data_root="./data/Road"):
    """
    Dynamic configuration for road_sensor_<id>.

    Phase-0 road hierarchy:
        5m -> 15m -> 1h -> 1d

    Operational decision is defined only for:
        5m  -> 15m
        15m -> 1h
        5m  -> 1h

    Daily forecasts are intentionally excluded from the road
    capacity-pressure decision because that decision is hourly.
    """
    match = re.fullmatch(r"road_sensor_(\d+)", dataset)

    if match is None:
        raise KeyError(
            f"Unknown dataset '{dataset}'. Expected one of "
            f"{list(STATIC_DATASET_CONFIGS)} or road_sensor_<id>."
        )

    sensor = int(match.group(1))

    return {
        "model_dir": "LACE_moirai1",
        "base_resolution": "5m",
        "factor_to_res": {
            1: "5m",
            3: "15m",
            12: "1h",
            288: "1d",
        },
        "daily_stride": 288,
        "pairs": [
            (1, 3),
            (3, 12),
            (12, 288),
            (1, 12),
            (1, 288),
        ],
        "decision": "road_capacity",
        "decision_pairs": {
            (1, 3),
            (3, 12),
            (1, 12),
        },
        "decision_params": {
            "sensor": sensor,
            "data_root": road_data_root,
            "train_end": "2018-02-17 00:00:00",
            "capacity_percentile": 95.0,
            "threshold_fraction": 0.8,
        },
        "sensor": sensor,
        "decision_ylabel": "Hourly decision disagreement",
        "delta_d_ylabel": "Decision route advantage",
    }


def get_dataset_config(dataset, road_data_root="./data/Road"):
    if dataset in STATIC_DATASET_CONFIGS:
        # Return a shallow copy so runtime fields never mutate registry.
        cfg = dict(STATIC_DATASET_CONFIGS[dataset])
        cfg["factor_to_res"] = dict(cfg["factor_to_res"])
        cfg["pairs"] = list(cfg["pairs"])
        cfg["decision_params"] = dict(cfg.get("decision_params", {}))
        return cfg

    return make_road_config(
        dataset,
        road_data_root=road_data_root,
    )


# ============================================================
# GENERIC NUMERICAL HELPERS
# ============================================================

def squeeze_last_dim(x):
    """
    [N,H,1] -> [N,H]
    Preserve [N,H].
    """
    x = np.asarray(x)

    if x.ndim == 3:
        if x.shape[-1] != 1:
            raise ValueError(
                f"Expected singleton last dimension, got {x.shape}."
            )
        return x[..., 0]

    if x.ndim != 2:
        raise ValueError(
            f"Expected [N,H] or [N,H,1], got {x.shape}."
        )

    return x


def aggregate_sum(x, ratio):
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError(
            f"aggregate_sum expects [N,H], got {x.shape}."
        )

    n, h = x.shape

    if h % ratio != 0:
        raise ValueError(
            f"H={h} is not divisible by aggregation ratio={ratio}."
        )

    return x.reshape(
        n,
        h // ratio,
        ratio,
    ).sum(axis=2)


def nmse(pred, true, eps=1e-12):
    pred = np.asarray(pred)
    true = np.asarray(true)

    mse = np.mean(
        (pred - true) ** 2
    )

    return mse / (
        np.var(true) + eps
    )


def cross_resolution_consistency(
    agg_pred,
    direct_pred,
    eps=1e-12,
):
    """
    Relative L1 discrepancy between:
        aggregate(fine forecast)
    and
        direct coarse forecast.
    """
    agg_pred = np.asarray(agg_pred)
    direct_pred = np.asarray(direct_pred)

    return (
        np.abs(
            agg_pred - direct_pred
        ).sum()
        /
        (
            np.abs(agg_pred).sum()
            + eps
        )
    )


# ============================================================
# DECISION EVALUATION: NETWORK
# ============================================================

def network_provisioning_metrics(
    capacity_pred,
    true_demand,
    alpha=2.0,
    eps=1e-12,
):
    """
    DeepCog-inspired network provisioning evaluation.

    Forecast is interpreted as provisioned capacity.

    Cost:
        normalized overprovisioning + alpha * SLA violation rate
    """
    capacity_pred = np.asarray(capacity_pred)
    true_demand = np.asarray(true_demand)

    # Capacity cannot physically be negative.
    capacity = np.maximum(
        capacity_pred,
        0.0,
    )

    violation = (
        capacity < true_demand
    )

    sla_rate = violation.mean()

    excess = np.maximum(
        capacity - true_demand,
        0.0,
    )

    overprovision = (
        excess.sum()
        /
        (
            np.abs(true_demand).sum()
            + eps
        )
    )

    cost = (
        overprovision
        + alpha * sla_rate
    )

    return {
        "SLA_rate": float(sla_rate),
        "Overprovision": float(overprovision),
        "Cost": float(cost),
        "Decision_state": violation,
    }


def network_decision_routes(
    agg_pred,
    direct_pred,
    true_coarse,
    params,
):
    agg = network_provisioning_metrics(
        agg_pred,
        true_coarse,
        **params,
    )

    direct = network_provisioning_metrics(
        direct_pred,
        true_coarse,
        **params,
    )

    mismatch = np.mean(
        agg["Decision_state"]
        !=
        direct["Decision_state"]
    )

    return {
        "Agg_SLA_rate":
            agg["SLA_rate"],

        "Direct_SLA_rate":
            direct["SLA_rate"],

        "Agg_overprovision":
            agg["Overprovision"],

        "Direct_overprovision":
            direct["Overprovision"],

        "Agg_decision_cost":
            agg["Cost"],

        "Direct_decision_cost":
            direct["Cost"],

        "Decision_mismatch":
            float(mismatch),

        # Positive = aggregated route has lower cost.
        "Delta_D":
            direct["Cost"] - agg["Cost"],
    }


# ============================================================
# DECISION EVALUATION: ELECTRICITY
# ============================================================

def energy_scheduling_routes(
    agg_pred,
    direct_pred,
    true_coarse,
    eps=1e-12,
):
    """
    Phase-0 ETT decision:
    total scheduled energy over each one-day forecast window.

    Decision error is normalized absolute error in daily scheduled
    energy. Decision mismatch is the normalized difference between
    aggregate-route and direct-route scheduled energy.
    """
    d_agg = np.asarray(agg_pred).sum(axis=1)
    d_dir = np.asarray(direct_pred).sum(axis=1)
    d_true = np.asarray(true_coarse).sum(axis=1)

    denom = (
        np.abs(d_true).sum()
        + eps
    )

    agg_error = (
        np.abs(d_agg - d_true).sum()
        / denom
    )

    dir_error = (
        np.abs(d_dir - d_true).sum()
        / denom
    )

    mismatch = (
        np.abs(d_agg - d_dir).sum()
        / denom
    )

    return {
        "Agg_decision_error":
            float(agg_error),

        "Direct_decision_error":
            float(dir_error),

        "Decision_mismatch":
            float(mismatch),

        # Positive = aggregated route has lower decision error.
        "Delta_D":
            float(dir_error - agg_error),
    }


# ============================================================
# DECISION EVALUATION: ROAD
# ============================================================

def compute_road_threshold(params):
    """
    Reproduce the Phase-0 road threshold exactly:

      1. use 5-minute training data before 2018-02-17
      2. sum to hourly traffic volume
      3. Q_cap = 95th percentile of hourly training volume
      4. threshold = 0.8 * Q_cap
    """
    sensor = params["sensor"]
    data_root = params["data_root"]
    train_end = pd.Timestamp(
        params["train_end"]
    )

    csv_path = os.path.join(
        data_root,
        f"road_sensor_{sensor}_5m.csv",
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Road threshold source not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    required = {
        "date",
        "value",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path} is missing columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    train = df[
        df["date"] < train_end
    ].copy()

    if train.empty:
        raise RuntimeError(
            f"No road training rows before {train_end} in {csv_path}."
        )

    train = train.set_index(
        "date"
    )

    hourly_train = (
        train["value"]
        .resample("1h")
        .sum()
        .dropna()
    )

    if hourly_train.empty:
        raise RuntimeError(
            f"No hourly road training values could be built from {csv_path}."
        )

    q_cap = np.percentile(
        hourly_train,
        params["capacity_percentile"],
    )

    threshold = (
        params["threshold_fraction"]
        * q_cap
    )

    return float(q_cap), float(threshold)


def road_to_hourly(
    volume,
    resolution,
):
    if resolution == "5m":
        return aggregate_sum(
            volume,
            12,
        )

    if resolution == "15m":
        return aggregate_sum(
            volume,
            4,
        )

    if resolution == "1h":
        return np.asarray(volume)

    raise ValueError(
        "Daily forecasts cannot be converted "
        "to hourly road operational decisions."
    )


def road_capacity_decision(
    volume,
    resolution,
    threshold_hourly,
):
    """
    Binary hourly capacity-pressure decision.

    Output:
        [N_days, 24]
    """
    hourly = road_to_hourly(
        volume,
        resolution,
    )

    return (
        hourly > threshold_hourly
    ).astype(int)


def road_decision_routes(
    agg_pred,
    direct_pred,
    true_coarse,
    coarse_resolution,
    threshold_hourly,
):
    d_agg = road_capacity_decision(
        agg_pred,
        coarse_resolution,
        threshold_hourly,
    )

    d_dir = road_capacity_decision(
        direct_pred,
        coarse_resolution,
        threshold_hourly,
    )

    d_true = road_capacity_decision(
        true_coarse,
        coarse_resolution,
        threshold_hourly,
    )

    if not (
        d_agg.shape
        == d_dir.shape
        == d_true.shape
    ):
        raise RuntimeError(
            "Road decision shapes do not match: "
            f"agg={d_agg.shape}, dir={d_dir.shape}, true={d_true.shape}."
        )

    agg_accuracy = np.mean(
        d_agg == d_true
    )

    direct_accuracy = np.mean(
        d_dir == d_true
    )

    mismatch = np.mean(
        d_agg != d_dir
    )

    return {
        "Agg_decision_accuracy":
            float(agg_accuracy),

        "Direct_decision_accuracy":
            float(direct_accuracy),

        "Decision_mismatch":
            float(mismatch),

        # Positive = aggregated route has higher decision accuracy.
        "Delta_D":
            float(
                agg_accuracy
                - direct_accuracy
            ),
    }


# ============================================================
# DECISION DISPATCH
# ============================================================

def unavailable_decision_result(decision_name):
    """
    Preserve accuracy/coherence rows for pairs where the operational
    decision is intentionally undefined.
    """
    if decision_name == "network_provisioning":
        return {
            "Agg_SLA_rate": np.nan,
            "Direct_SLA_rate": np.nan,
            "Agg_overprovision": np.nan,
            "Direct_overprovision": np.nan,
            "Agg_decision_cost": np.nan,
            "Direct_decision_cost": np.nan,
            "Decision_mismatch": np.nan,
            "Delta_D": np.nan,
        }

    if decision_name == "energy_scheduling":
        return {
            "Agg_decision_error": np.nan,
            "Direct_decision_error": np.nan,
            "Decision_mismatch": np.nan,
            "Delta_D": np.nan,
        }

    if decision_name == "road_capacity":
        return {
            "Agg_decision_accuracy": np.nan,
            "Direct_decision_accuracy": np.nan,
            "Decision_mismatch": np.nan,
            "Delta_D": np.nan,
        }

    raise KeyError(
        f"Unknown decision type: {decision_name}"
    )


def evaluate_decision_routes(
    cfg,
    fine,
    coarse,
    agg_pred,
    direct_pred,
    true_coarse,
    road_threshold=None,
):
    decision_name = cfg["decision"]
    decision_pairs = cfg.get(
        "decision_pairs"
    )

    if (
        decision_pairs is not None
        and (fine, coarse) not in decision_pairs
    ):
        return unavailable_decision_result(
            decision_name
        )

    if decision_name == "network_provisioning":
        return network_decision_routes(
            agg_pred,
            direct_pred,
            true_coarse,
            cfg.get(
                "decision_params",
                {},
            ),
        )

    if decision_name == "energy_scheduling":
        return energy_scheduling_routes(
            agg_pred,
            direct_pred,
            true_coarse,
        )

    if decision_name == "road_capacity":
        if road_threshold is None:
            raise RuntimeError(
                "Road decision evaluation requires a threshold."
            )

        coarse_resolution = (
            cfg["factor_to_res"][coarse]
        )

        return road_decision_routes(
            agg_pred,
            direct_pred,
            true_coarse,
            coarse_resolution,
            road_threshold,
        )

    raise NotImplementedError(
        f"Decision evaluator '{decision_name}' is not implemented."
    )


# ============================================================
# FILE LOADING / DAILY ALIGNMENT
# ============================================================

def get_evaluation_folder(
    dataset,
    mode,
    cfg,
):
    return os.path.join(
        EVALUATION_RESULTS_PATH,
        cfg["model_dir"],
        dataset,
        cfg["base_resolution"],
        mode,
    )


def load_lace_outputs(
    folder,
    factor_to_res,
):
    preds = {}
    trues = {}

    for factor, resolution in factor_to_res.items():

        pred_path = os.path.join(
            folder,
            f"pred_factor_{factor}.npy",
        )

        true_path = os.path.join(
            folder,
            f"true_factor_{factor}.npy",
        )

        if not os.path.exists(pred_path):
            raise FileNotFoundError(
                f"Missing prediction file: {pred_path}"
            )

        if not os.path.exists(true_path):
            raise FileNotFoundError(
                f"Missing truth file: {true_path}"
            )

        pred = squeeze_last_dim(
            np.load(pred_path)
        )

        true = squeeze_last_dim(
            np.load(true_path)
        )

        if pred.shape != true.shape:
            raise RuntimeError(
                f"{resolution}: prediction shape {pred.shape} "
                f"!= truth shape {true.shape}."
            )

        preds[factor] = pred
        trues[factor] = true

        print(
            f"{resolution:>5s} | "
            f"factor={factor:<4d} | "
            f"pred={pred.shape} | "
            f"true={true.shape}"
        )

    return preds, trues


def recover_daily_windows(
    preds,
    trues,
    daily_stride,
):
    """
    LACE-specific alignment.

    All resolution heads correspond to the same base-resolution
    forecast origins, so use the SAME base daily stride for every
    factor.
    """
    available = {
        factor: len(
            true[::daily_stride]
        )
        for factor, true in trues.items()
    }

    n_days = min(
        available.values()
    )

    if n_days < 1:
        raise RuntimeError(
            "No common daily forecast windows were recovered."
        )

    daily_pred = {
        factor:
            pred[::daily_stride][:n_days]
        for factor, pred in preds.items()
    }

    daily_true = {
        factor:
            true[::daily_stride][:n_days]
        for factor, true in trues.items()
    }

    print(
        "\nAvailable common-day candidates:",
        available,
    )

    print(
        "Using common days:",
        n_days,
    )

    return daily_pred, daily_true


# ============================================================
# GROUND-TRUTH COHERENCE CHECK
# ============================================================

def verify_ground_truth_coherence(
    daily_true,
    pairs,
    factor_to_res,
):
    print(
        "\n=== GROUND-TRUTH COHERENCE ==="
    )

    for fine, coarse in pairs:

        if coarse % fine != 0:
            raise ValueError(
                f"Invalid hierarchy pair {fine}->{coarse}."
            )

        ratio = coarse // fine

        agg = aggregate_sum(
            daily_true[fine],
            ratio,
        )

        direct = daily_true[coarse]

        if agg.shape != direct.shape:
            raise RuntimeError(
                f"Truth shape mismatch for "
                f"{factor_to_res[fine]}"
                f"->{factor_to_res[coarse]}: "
                f"aggregate={agg.shape}, "
                f"direct={direct.shape}."
            )

        diff = (
            agg - direct
        )

        relative_l1 = (
            np.abs(diff).sum()
            /
            (
                np.abs(direct).sum()
                + 1e-12
            )
        )

        allclose = np.allclose(
            agg,
            direct,
            rtol=1e-5,
            atol=1e-5,
        )

        print(
            f"{factor_to_res[fine]}"
            f"->{factor_to_res[coarse]}: "
            f"relative_L1={relative_l1:.3e} | "
            f"max_abs={np.abs(diff).max():.6f} | "
            f"allclose={allclose}"
        )

        if not allclose:
            raise RuntimeError(
                "Ground-truth hierarchy is not coherent for "
                f"{factor_to_res[fine]}"
                f"->{factor_to_res[coarse]}."
            )


# ============================================================
# FULL DATASET / MODE EVALUATION
# ============================================================

def evaluate_dataset_mode(
    dataset,
    mode,
    road_data_root="./data/Road",
    model_dir=None,
):
    if mode not in MODE_LABELS:
        raise ValueError(
            f"Unsupported mode '{mode}'. "
            f"Choose from {list(MODE_LABELS)}."
        )

    cfg = get_dataset_config(
        dataset,
        road_data_root=road_data_root,
    )
    if model_dir is not None:
        cfg["model_dir"] = model_dir

    factor_to_res = cfg[
        "factor_to_res"
    ]

    pairs = cfg[
        "pairs"
    ]

    folder = get_evaluation_folder(
        dataset,
        mode,
        cfg,
    )

    print(
        "\n============================================================"
    )
    print(
        f"Dataset:  {dataset}"
    )
    print(
        f"Mode:     {mode}"
    )
    print(
    f"Backbone: {cfg['model_dir']}"
    )
    print(
        f"Label:    {MODE_LABELS[mode]}"
    )
    print(
        f"Hierarchy: "
        + " -> ".join(
            factor_to_res[f]
            for f in factor_to_res
        )
    )
    print(
        f"Folder:   {folder}"
    )
    print(
        "============================================================\n"
    )

    preds, trues = load_lace_outputs(
        folder,
        factor_to_res,
    )

    daily_pred, daily_true = recover_daily_windows(
        preds,
        trues,
        cfg["daily_stride"],
    )

    verify_ground_truth_coherence(
        daily_true,
        pairs,
        factor_to_res,
    )

    road_q_cap = None
    road_threshold = None

    if cfg["decision"] == "road_capacity":
        road_q_cap, road_threshold = (
            compute_road_threshold(
                cfg["decision_params"]
            )
        )

        print(
            "\n=== ROAD DECISION THRESHOLD ==="
        )
        print(
            f"Sensor {cfg['sensor']}: "
            f"Qcap={road_q_cap:.2f} veh/h | "
            f"threshold={road_threshold:.2f} veh/h"
        )

    rows = []

    for fine, coarse in pairs:

        ratio = (
            coarse // fine
        )

        fine_pred = daily_pred[
            fine
        ]

        fine_true = daily_true[
            fine
        ]

        direct_pred = daily_pred[
            coarse
        ]

        coarse_true = daily_true[
            coarse
        ]

        agg_pred = aggregate_sum(
            fine_pred,
            ratio,
        )

        if agg_pred.shape != coarse_true.shape:
            raise RuntimeError(
                f"Aggregated prediction shape mismatch for "
                f"{factor_to_res[fine]}"
                f"->{factor_to_res[coarse]}: "
                f"{agg_pred.shape} vs {coarse_true.shape}."
            )

        if direct_pred.shape != coarse_true.shape:
            raise RuntimeError(
                f"Direct prediction shape mismatch for "
                f"{factor_to_res[coarse]}: "
                f"{direct_pred.shape} vs {coarse_true.shape}."
            )

        e_fine = nmse(
            fine_pred,
            fine_true,
        )

        e_agg = nmse(
            agg_pred,
            coarse_true,
        )

        e_dir = nmse(
            direct_pred,
            coarse_true,
        )

        c_rs = cross_resolution_consistency(
            agg_pred,
            direct_pred,
        )

        pair_name = (
            f"{factor_to_res[fine]}"
            f"->{factor_to_res[coarse]}"
        )

        row = {
            "Dataset": dataset,
            "Backbone": cfg["model_dir"],
            "Mode": mode,
            "Model": MODE_LABELS[mode],
            "Pair": pair_name,
            "Fine_factor": fine,
            "Coarse_factor": coarse,
            "E_fine": e_fine,
            "E_agg": e_agg,
            "E_dir": e_dir,
            "C_rs": c_rs,

            # Positive = aggregate route has lower forecast error.
            "Delta_E": e_dir - e_agg,
        }

        row[
            "Abs_Delta_E"
        ] = abs(
            row["Delta_E"]
        )

        if "sensor" in cfg:
            row["Sensor"] = cfg[
                "sensor"
            ]

            row["Q_cap_veh_per_h"] = (
                road_q_cap
            )

            row["Decision_threshold_veh_per_h"] = (
                road_threshold
            )

        decision_result = evaluate_decision_routes(
            cfg=cfg,
            fine=fine,
            coarse=coarse,
            agg_pred=agg_pred,
            direct_pred=direct_pred,
            true_coarse=coarse_true,
            road_threshold=road_threshold,
        )

        row.update(
            decision_result
        )

        rows.append(
            row
        )

    full_table = pd.DataFrame(
        rows
    )

    output_csv = os.path.join(
        folder,
        f"{dataset}_{mode}_evaluation.csv",
    )

    full_table.to_csv(
        output_csv,
        index=False,
    )

    print(
        "\n=== EVALUATION RESULTS ==="
    )

    print(
        full_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        "\nSaved:",
        output_csv,
    )

    return (
        full_table,
        folder,
        cfg,
    )


# ============================================================
# RELATIONSHIP PLOT
# ============================================================

def safe_spearman(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    finite = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    if len(x) < 3:
        return (
            np.nan,
            np.nan,
        )

    if np.allclose(
        x,
        x[0],
    ):
        return (
            np.nan,
            np.nan,
        )

    if np.allclose(
        y,
        y[0],
    ):
        return (
            np.nan,
            np.nan,
        )

    rho, p = spearmanr(
        x,
        y,
    )

    return (
        float(rho),
        float(p),
    )


def plot_relationships(
    full_table,
    output_folder,
    dataset,
    mode,
    cfg,
):
    """
    Plot only rows for which the dataset's operational decision
    is defined. This automatically excludes the two daily road
    pairs while keeping them in the saved numerical CSV.
    """
    plot_df = full_table.dropna(
        subset=[
            "Decision_mismatch",
            "Delta_D",
        ]
    ).copy()

    if plot_df.empty:
        print(
            "No rows have decision metrics; skipping relationship plot."
        )
        return

    print(
        f"\nRelationship plot points used: {len(plot_df)}"
    )

    marker_cycle = [
        "o",
        "s",
        "^",
        "D",
        "X",
        "P",
        "v",
        "<",
        ">",
    ]

    pairs = list(
        plot_df["Pair"].unique()
    )

    marker_map = {
        pair:
            marker_cycle[
                i % len(marker_cycle)
            ]
        for i, pair in enumerate(
            pairs
        )
    }

    def scatter_panel(
        ax,
        x_col,
        y_col,
    ):
        for _, row in plot_df.iterrows():
            ax.scatter(
                row[x_col],
                row[y_col],
                marker=marker_map[
                    row["Pair"]
                ],
                s=85,
                alpha=0.85,
            )

    def add_spearman(
        ax,
        x_col,
        y_col,
    ):
        rho, p = safe_spearman(
            plot_df[x_col],
            plot_df[y_col],
        )

        if np.isnan(rho):
            text = (
                r"$\rho_s$ unavailable"
            )
        else:
            if p < 0.0001:
                p_text = (
                    r"$p<0.0001$"
                )
            else:
                p_text = (
                    rf"$p={p:.4f}$"
                )

            text = (
                rf"$\rho_s={rho:.2f}$"
                + "\n"
                + p_text
            )

        ax.text(
            0.96,
            0.95,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
        )

        return (
            rho,
            p,
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            4.6,
        ),
    )

    # --------------------------------------------------------
    # A: inconsistency -> accuracy sensitivity
    # --------------------------------------------------------
    ax = axes[0]

    scatter_panel(
        ax,
        "C_rs",
        "Abs_Delta_E",
    )

    rho_a, p_a = add_spearman(
        ax,
        "C_rs",
        "Abs_Delta_E",
    )

    ax.set_xlabel(
        r"Forecast inconsistency $C_{r,s}$"
    )

    ax.set_ylabel(
        r"Route accuracy sensitivity $|\Delta E|$"
    )

    ax.set_title(
        "(a) Inconsistency vs. accuracy sensitivity"
    )

    ax.grid(
        alpha=0.2
    )

    # --------------------------------------------------------
    # B: inconsistency -> decision disagreement
    # --------------------------------------------------------
    ax = axes[1]

    scatter_panel(
        ax,
        "C_rs",
        "Decision_mismatch",
    )

    rho_b, p_b = add_spearman(
        ax,
        "C_rs",
        "Decision_mismatch",
    )

    ax.set_xlabel(
        r"Forecast inconsistency $C_{r,s}$"
    )

    ax.set_ylabel(
        cfg[
            "decision_ylabel"
        ]
    )

    ax.set_title(
        "(b) Inconsistency vs. decision disagreement"
    )

    ax.grid(
        alpha=0.2
    )

    # --------------------------------------------------------
    # C: forecast route advantage -> decision route advantage
    # --------------------------------------------------------
    ax = axes[2]

    scatter_panel(
        ax,
        "Delta_E",
        "Delta_D",
    )

    rho_c, p_c = add_spearman(
        ax,
        "Delta_E",
        "Delta_D",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        r"Forecast route advantage $\Delta E$"
    )

    ax.set_ylabel(
        cfg[
            "delta_d_ylabel"
        ]
    )

    ax.set_title(
        "(c) Forecast vs. decision route advantage"
    )

    ax.grid(
        alpha=0.2
    )

    pair_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker_map[
                pair
            ],
            linestyle="",
            label=pair,
            markersize=8,
        )
        for pair in pairs
    ]

    fig.legend(
        handles=pair_handles,
        title="Resolution pair",
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.08,
        ),
        ncol=min(
            5,
            len(pairs),
        ),
        frameon=False,
    )

    non_ties = plot_df[
        (plot_df["Delta_E"] != 0)
        &
        (plot_df["Delta_D"] != 0)
    ]

    if len(non_ties) > 0:
        same_direction = (
            np.sign(
                non_ties["Delta_E"]
            )
            ==
            np.sign(
                non_ties["Delta_D"]
            )
        )

        agreement_text = (
            f"{same_direction.sum()}/"
            f"{len(same_direction)} "
            f"({100 * same_direction.mean():.1f}%)"
        )

    else:
        agreement_text = (
            "unavailable (all ties)"
        )

    print(
        "\n=== RELATIONSHIP DIAGNOSTICS ==="
    )

    print(
        f"A: C vs |Delta_E|          "
        f"rho={rho_a}, p={p_a}"
    )

    print(
        f"B: C vs decision mismatch "
        f"rho={rho_b}, p={p_b}"
    )

    print(
        f"C: Delta_E vs Delta_D     "
        f"rho={rho_c}, p={p_c}"
    )

    print(
        "Forecast/decision route agreement "
        f"(excluding ties): {agreement_text}"
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
        ]
    )

    pdf_path = os.path.join(
        output_folder,
        f"{dataset}_{mode}_relationships.pdf",
    )

    png_path = os.path.join(
        output_folder,
        f"{dataset}_{mode}_relationships.png",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        pdf_path,
    )

    print(
        "Saved:",
        png_path,
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reusable LACE multi-resolution accuracy, coherence, "
            "decision, and relationship evaluator."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "c_traffic, k_traffic, ETTm1_HUFL, "
            "or road_sensor_<id>."
        ),
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "none",
            "structural",
        ],
    )

    parser.add_argument(
        "--model_dir",
        default="LACE_moirai1",
        choices=[
            "LACE_moirai1",
            "LACE_moirai2",
            "LACE_chronos2",
            "LACE_timesfm2",
        ],
        help=(
            "Model/result directory under EVALUATION_RESULTS_PATH. "
            "Defaults to LACE_moirai1."
        ),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Save the three-panel relationship diagnostic."
        ),
    )

    parser.add_argument(
        "--road_data_root",
        default="./data/Road",
        help=(
            "Directory containing road_sensor_<id>_5m.csv. "
            "Used only for road datasets."
        ),
    )

    args = parser.parse_args()

    full_table, folder, cfg = evaluate_dataset_mode(
        dataset=args.dataset,
        mode=args.mode,
        road_data_root=args.road_data_root,
        model_dir=args.model_dir,
    )

    if args.plot:
        plot_relationships(
            full_table=full_table,
            output_folder=folder,
            dataset=args.dataset,
            mode=args.mode,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
