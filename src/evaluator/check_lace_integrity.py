"""Check saved LACE forecasts, evaluation tables, logs, and checkpoints.

Validate shapes, finite values, forecast alignment, truth aggregation, and
evaluation metrics. Use --allow-missing while an experiment queue is running.

Run from the repository root:
    python -m src.evaluator.check_lace_integrity --help"""

from __future__ import annotations

import os
import re
import sys
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.config import EVALUATION_RESULTS_PATH
from src.evaluator.evaluate_lace import get_dataset_config


# ============================================================
# DEFAULT EXPERIMENT MATRIX
# ============================================================

DEFAULT_DATASETS = [
    "c_traffic",
    "k_traffic",
    "ETTm1_HUFL",
    "road_sensor_62",
    "road_sensor_103",
    "road_sensor_240",
]

DEFAULT_MODES = [
    "none",
    "structural",
]

MODE_LABELS = {
    "none": "A0",
    "structural": "A1",
}

EXPECTED_LAMBDA = {
    "none": 0.0,
    "structural": 1.0,
}

REQUIRED_CSV_COLUMNS = {
    "Dataset",
    "Mode",
    "Model",
    "Pair",
    "Fine_factor",
    "Coarse_factor",
    "E_fine",
    "E_agg",
    "E_dir",
    "C_rs",
    "Delta_E",
    "Abs_Delta_E",
    "Decision_mismatch",
    "Delta_D",
}

NUMERIC_CLOSE_RTOL = 1e-7
NUMERIC_CLOSE_ATOL = 1e-9
TRUTH_RTOL = 1e-5
TRUTH_ATOL = 1e-5


# ============================================================
# REPORTING
# ============================================================

@dataclass
class Check:
    dataset: str
    mode: str
    check: str
    status: str
    message: str = ""


class Reporter:
    def __init__(self):
        self.rows: List[Check] = []

    def add(
        self,
        dataset: str,
        mode: str,
        check: str,
        status: str,
        message: str = "",
    ):
        self.rows.append(
            Check(
                dataset=dataset,
                mode=mode,
                check=check,
                status=status,
                message=message,
            )
        )

    def pass_(
        self,
        dataset: str,
        mode: str,
        check: str,
        message: str = "",
    ):
        self.add(dataset, mode, check, "PASS", message)

    def warn(
        self,
        dataset: str,
        mode: str,
        check: str,
        message: str,
    ):
        self.add(dataset, mode, check, "WARNING", message)

    def error(
        self,
        dataset: str,
        mode: str,
        check: str,
        message: str,
    ):
        self.add(dataset, mode, check, "ERROR", message)

    def missing(
        self,
        dataset: str,
        mode: str,
        check: str,
        message: str,
    ):
        self.add(dataset, mode, check, "MISSING", message)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Dataset": r.dataset,
                    "Mode": r.mode,
                    "Label": MODE_LABELS.get(r.mode, r.mode),
                    "Check": r.check,
                    "Status": r.status,
                    "Message": r.message,
                }
                for r in self.rows
            ]
        )


# ============================================================
# INDEPENDENT NUMERICAL HELPERS
# ============================================================

def squeeze_saved_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim == 3:
        if x.shape[-1] != 1:
            raise ValueError(
                f"3-D saved array must have singleton final dimension; "
                f"got {x.shape}."
            )
        x = x[..., 0]

    if x.ndim != 2:
        raise ValueError(
            f"Expected saved array [N,H] or [N,H,1]; got {x.shape}."
        )

    return x


def aggregate_sum(x: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError(
            f"aggregate_sum expects [N,H]; got {x.shape}."
        )

    n, h = x.shape

    if h % factor != 0:
        raise ValueError(
            f"H={h} is not divisible by aggregation factor={factor}."
        )

    return x.reshape(
        n,
        h // factor,
        factor,
    ).sum(axis=2)


def nmse(
    pred: np.ndarray,
    true: np.ndarray,
    eps: float = 1e-12,
) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)

    mse = np.mean(
        (pred - true) ** 2
    )

    return float(
        mse
        /
        (
            np.var(true)
            + eps
        )
    )


def consistency(
    agg_pred: np.ndarray,
    direct_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    return float(
        np.abs(
            np.asarray(agg_pred)
            -
            np.asarray(direct_pred)
        ).sum()
        /
        (
            np.abs(
                np.asarray(direct_pred)
            ).sum()
            + eps
        )
    )


def close_scalar(
    a,
    b,
    rtol=NUMERIC_CLOSE_RTOL,
    atol=NUMERIC_CLOSE_ATOL,
) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True

    if pd.isna(a) or pd.isna(b):
        return False

    return bool(
        np.isclose(
            float(a),
            float(b),
            rtol=rtol,
            atol=atol,
        )
    )


# ============================================================
# EXPECTED PATHS
# ============================================================

def result_folder(
    dataset: str,
    mode: str,
    cfg: dict,
) -> str:
    return os.path.join(
        EVALUATION_RESULTS_PATH,
        cfg["model_dir"],
        dataset,
        cfg["base_resolution"],
        mode,
    )


def evaluator_csv_path(
    dataset: str,
    mode: str,
    folder: str,
) -> str:
    return os.path.join(
        folder,
        f"{dataset}_{mode}_evaluation.csv",
    )


def training_log_path(
    dataset: str,
    mode: str,
    log_dir: str,
) -> str:
    label = MODE_LABELS.get(
        mode,
        mode,
    )

    return os.path.join(
        log_dir,
        f"{label}_{dataset}_{mode}.log",
    )


# ============================================================
# FACTOR FILE DISCOVERY
# ============================================================

FACTOR_FILE_RE = re.compile(
    r"^(pred|true)_factor_(\d+)\.npy$"
)


def discover_factor_files(folder: str):
    pred = {}
    true = {}

    if not os.path.isdir(folder):
        return pred, true

    for name in os.listdir(folder):
        match = FACTOR_FILE_RE.match(
            name
        )

        if match is None:
            continue

        kind = match.group(1)
        factor = int(
            match.group(2)
        )

        full_path = os.path.join(
            folder,
            name,
        )

        if kind == "pred":
            pred[factor] = full_path
        else:
            true[factor] = full_path

    return pred, true


# ============================================================
# ARRAY / HIERARCHY CHECKS
# ============================================================

def check_saved_arrays(
    reporter: Reporter,
    dataset: str,
    mode: str,
    cfg: dict,
    folder: str,
):
    expected_factors = list(
        cfg["factor_to_res"].keys()
    )

    expected_set = set(
        expected_factors
    )

    pred_files, true_files = (
        discover_factor_files(
            folder
        )
    )

    pred_set = set(
        pred_files
    )

    true_set = set(
        true_files
    )

    if pred_set != expected_set:
        missing = sorted(
            expected_set - pred_set
        )
        extra = sorted(
            pred_set - expected_set
        )

        reporter.error(
            dataset,
            mode,
            "prediction factor files",
            f"expected={sorted(expected_set)}, "
            f"found={sorted(pred_set)}, "
            f"missing={missing}, extra={extra}",
        )

    else:
        reporter.pass_(
            dataset,
            mode,
            "prediction factor files",
            f"factors={sorted(pred_set)}",
        )

    if true_set != expected_set:
        missing = sorted(
            expected_set - true_set
        )
        extra = sorted(
            true_set - expected_set
        )

        reporter.error(
            dataset,
            mode,
            "truth factor files",
            f"expected={sorted(expected_set)}, "
            f"found={sorted(true_set)}, "
            f"missing={missing}, extra={extra}",
        )

    else:
        reporter.pass_(
            dataset,
            mode,
            "truth factor files",
            f"factors={sorted(true_set)}",
        )

    # Cannot continue reliably unless exact factor sets exist.
    if (
        pred_set != expected_set
        or true_set != expected_set
    ):
        return None, None

    preds: Dict[int, np.ndarray] = {}
    trues: Dict[int, np.ndarray] = {}

    n_values = []

    for factor in expected_factors:
        try:
            pred = squeeze_saved_array(
                np.load(
                    pred_files[factor]
                )
            )

            true = squeeze_saved_array(
                np.load(
                    true_files[factor]
                )
            )

        except Exception as exc:
            reporter.error(
                dataset,
                mode,
                f"load factor {factor}",
                str(exc),
            )
            return None, None

        if pred.shape != true.shape:
            reporter.error(
                dataset,
                mode,
                f"shape factor {factor}",
                f"pred={pred.shape}, true={true.shape}",
            )
            return None, None

        if (
            cfg["daily_stride"]
            % factor
            != 0
        ):
            reporter.error(
                dataset,
                mode,
                f"expected horizon factor {factor}",
                f"daily_stride={cfg['daily_stride']} is not "
                f"divisible by factor={factor}",
            )
            return None, None

        expected_h = (
            cfg["daily_stride"]
            // factor
        )

        if pred.shape[1] != expected_h:
            reporter.error(
                dataset,
                mode,
                f"horizon factor {factor}",
                f"expected H={expected_h}, got {pred.shape[1]}",
            )
        else:
            reporter.pass_(
                dataset,
                mode,
                f"horizon factor {factor}",
                f"shape={pred.shape}",
            )

        if pred.shape[0] < 1:
            reporter.error(
                dataset,
                mode,
                f"nonempty factor {factor}",
                "zero saved forecast windows",
            )
        else:
            n_values.append(
                pred.shape[0]
            )

        if not np.isfinite(pred).all():
            n_bad = int(
                (~np.isfinite(pred)).sum()
            )

            reporter.error(
                dataset,
                mode,
                f"finite predictions factor {factor}",
                f"{n_bad} NaN/Inf values",
            )
        else:
            reporter.pass_(
                dataset,
                mode,
                f"finite predictions factor {factor}",
            )

        if not np.isfinite(true).all():
            n_bad = int(
                (~np.isfinite(true)).sum()
            )

            reporter.error(
                dataset,
                mode,
                f"finite truth factor {factor}",
                f"{n_bad} NaN/Inf values",
            )
        else:
            reporter.pass_(
                dataset,
                mode,
                f"finite truth factor {factor}",
            )

        preds[factor] = pred
        trues[factor] = true

    if (
        n_values
        and len(set(n_values)) == 1
    ):
        reporter.pass_(
            dataset,
            mode,
            "aligned rolling origins",
            f"N={n_values[0]} for every factor",
        )

    else:
        reporter.error(
            dataset,
            mode,
            "aligned rolling origins",
            f"N by factor="
            f"{ {f: preds[f].shape[0] for f in expected_factors} }",
        )

    # --------------------------------------------------------
    # Strong truth check: ALL rolling windows, not only daily
    # non-overlapping evaluation windows.
    # --------------------------------------------------------

    for fine, coarse in cfg["pairs"]:
        ratio = (
            coarse // fine
        )

        pair_name = (
            f"{cfg['factor_to_res'][fine]}"
            f"->{cfg['factor_to_res'][coarse]}"
        )

        try:
            agg_true = aggregate_sum(
                trues[fine],
                ratio,
            )

            direct_true = trues[
                coarse
            ]

            if agg_true.shape != direct_true.shape:
                reporter.error(
                    dataset,
                    mode,
                    f"truth coherence {pair_name}",
                    f"aggregate shape={agg_true.shape}, "
                    f"direct shape={direct_true.shape}",
                )
                continue

            diff = (
                agg_true
                -
                direct_true
            )

            allclose = np.allclose(
                agg_true,
                direct_true,
                rtol=TRUTH_RTOL,
                atol=TRUTH_ATOL,
            )

            max_abs = float(
                np.max(
                    np.abs(diff)
                )
            )

            relative_l1 = float(
                np.abs(diff).sum()
                /
                (
                    np.abs(
                        direct_true
                    ).sum()
                    + 1e-12
                )
            )

            if allclose:
                reporter.pass_(
                    dataset,
                    mode,
                    f"truth coherence {pair_name}",
                    f"all windows | "
                    f"relative_L1={relative_l1:.3e}, "
                    f"max_abs={max_abs:.6g}",
                )
            else:
                reporter.error(
                    dataset,
                    mode,
                    f"truth coherence {pair_name}",
                    f"FAILED over all rolling windows | "
                    f"relative_L1={relative_l1:.3e}, "
                    f"max_abs={max_abs:.6g}",
                )

        except Exception as exc:
            reporter.error(
                dataset,
                mode,
                f"truth coherence {pair_name}",
                str(exc),
            )

    return preds, trues


# ============================================================
# DAILY SUBSAMPLING — INDEPENDENT REPRODUCTION
# ============================================================

def daily_views(
    preds: Dict[int, np.ndarray],
    trues: Dict[int, np.ndarray],
    daily_stride: int,
):
    available = {
        factor:
            len(
                true[
                    ::daily_stride
                ]
            )
        for factor, true in trues.items()
    }

    n_days = min(
        available.values()
    )

    daily_pred = {
        factor:
            pred[
                ::daily_stride
            ][
                :n_days
            ]
        for factor, pred in preds.items()
    }

    daily_true = {
        factor:
            true[
                ::daily_stride
            ][
                :n_days
            ]
        for factor, true in trues.items()
    }

    return (
        daily_pred,
        daily_true,
        n_days,
    )


# ============================================================
# INDEPENDENT DECISION RECOMPUTATION
# ============================================================

def network_decision_metrics(
    agg_pred,
    direct_pred,
    true_coarse,
    alpha=2.0,
    eps=1e-12,
):
    def one(pred):
        capacity = np.maximum(
            pred,
            0.0,
        )

        violation = (
            capacity
            <
            true_coarse
        )

        sla = float(
            violation.mean()
        )

        excess = np.maximum(
            capacity
            -
            true_coarse,
            0.0,
        )

        over = float(
            excess.sum()
            /
            (
                np.abs(
                    true_coarse
                ).sum()
                + eps
            )
        )

        cost = float(
            over
            +
            alpha * sla
        )

        return (
            violation,
            sla,
            over,
            cost,
        )

    a_state, a_sla, a_over, a_cost = one(
        agg_pred
    )

    d_state, d_sla, d_over, d_cost = one(
        direct_pred
    )

    mismatch = float(
        np.mean(
            a_state
            != d_state
        )
    )

    return {
        "Agg_SLA_rate": a_sla,
        "Direct_SLA_rate": d_sla,
        "Agg_overprovision": a_over,
        "Direct_overprovision": d_over,
        "Agg_decision_cost": a_cost,
        "Direct_decision_cost": d_cost,
        "Decision_mismatch": mismatch,
        "Delta_D": d_cost - a_cost,
    }


def energy_decision_metrics(
    agg_pred,
    direct_pred,
    true_coarse,
    eps=1e-12,
):
    d_agg = np.asarray(
        agg_pred
    ).sum(
        axis=1
    )

    d_dir = np.asarray(
        direct_pred
    ).sum(
        axis=1
    )

    d_true = np.asarray(
        true_coarse
    ).sum(
        axis=1
    )

    denom = (
        np.abs(
            d_true
        ).sum()
        + eps
    )

    agg_error = float(
        np.abs(
            d_agg
            -
            d_true
        ).sum()
        /
        denom
    )

    dir_error = float(
        np.abs(
            d_dir
            -
            d_true
        ).sum()
        /
        denom
    )

    mismatch = float(
        np.abs(
            d_agg
            -
            d_dir
        ).sum()
        /
        denom
    )

    return {
        "Agg_decision_error": agg_error,
        "Direct_decision_error": dir_error,
        "Decision_mismatch": mismatch,
        "Delta_D": dir_error - agg_error,
    }


def compute_road_threshold(
    cfg: dict,
):
    params = cfg[
        "decision_params"
    ]

    sensor = params[
        "sensor"
    ]

    csv_path = os.path.join(
        params[
            "data_root"
        ],
        f"road_sensor_{sensor}_5m.csv",
    )

    df = pd.read_csv(
        csv_path
    )

    if (
        "date" not in df.columns
        or "value" not in df.columns
    ):
        raise ValueError(
            f"{csv_path} must contain date,value."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    train_end = pd.Timestamp(
        params["train_end"]
    )

    train = (
        df[
            df["date"]
            <
            train_end
        ]
        .copy()
        .set_index(
            "date"
        )
    )

    hourly = (
        train["value"]
        .resample("1h")
        .sum()
        .dropna()
    )

    q_cap = float(
        np.percentile(
            hourly,
            params[
                "capacity_percentile"
            ],
        )
    )

    threshold = float(
        params[
            "threshold_fraction"
        ]
        *
        q_cap
    )

    return threshold


def road_to_hourly(
    x,
    resolution,
):
    if resolution == "5m":
        return aggregate_sum(
            x,
            12,
        )

    if resolution == "15m":
        return aggregate_sum(
            x,
            4,
        )

    if resolution == "1h":
        return np.asarray(
            x
        )

    raise ValueError(
        f"Road operational decision is undefined at {resolution}."
    )


def road_decision_metrics(
    agg_pred,
    direct_pred,
    true_coarse,
    coarse_resolution,
    threshold,
):
    def decision(x):
        hourly = road_to_hourly(
            x,
            coarse_resolution,
        )

        return (
            hourly
            >
            threshold
        ).astype(int)

    d_agg = decision(
        agg_pred
    )

    d_dir = decision(
        direct_pred
    )

    d_true = decision(
        true_coarse
    )

    return {
        "Agg_decision_accuracy":
            float(
                np.mean(
                    d_agg
                    ==
                    d_true
                )
            ),

        "Direct_decision_accuracy":
            float(
                np.mean(
                    d_dir
                    ==
                    d_true
                )
            ),

        "Decision_mismatch":
            float(
                np.mean(
                    d_agg
                    !=
                    d_dir
                )
            ),

        "Delta_D":
            float(
                np.mean(
                    d_agg
                    ==
                    d_true
                )
                -
                np.mean(
                    d_dir
                    ==
                    d_true
                )
            ),
    }


# ============================================================
# EVALUATOR CSV CHECK
# ============================================================

def compare_metric(
    reporter: Reporter,
    dataset: str,
    mode: str,
    pair_name: str,
    column: str,
    observed,
    expected,
):
    check_name = (
        f"CSV {pair_name} {column}"
    )

    if close_scalar(
        observed,
        expected,
    ):
        reporter.pass_(
            dataset,
            mode,
            check_name,
            f"value={observed}",
        )
    else:
        reporter.error(
            dataset,
            mode,
            check_name,
            f"CSV={observed}, recomputed={expected}",
        )


def check_evaluator_csv(
    reporter: Reporter,
    dataset: str,
    mode: str,
    cfg: dict,
    folder: str,
    preds: Dict[int, np.ndarray],
    trues: Dict[int, np.ndarray],
):
    csv_path = evaluator_csv_path(
        dataset,
        mode,
        folder,
    )

    if not os.path.exists(
        csv_path
    ):
        reporter.error(
            dataset,
            mode,
            "evaluator CSV",
            f"missing {csv_path}",
        )
        return

    try:
        df = pd.read_csv(
            csv_path
        )
    except Exception as exc:
        reporter.error(
            dataset,
            mode,
            "evaluator CSV",
            f"cannot read: {exc}",
        )
        return

    missing_columns = sorted(
        REQUIRED_CSV_COLUMNS
        -
        set(df.columns)
    )

    if missing_columns:
        reporter.error(
            dataset,
            mode,
            "evaluator CSV schema",
            f"missing columns={missing_columns}",
        )
        return

    reporter.pass_(
        dataset,
        mode,
        "evaluator CSV exists",
        csv_path,
    )

    expected_pairs = [
        (
            f"{cfg['factor_to_res'][fine]}"
            f"->{cfg['factor_to_res'][coarse]}"
        )
        for fine, coarse
        in cfg["pairs"]
    ]

    observed_pairs = (
        df["Pair"]
        .astype(str)
        .tolist()
    )

    if (
        len(df)
        == len(expected_pairs)
        and set(observed_pairs)
        == set(expected_pairs)
        and not df["Pair"].duplicated().any()
    ):
        reporter.pass_(
            dataset,
            mode,
            "evaluator CSV pairs",
            f"{observed_pairs}",
        )
    else:
        reporter.error(
            dataset,
            mode,
            "evaluator CSV pairs",
            f"expected exactly {expected_pairs}, "
            f"found {observed_pairs}",
        )
        return

    dataset_values = set(
        df["Dataset"]
        .astype(str)
        .unique()
    )

    mode_values = set(
        df["Mode"]
        .astype(str)
        .unique()
    )

    if dataset_values == {
        dataset
    }:
        reporter.pass_(
            dataset,
            mode,
            "CSV dataset identity",
        )
    else:
        reporter.error(
            dataset,
            mode,
            "CSV dataset identity",
            f"found={dataset_values}",
        )

    if mode_values == {
        mode
    }:
        reporter.pass_(
            dataset,
            mode,
            "CSV mode identity",
        )
    else:
        reporter.error(
            dataset,
            mode,
            "CSV mode identity",
            f"found={mode_values}",
        )

    (
        daily_pred,
        daily_true,
        n_days,
    ) = daily_views(
        preds,
        trues,
        cfg["daily_stride"],
    )

    reporter.pass_(
        dataset,
        mode,
        "daily evaluation windows",
        f"N_days={n_days}, stride={cfg['daily_stride']}",
    )

    road_threshold = None

    if cfg["decision"] == "road_capacity":
        try:
            road_threshold = compute_road_threshold(
                cfg
            )

            reporter.pass_(
                dataset,
                mode,
                "road decision threshold",
                f"threshold={road_threshold:.6f}",
            )

        except Exception as exc:
            reporter.error(
                dataset,
                mode,
                "road decision threshold",
                str(exc),
            )
            return

    decision_pairs = cfg.get(
        "decision_pairs"
    )

    for fine, coarse in cfg[
        "pairs"
    ]:
        pair_name = (
            f"{cfg['factor_to_res'][fine]}"
            f"->{cfg['factor_to_res'][coarse]}"
        )

        row = df[
            df["Pair"]
            ==
            pair_name
        ].iloc[0]

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

        true_coarse = daily_true[
            coarse
        ]

        agg_pred = aggregate_sum(
            fine_pred,
            ratio,
        )

        expected_common = {
            "E_fine":
                nmse(
                    fine_pred,
                    fine_true,
                ),

            "E_agg":
                nmse(
                    agg_pred,
                    true_coarse,
                ),

            "E_dir":
                nmse(
                    direct_pred,
                    true_coarse,
                ),

            "C_rs":
                consistency(
                    agg_pred,
                    direct_pred,
                ),
        }

        expected_common[
            "Delta_E"
        ] = (
            expected_common["E_dir"]
            -
            expected_common["E_agg"]
        )

        expected_common[
            "Abs_Delta_E"
        ] = abs(
            expected_common[
                "Delta_E"
            ]
        )

        for column, expected in expected_common.items():
            compare_metric(
                reporter,
                dataset,
                mode,
                pair_name,
                column,
                row[column],
                expected,
            )

        decision_defined = (
            decision_pairs is None
            or (
                fine,
                coarse,
            )
            in decision_pairs
        )

        if not decision_defined:
            # Road daily pairs should remain explicitly undefined.
            for column in [
                "Decision_mismatch",
                "Delta_D",
            ]:
                if pd.isna(
                    row[column]
                ):
                    reporter.pass_(
                        dataset,
                        mode,
                        f"CSV {pair_name} {column} undefined",
                        "NaN as expected",
                    )
                else:
                    reporter.error(
                        dataset,
                        mode,
                        f"CSV {pair_name} {column} undefined",
                        f"expected NaN, found {row[column]}",
                    )

            continue

        if cfg["decision"] == "network_provisioning":
            expected_decision = network_decision_metrics(
                agg_pred,
                direct_pred,
                true_coarse,
                alpha=cfg[
                    "decision_params"
                ].get(
                    "alpha",
                    2.0,
                ),
            )

        elif cfg["decision"] == "energy_scheduling":
            expected_decision = energy_decision_metrics(
                agg_pred,
                direct_pred,
                true_coarse,
            )

        elif cfg["decision"] == "road_capacity":
            expected_decision = road_decision_metrics(
                agg_pred,
                direct_pred,
                true_coarse,
                cfg[
                    "factor_to_res"
                ][coarse],
                road_threshold,
            )

        else:
            reporter.error(
                dataset,
                mode,
                f"decision {pair_name}",
                f"unsupported decision={cfg['decision']}",
            )
            continue

        for column, expected in expected_decision.items():
            if column not in df.columns:
                reporter.error(
                    dataset,
                    mode,
                    f"CSV {pair_name} {column}",
                    "column missing",
                )
                continue

            compare_metric(
                reporter,
                dataset,
                mode,
                pair_name,
                column,
                row[column],
                expected,
            )

        # ----------------------------------------------------
        # Domain-range sanity checks.
        # ----------------------------------------------------

        mismatch = float(
            row[
                "Decision_mismatch"
            ]
        )

        if cfg[
            "decision"
        ] in {
            "network_provisioning",
            "road_capacity",
        }:
            if (
                0.0
                <= mismatch
                <= 1.0
            ):
                reporter.pass_(
                    dataset,
                    mode,
                    f"decision range {pair_name}",
                    f"mismatch={mismatch:.6f}",
                )
            else:
                reporter.error(
                    dataset,
                    mode,
                    f"decision range {pair_name}",
                    f"mismatch={mismatch} outside [0,1]",
                )

        elif cfg[
            "decision"
        ] == "energy_scheduling":
            if mismatch >= 0.0:
                reporter.pass_(
                    dataset,
                    mode,
                    f"decision range {pair_name}",
                    f"normalized mismatch={mismatch:.6f}",
                )
            else:
                reporter.error(
                    dataset,
                    mode,
                    f"decision range {pair_name}",
                    f"negative normalized mismatch={mismatch}",
                )


# ============================================================
# TRAINING LOG CHECK
# ============================================================

FLOAT_TOKEN = (
    r"[-+]?"
    r"(?:"
    r"(?:\d+(?:\.\d*)?)"
    r"|"
    r"(?:\.\d+)"
    r")"
    r"(?:[eE][-+]?\d+)?"
    r"|[-+]?(?:nan|inf)"
)


def parse_first(
    pattern: str,
    text: str,
    flags=0,
):
    match = re.search(
        pattern,
        text,
        flags,
    )

    if match is None:
        return None

    return match.group(1)


def check_training_log(
    reporter: Reporter,
    dataset: str,
    mode: str,
    cfg: dict,
    log_dir: str,
    checkpoint_check: bool,
):
    path = training_log_path(
        dataset,
        mode,
        log_dir,
    )

    if not os.path.exists(
        path
    ):
        reporter.error(
            dataset,
            mode,
            "training log",
            f"missing {path}",
        )
        return

    try:
        text = Path(
            path
        ).read_text(
            errors="replace"
        )
    except Exception as exc:
        reporter.error(
            dataset,
            mode,
            "training log",
            f"cannot read: {exc}",
        )
        return

    reporter.pass_(
        dataset,
        mode,
        "training log exists",
        path,
    )

    if "Traceback (most recent call last)" in text:
        reporter.error(
            dataset,
            mode,
            "training traceback",
            "Traceback found in training log",
        )
    else:
        reporter.pass_(
            dataset,
            mode,
            "training traceback",
            "none",
        )

    # --------------------------------------------------------
    # Configuration printed by Exp_Multiscale_LACE_Forecast.
    # --------------------------------------------------------

    factor_text = parse_first(
        r"factors\s*=\s*\[([^\]]+)\]",
        text,
    )

    if factor_text is None:
        reporter.error(
            dataset,
            mode,
            "logged factors",
            "could not parse factors from [LACE] configuration",
        )
    else:
        try:
            factors = [
                int(
                    token.strip()
                )
                for token
                in factor_text.split(",")
                if token.strip()
            ]

            expected = list(
                cfg[
                    "factor_to_res"
                ].keys()
            )

            if factors == expected:
                reporter.pass_(
                    dataset,
                    mode,
                    "logged factors",
                    f"{factors}",
                )
            else:
                reporter.error(
                    dataset,
                    mode,
                    "logged factors",
                    f"expected={expected}, found={factors}",
                )

        except Exception as exc:
            reporter.error(
                dataset,
                mode,
                "logged factors",
                str(exc),
            )

    logged_mode = parse_first(
        r"coherence mode\s*=\s*([A-Za-z_]+)",
        text,
        flags=re.IGNORECASE,
    )

    if logged_mode is None:
        reporter.error(
            dataset,
            mode,
            "logged coherence mode",
            "could not parse",
        )
    elif (
        logged_mode.strip().lower()
        ==
        mode.lower()
    ):
        reporter.pass_(
            dataset,
            mode,
            "logged coherence mode",
            logged_mode,
        )
    else:
        reporter.error(
            dataset,
            mode,
            "logged coherence mode",
            f"expected={mode}, found={logged_mode}",
        )

    logged_lambda = parse_first(
        rf"lambda coherence\s*=\s*({FLOAT_TOKEN})",
        text,
        flags=re.IGNORECASE,
    )

    expected_lambda = EXPECTED_LAMBDA[
        mode
    ]

    if logged_lambda is None:
        reporter.error(
            dataset,
            mode,
            "logged lambda",
            "could not parse",
        )
    else:
        try:
            value = float(
                logged_lambda
            )

            if np.isclose(
                value,
                expected_lambda,
                rtol=0.0,
                atol=1e-12,
            ):
                reporter.pass_(
                    dataset,
                    mode,
                    "logged lambda",
                    f"{value}",
                )
            else:
                reporter.error(
                    dataset,
                    mode,
                    "logged lambda",
                    f"expected={expected_lambda}, found={value}",
                )

        except Exception as exc:
            reporter.error(
                dataset,
                mode,
                "logged lambda",
                str(exc),
            )

    # --------------------------------------------------------
    # Real-batch sanity.
    # --------------------------------------------------------

    if re.search(
        r"status:\s*PASS",
        text,
        flags=re.IGNORECASE,
    ):
        reporter.pass_(
            dataset,
            mode,
            "real-batch sanity",
            "PASS found in log",
        )
    else:
        reporter.error(
            dataset,
            mode,
            "real-batch sanity",
            "PASS marker not found",
        )

    # --------------------------------------------------------
    # Logged losses: reject NaN/Inf.
    # --------------------------------------------------------

    loss_tokens = re.findall(
        rf"(?:total|acc|coh|train_total|train_acc|train_coh|"
        rf"val_total|val_acc|val_coh)=({FLOAT_TOKEN})",
        text,
        flags=re.IGNORECASE,
    )

    parsed_losses = []

    for token in loss_tokens:
        try:
            parsed_losses.append(
                float(token)
            )
        except ValueError:
            parsed_losses.append(
                np.nan
            )

    if not parsed_losses:
        reporter.warn(
            dataset,
            mode,
            "logged finite losses",
            "no loss values could be parsed",
        )

    elif not np.isfinite(
        np.asarray(
            parsed_losses,
            dtype=float,
        )
    ).all():
        reporter.error(
            dataset,
            mode,
            "logged finite losses",
            "NaN/Inf loss found",
        )

    else:
        reporter.pass_(
            dataset,
            mode,
            "logged finite losses",
            f"{len(parsed_losses)} values checked",
        )

    # --------------------------------------------------------
    # Coherence-loss activation.
    # --------------------------------------------------------

    coh_tokens = re.findall(
        rf"(?:^|\|\s*)coh=({FLOAT_TOKEN})",
        text,
        flags=(
            re.IGNORECASE
            |
            re.MULTILINE
        ),
    )

    coh_values = []

    for token in coh_tokens:
        try:
            coh_values.append(
                float(token)
            )
        except ValueError:
            pass

    finite_coh = [
        value
        for value in coh_values
        if np.isfinite(value)
    ]

    if mode == "none":
        if not finite_coh:
            reporter.warn(
                dataset,
                mode,
                "A0 coherence loss",
                "no coh= values parsed",
            )
        elif all(
            abs(value)
            <= 1e-12
            for value
            in finite_coh
        ):
            reporter.pass_(
                dataset,
                mode,
                "A0 coherence loss",
                f"{len(finite_coh)} logged values are zero",
            )
        else:
            reporter.error(
                dataset,
                mode,
                "A0 coherence loss",
                f"non-zero values found; max_abs="
                f"{max(abs(v) for v in finite_coh):.6g}",
            )

    elif mode in {
        "structural",
    }:
        if not finite_coh:
            reporter.warn(
                dataset,
                mode,
                f"{MODE_LABELS[mode]} coherence loss",
                "no coh= values parsed",
            )
        elif any(
            abs(value)
            > 1e-12
            for value
            in finite_coh
        ):
            reporter.pass_(
                dataset,
                mode,
                f"{MODE_LABELS[mode]} coherence loss active",
                f"max_abs={max(abs(v) for v in finite_coh):.6g}",
            )
        else:
            reporter.error(
                dataset,
                mode,
                f"{MODE_LABELS[mode]} coherence loss active",
                "all parsed coherence losses are zero",
            )

    # --------------------------------------------------------
    # Best checkpoint existence + finite state.
    # --------------------------------------------------------

    checkpoint_paths = re.findall(
        r"saved best checkpoint:\s*(.+?\.pth)\s*$",
        text,
        flags=re.MULTILINE,
    )

    if not checkpoint_paths:
        reporter.error(
            dataset,
            mode,
            "best checkpoint",
            "no saved-best-checkpoint entry found in log",
        )
        return

    checkpoint_path = (
        checkpoint_paths[-1]
        .strip()
    )

    if not os.path.exists(
        checkpoint_path
    ):
        reporter.error(
            dataset,
            mode,
            "best checkpoint",
            f"referenced checkpoint does not exist: {checkpoint_path}",
        )
        return

    reporter.pass_(
        dataset,
        mode,
        "best checkpoint exists",
        checkpoint_path,
    )

    if not checkpoint_check:
        return

    try:
        import torch

        state = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if not isinstance(
            state,
            dict,
        ):
            reporter.error(
                dataset,
                mode,
                "checkpoint state",
                f"expected dict, got {type(state)}",
            )
            return

        tensor_count = 0
        bad_count = 0

        for value in state.values():
            if torch.is_tensor(
                value
            ):
                tensor_count += 1

                if (
                    value.is_floating_point()
                    or value.is_complex()
                ):
                    if not torch.isfinite(
                        value
                    ).all():
                        bad_count += 1

        if tensor_count == 0:
            reporter.error(
                dataset,
                mode,
                "checkpoint state",
                "no tensors found in state_dict",
            )

        elif bad_count > 0:
            reporter.error(
                dataset,
                mode,
                "checkpoint finite tensors",
                f"{bad_count} tensors contain NaN/Inf",
            )

        else:
            reporter.pass_(
                dataset,
                mode,
                "checkpoint finite tensors",
                f"{tensor_count} tensors checked",
            )

    except Exception as exc:
        reporter.error(
            dataset,
            mode,
            "checkpoint load",
            str(exc),
        )


# ============================================================
# ONE RUN
# ============================================================

def check_run(
    reporter: Reporter,
    dataset: str,
    mode: str,
    road_data_root: str,
    log_dir: str,
    skip_log_checks: bool,
    checkpoint_check: bool,
):
    try:
        cfg = get_dataset_config(
            dataset,
            road_data_root=road_data_root,
        )
    except Exception as exc:
        reporter.error(
            dataset,
            mode,
            "dataset configuration",
            str(exc),
        )
        return

    folder = result_folder(
        dataset,
        mode,
        cfg,
    )

    if not os.path.isdir(
        folder
    ):
        reporter.missing(
            dataset,
            mode,
            "result folder",
            folder,
        )
        return

    reporter.pass_(
        dataset,
        mode,
        "result folder",
        folder,
    )

    preds, trues = (
        check_saved_arrays(
            reporter,
            dataset,
            mode,
            cfg,
            folder,
        )
    )

    if (
        preds is not None
        and trues is not None
    ):
        check_evaluator_csv(
            reporter,
            dataset,
            mode,
            cfg,
            folder,
            preds,
            trues,
        )

    if not skip_log_checks:
        check_training_log(
            reporter,
            dataset,
            mode,
            cfg,
            log_dir,
            checkpoint_check,
        )


# ============================================================
# SUMMARY
# ============================================================

def summarize_runs(
    report_df: pd.DataFrame,
    datasets: List[str],
    modes: List[str],
):
    rows = []

    for dataset in datasets:
        for mode in modes:
            group = report_df[
                (
                    report_df[
                        "Dataset"
                    ]
                    ==
                    dataset
                )
                &
                (
                    report_df[
                        "Mode"
                    ]
                    ==
                    mode
                )
            ]

            counts = (
                group[
                    "Status"
                ]
                .value_counts()
                .to_dict()
            )

            if counts.get(
                "ERROR",
                0,
            ) > 0:
                overall = "FAIL"

            elif counts.get(
                "MISSING",
                0,
            ) > 0:
                overall = "MISSING"

            elif counts.get(
                "WARNING",
                0,
            ) > 0:
                overall = "PASS_WITH_WARNINGS"

            else:
                overall = "PASS"

            rows.append({
                "Dataset": dataset,
                "Mode": mode,
                "Label": MODE_LABELS.get(
                    mode,
                    mode,
                ),
                "Overall": overall,
                "Pass_checks": counts.get(
                    "PASS",
                    0,
                ),
                "Warnings": counts.get(
                    "WARNING",
                    0,
                ),
                "Errors": counts.get(
                    "ERROR",
                    0,
                ),
                "Missing": counts.get(
                    "MISSING",
                    0,
                ),
            })

    return pd.DataFrame(
        rows
    )


def print_summary(
    summary: pd.DataFrame,
):
    print(
        "\n============================================================"
    )
    print(
        "LACE INTEGRITY SUMMARY"
    )
    print(
        "============================================================"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    for _, row in summary.iterrows():
        symbol = {
            "PASS": "PASS",
            "PASS_WITH_WARNINGS": "WARN",
            "FAIL": "FAIL",
            "MISSING": "MISS",
        }[
            row["Overall"]
        ]

        print(
            f"[{symbol}] "
            f"{row['Dataset']} "
            f"{row['Label']}: "
            f"{row['Pass_checks']} pass, "
            f"{row['Warnings']} warning, "
            f"{row['Errors']} error, "
            f"{row['Missing']} missing"
        )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Strict integrity checker for LACE experiment outputs."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=[
            "none",
            "structural",
        ],
    )

    parser.add_argument(
        "--road_data_root",
        default="./data/Road",
    )

    parser.add_argument(
        "--log_dir",
        default="./logs/lace_overnight",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Do not return a failing exit code solely because an "
            "expected run has not finished yet."
        ),
    )

    parser.add_argument(
        "--skip-log-checks",
        action="store_true",
        help=(
            "Check only saved predictions/truths/evaluator CSVs."
        ),
    )

    parser.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help=(
            "Verify checkpoint path exists but do not torch.load it."
        ),
    )

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = os.path.join(
            EVALUATION_RESULTS_PATH,
            "LACE_moirai1",
            "_summary",
        )
    else:
        output_dir = args.output_dir

    Path(
        output_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    reporter = Reporter()

    for dataset in args.datasets:
        for mode in args.modes:
            print(
                f"Checking {dataset} / {mode} ..."
            )

            check_run(
                reporter=reporter,
                dataset=dataset,
                mode=mode,
                road_data_root=args.road_data_root,
                log_dir=args.log_dir,
                skip_log_checks=args.skip_log_checks,
                checkpoint_check=(
                    not args.skip_checkpoint_load
                ),
            )

    report_df = reporter.dataframe()

    summary = summarize_runs(
        report_df,
        args.datasets,
        args.modes,
    )

    report_path = os.path.join(
        output_dir,
        "LACE_integrity_checks.csv",
    )

    summary_path = os.path.join(
        output_dir,
        "LACE_integrity_summary.csv",
    )

    report_df.to_csv(
        report_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print_summary(
        summary
    )

    print(
        "\nSaved detailed report:"
    )
    print(
        f"  {report_path}"
    )

    print(
        "Saved run summary:"
    )
    print(
        f"  {summary_path}"
    )

    has_error = bool(
        (
            report_df[
                "Status"
            ]
            ==
            "ERROR"
        ).any()
    )

    has_missing = bool(
        (
            report_df[
                "Status"
            ]
            ==
            "MISSING"
        ).any()
    )

    if has_error:
        sys.exit(
            1
        )

    if (
        has_missing
        and not args.allow_missing
    ):
        sys.exit(
            2
        )

    sys.exit(
        0
    )


if __name__ == "__main__":
    main()
