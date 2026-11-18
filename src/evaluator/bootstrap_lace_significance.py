"""Estimate paired moving-block bootstrap intervals for A0-vs-A1 changes.

Resample paired test windows from saved predictions. Dataset summaries use
all hierarchy resolutions for accuracy and adjacent edges for coherence.
Road coherence and decision summaries use the two finer-resolution edges.
Positive improvement values indicate better results under A1.

Run from the repository root:
    python -m src.evaluator.bootstrap_lace_significance --help"""

from __future__ import annotations

import os
import math
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import EVALUATION_RESULTS_PATH
from src.evaluator.evaluate_lace import (
    get_dataset_config,
    get_evaluation_folder,
    load_lace_outputs,
    recover_daily_windows,
    aggregate_sum,
    nmse,
    cross_resolution_consistency,
    compute_road_threshold,
    evaluate_decision_routes,
)

DEFAULT_DATASETS = [
    "c_traffic", "k_traffic", "ETTm1_HUFL",
    "road_sensor_62", "road_sensor_103", "road_sensor_240",
]
DEFAULT_MODEL_DIRS = [
    "LACE_moirai1", "LACE_moirai2", "LACE_chronos2", "LACE_timesfm2",
]
A0_MODE = "none"
A1_MODE = "structural"

LOWER_BETTER = {
    "NMSE", "Mean_NMSE", "C_rs", "Mean_C_rs",
    "Decision_mismatch", "Mean_decision_mismatch",
    "Decision_cost", "Scheduling_error",
}
HIGHER_BETTER = {"Decision_accuracy"}


def metric_direction(metric: str) -> str:
    if metric in LOWER_BETTER:
        return "lower"
    if metric in HIGHER_BETTER:
        return "higher"
    raise KeyError(f"No metric direction registered for {metric!r}")


def decision_metric_name(cfg: dict) -> str:
    name = cfg["decision"]
    if name == "network_provisioning":
        return "Decision_cost"
    if name == "energy_scheduling":
        return "Scheduling_error"
    if name == "road_capacity":
        return "Decision_accuracy"
    raise KeyError(name)


def decision_quality_from_result(cfg: dict, result: dict) -> float:
    """Match the current absolute decision-quality table convention."""
    name = cfg["decision"]
    if name == "network_provisioning":
        vals = [result.get("Agg_decision_cost", np.nan),
                result.get("Direct_decision_cost", np.nan)]
    elif name == "energy_scheduling":
        vals = [result.get("Agg_decision_error", np.nan),
                result.get("Direct_decision_error", np.nan)]
    elif name == "road_capacity":
        vals = [result.get("Agg_decision_accuracy", np.nan),
                result.get("Direct_decision_accuracy", np.nan)]
    else:
        raise KeyError(name)
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def load_one_mode(dataset: str, mode: str, model_dir: str, road_data_root: str):
    cfg = get_dataset_config(dataset, road_data_root=road_data_root)
    cfg = dict(cfg)
    cfg["factor_to_res"] = dict(cfg["factor_to_res"])
    cfg["pairs"] = list(cfg["pairs"])
    cfg["decision_params"] = dict(cfg.get("decision_params", {}))
    cfg["model_dir"] = model_dir

    folder = get_evaluation_folder(dataset, mode, cfg)
    preds, trues = load_lace_outputs(folder, cfg["factor_to_res"])
    daily_pred, daily_true = recover_daily_windows(
        preds, trues, cfg["daily_stride"]
    )
    return cfg, daily_pred, daily_true, folder


def load_paired_modes(dataset: str, model_dir: str, road_data_root: str):
    cfg0, pred0, true0, folder0 = load_one_mode(
        dataset, A0_MODE, model_dir, road_data_root
    )
    cfg1, pred1, true1, folder1 = load_one_mode(
        dataset, A1_MODE, model_dir, road_data_root
    )

    if cfg0["factor_to_res"] != cfg1["factor_to_res"]:
        raise RuntimeError(f"{dataset}/{model_dir}: A0/A1 hierarchy mismatch")

    factors = list(cfg0["factor_to_res"].keys())
    n_days = min(
        min(len(true0[f]) for f in factors),
        min(len(true1[f]) for f in factors),
    )
    if n_days < 2:
        raise RuntimeError(f"{dataset}/{model_dir}: only {n_days} paired days")

    pred0 = {f: pred0[f][:n_days] for f in factors}
    pred1 = {f: pred1[f][:n_days] for f in factors}
    true0 = {f: true0[f][:n_days] for f in factors}
    true1 = {f: true1[f][:n_days] for f in factors}

    for f in factors:
        if true0[f].shape != true1[f].shape:
            raise RuntimeError(f"{dataset}/{model_dir}/factor={f}: truth shape mismatch")
        if not np.allclose(true0[f], true1[f], rtol=1e-7, atol=1e-9):
            d = np.max(np.abs(true0[f] - true1[f]))
            raise RuntimeError(
                f"{dataset}/{model_dir}/factor={f}: A0/A1 truths differ "
                f"(max_abs={d:.3e})"
            )

    return cfg0, pred0, pred1, true0, n_days, folder0, folder1


def compute_snapshot(cfg: dict, preds: Dict[int, np.ndarray],
                     trues: Dict[int, np.ndarray], road_threshold):
    """Metrics keyed by (Scope, Item, Metric)."""
    out = {}
    factor_to_res = cfg["factor_to_res"]

    # Per-resolution and macro accuracy.
    vals = []
    for f, res in factor_to_res.items():
        v = float(nmse(preds[f], trues[f]))
        out[("resolution", str(res), "NMSE")] = v
        vals.append(v)
    out[("dataset", "ALL", "Mean_NMSE")] = float(np.mean(vals))

    # Pairwise coherence / decisions and dataset means.
    #
    # Summary edges match collect_final_plot_matrix.py:
    #   - adjacent hierarchy edges only;
    #   - for road, exclude the edge ending at 1d.
    #
    # This ensures the dataset-level A0->A1 bootstrap summaries use the
    # same aggregation rules as FINAL_A1_vs_A0.csv / the paper heatmap.
    coherence_vals, mismatch_vals, quality_vals = [], [], []
    quality_metric = decision_metric_name(cfg)

    factors = sorted(factor_to_res.keys())
    pairs = list(zip(factors[:-1], factors[1:]))

    if cfg["decision"] == "road_capacity":
        pairs = [
            (fine, coarse)
            for fine, coarse in pairs
            if factor_to_res[coarse] != "1d"
        ]

    for fine, coarse in pairs:
        if coarse % fine != 0:
            raise ValueError(f"Invalid hierarchy pair {fine}->{coarse}")
        ratio = coarse // fine
        agg_pred = aggregate_sum(preds[fine], ratio)
        direct_pred = preds[coarse]
        true_coarse = trues[coarse]
        if agg_pred.shape != direct_pred.shape:
            raise RuntimeError(
                f"Pair {fine}->{coarse}: {agg_pred.shape} != {direct_pred.shape}"
            )

        pair_name = f"{factor_to_res[fine]}->{factor_to_res[coarse]}"
        c = float(cross_resolution_consistency(agg_pred, direct_pred))
        out[("pair", pair_name, "C_rs")] = c
        coherence_vals.append(c)

        dr = evaluate_decision_routes(
            cfg=cfg, fine=fine, coarse=coarse,
            agg_pred=agg_pred, direct_pred=direct_pred,
            true_coarse=true_coarse, road_threshold=road_threshold,
        )
        mismatch = float(dr.get("Decision_mismatch", np.nan))
        if np.isfinite(mismatch):
            out[("pair", pair_name, "Decision_mismatch")] = mismatch
            mismatch_vals.append(mismatch)

        q = decision_quality_from_result(cfg, dr)
        if np.isfinite(q):
            out[("pair", pair_name, quality_metric)] = q
            quality_vals.append(q)

    out[("dataset", "ALL", "Mean_C_rs")] = float(np.mean(coherence_vals))
    if mismatch_vals:
        out[("dataset", "ALL", "Mean_decision_mismatch")] = float(
            np.mean(mismatch_vals)
        )
    if quality_vals:
        out[("dataset", "ALL", quality_metric)] = float(np.mean(quality_vals))
    return out


def circular_block_indices(rng: np.random.Generator, n: int,
                           block_length: int) -> np.ndarray:
    """Circular moving-block bootstrap over days."""
    b = max(1, min(int(block_length), n))
    starts = rng.integers(0, n, size=math.ceil(n / b))
    offsets = np.arange(b, dtype=int)
    idx = np.concatenate([(s + offsets) % n for s in starts])
    return idx[:n]


def subset_dict(arrays: Dict[int, np.ndarray], idx: np.ndarray):
    return {f: x[idx] for f, x in arrays.items()}


def summarize_metric(point_a0, point_a1, boot_a0, boot_a1,
                     direction: str, confidence: float):
    """
    Summarize paired A0 -> A1 change.

    Absolute Improvement:
        positive always means A1 is better.

    Relative Improvement (%):
        same direction convention, normalized by the observed A0 value.
        This matches the relative-improvement convention used in the paper figure.

    Bootstrap CIs are first computed for the paired absolute improvement and
    then expressed as a percentage of the observed A0 baseline. Because this
    is a positive linear rescaling, CI_result/significance are unchanged.
    """
    point_a0 = float(point_a0)
    point_a1 = float(point_a1)

    boot_a0 = np.asarray(boot_a0, dtype=float)
    boot_a1 = np.asarray(boot_a1, dtype=float)

    change = point_a1 - point_a0
    boot_change = boot_a1 - boot_a0

    # Direction-adjusted ABSOLUTE improvement: positive always means A1 is better.
    if direction == "lower":
        improvement = point_a0 - point_a1
        boot_improvement = boot_a0 - boot_a1
    elif direction == "higher":
        improvement = point_a1 - point_a0
        boot_improvement = boot_a1 - boot_a0
    else:
        raise ValueError(direction)

    # Absolute bootstrap CI.
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(
        boot_improvement, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    ci_low, ci_high = float(ci_low), float(ci_high)

    if ci_low > 0:
        result = "IMPROVED"
    elif ci_high < 0:
        result = "DEGRADED"
    else:
        result = "NO_CLEAR_DIFFERENCE"

    # Relative improvement (%) using the observed A0 baseline.
    # lower-is-better:  100 * (A0 - A1) / A0
    # higher-is-better: 100 * (A1 - A0) / A0
    denom = abs(point_a0)
    if denom > 1e-12:
        improvement_pct = 100.0 * improvement / denom
        ci_low_pct = 100.0 * ci_low / denom
        ci_high_pct = 100.0 * ci_high / denom
    else:
        improvement_pct = np.nan
        ci_low_pct = np.nan
        ci_high_pct = np.nan

    return {
        "A0_value": point_a0,
        "A1_value": point_a1,
        "Change_A1_minus_A0": change,
        "Improvement": improvement,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "Improvement_pct": improvement_pct,
        "CI_low_pct": ci_low_pct,
        "CI_high_pct": ci_high_pct,
        "CI_result": result,
        "Significant_95CI": bool(ci_low > 0 or ci_high < 0),
    }


def bootstrap_one_dataset(dataset: str, model_dir: str, road_data_root: str,
                          n_bootstrap: int, block_length: int,
                          confidence: float, seed: int):
    cfg, pred0, pred1, trues, n_days, folder0, folder1 = load_paired_modes(
        dataset, model_dir, road_data_root
    )

    road_threshold = None
    if cfg["decision"] == "road_capacity":
        _, road_threshold = compute_road_threshold(cfg["decision_params"])

    point0 = compute_snapshot(cfg, pred0, trues, road_threshold)
    point1 = compute_snapshot(cfg, pred1, trues, road_threshold)
    if set(point0) != set(point1):
        raise RuntimeError(f"{dataset}/{model_dir}: A0/A1 metric-key mismatch")

    keys = sorted(point0)
    boot0 = {k: np.empty(n_bootstrap, dtype=float) for k in keys}
    boot1 = {k: np.empty(n_bootstrap, dtype=float) for k in keys}

    stable_offset = sum(ord(c) for c in f"{model_dir}|{dataset}")
    rng = np.random.default_rng(int(seed) + stable_offset)

    for b in range(n_bootstrap):
        idx = circular_block_indices(rng, n_days, block_length)
        t = subset_dict(trues, idx)
        p0 = subset_dict(pred0, idx)
        p1 = subset_dict(pred1, idx)
        s0 = compute_snapshot(cfg, p0, t, road_threshold)
        s1 = compute_snapshot(cfg, p1, t, road_threshold)
        for k in keys:
            boot0[k][b] = s0[k]
            boot1[k][b] = s1[k]

    rows = []
    for scope, item, metric in keys:
        direction = metric_direction(metric)
        stats = summarize_metric(
            point0[(scope, item, metric)], point1[(scope, item, metric)],
            boot0[(scope, item, metric)], boot1[(scope, item, metric)],
            direction, confidence,
        )
        rows.append({
            "Backbone": model_dir,
            "Dataset": dataset,
            "Scope": scope,
            "Item": item,
            "Metric": metric,
            "Direction": direction,
            "N_days": int(n_days),
            "Block_length_days": int(min(block_length, n_days)),
            "N_bootstrap": int(n_bootstrap),
            "Confidence": float(confidence),
            "A0_folder": folder0,
            "A1_folder": folder1,
            **stats,
        })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(
        description="Paired moving-block bootstrap CIs for LACE A0 vs A1"
    )
    p.add_argument("--model-dirs", nargs="+", default=DEFAULT_MODEL_DIRS)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--road-data-root", default="./data/Road")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--block-length", type=int, default=7)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--allow-missing", action="store_true",
        help="Skip model/dataset pairs whose A0 or A1 results are missing"
    )
    args = p.parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap should be >= 100")
    if args.block_length < 1:
        raise ValueError("--block-length must be >= 1")
    if not (0 < args.confidence < 1):
        raise ValueError("--confidence must lie in (0,1)")
    return args


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        EVALUATION_RESULTS_PATH, "bootstrap_lace_significance"
    )
    os.makedirs(output_dir, exist_ok=True)

    all_rows, status_rows = [], []

    for model_dir in args.model_dirs:
        for dataset in args.datasets:
            print("\n" + "=" * 60)
            print(f"Backbone: {model_dir}")
            print(f"Dataset:  {dataset}")
            print("=" * 60)
            try:
                df = bootstrap_one_dataset(
                    dataset=dataset,
                    model_dir=model_dir,
                    road_data_root=args.road_data_root,
                    n_bootstrap=args.n_bootstrap,
                    block_length=args.block_length,
                    confidence=args.confidence,
                    seed=args.seed,
                )
                all_rows.append(df)
                status_rows.append({
                    "Backbone": model_dir, "Dataset": dataset,
                    "Status": "COMPLETE", "Message": "",
                })
                show = df[df["Scope"] == "dataset"][
                    [
                        "Metric",
                        "A0_value",
                        "A1_value",
                        "Improvement_pct",
                        "CI_low_pct",
                        "CI_high_pct",
                        "CI_result",
                    ]
                ]
                print("\nDataset-level bootstrap summary:")
                print(
                    show.to_string(
                        index=False,
                        formatters={
                            "A0_value": lambda x: f"{x:.6f}",
                            "A1_value": lambda x: f"{x:.6f}",
                            "Improvement_pct": lambda x: f"{x:+.2f}%",
                            "CI_low_pct": lambda x: f"{x:+.2f}%",
                            "CI_high_pct": lambda x: f"{x:+.2f}%",
                        },
                    )
                )
            except Exception as exc:
                status = "SKIPPED" if args.allow_missing else "ERROR"
                status_rows.append({
                    "Backbone": model_dir, "Dataset": dataset,
                    "Status": status, "Message": str(exc),
                })
                print(f"\n[{status}] {model_dir}/{dataset}: {exc}")
                if not args.allow_missing:
                    raise

    status_df = pd.DataFrame(status_rows)
    status_path = os.path.join(output_dir, "bootstrap_status.csv")
    status_df.to_csv(status_path, index=False)

    if not all_rows:
        print("\nNo complete A0/A1 pairs were available.")
        print("Saved:", status_path)
        return

    detailed = pd.concat(all_rows, ignore_index=True)
    detailed_path = os.path.join(output_dir, "bootstrap_detailed.csv")
    detailed.to_csv(detailed_path, index=False)

    summary = detailed[detailed["Scope"] == "dataset"].copy()
    summary_path = os.path.join(output_dir, "bootstrap_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("BOOTSTRAP COMPLETE")
    print("=" * 60)
    print("Saved:", summary_path)
    print("Saved:", detailed_path)
    print("Saved:", status_path)
    print("\nPositive Improvement always means A1 is better.")


if __name__ == "__main__":
    main()
