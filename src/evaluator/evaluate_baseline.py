"""Evaluate independent and reconciled baseline forecasts using the study metrics."""
# src/evaluator/evaluate_baseline.py

import os
import argparse

import numpy as np
import pandas as pd

from src.config import EVALUATION_RESULTS_PATH
from src.utils.metrics import metric

from src.evaluator.evaluate_lace import (
    get_dataset_config,
    squeeze_last_dim,
    aggregate_sum,
    nmse,
    cross_resolution_consistency,
    verify_ground_truth_coherence,
    compute_road_threshold,
    evaluate_decision_routes,
    plot_relationships,
)


SUPPORTED_MODELS = [
    "DLinear",
    "LSTM",
]

SUPPORTED_BASELINES = [
    "independent",
    "bottomup",
    "mint_struct",
    "ols",
]

BASELINE_LABELS = {
    "independent": "Independent",
    "bottomup": "Bottom-Up",
    "mint_struct": "WLS-Struct",
    "ols": "OLS",
}


# ============================================================
# PATHS
# ============================================================

def source_folder(model, baseline, dataset, resolution=None, cfg=None):
    """
    Return the source-array folder for a baseline.

    independent:
        Evaluation/<Model>_independent/<dataset>/<resolution>/direct/

    bottomup:
        Evaluation/<Model>_bottomup/<dataset>/<base_resolution>/bottomup/

    mint_struct:
        Evaluation/<Model>_mint_struct/<dataset>/<base_resolution>/mint_struct/

    """

    if baseline == "independent":
        if resolution is None:
            raise ValueError("resolution is required for independent baseline.")

        return os.path.join(
            EVALUATION_RESULTS_PATH,
            f"{model}_independent",
            dataset,
            resolution,
            "direct",
        )

    if cfg is None:
        raise ValueError("cfg is required for reconciled/bottom-up baselines.")

    if baseline == "bottomup":
        root = f"{model}_bottomup"
        mode = "bottomup"

    elif baseline == "mint_struct":
        root = f"{model}_mint_struct"
        mode = "mint_struct"

    elif baseline == "ols":
        root = f"{model}_ols"
        mode = "ols"

    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    return os.path.join(
        EVALUATION_RESULTS_PATH,
        root,
        dataset,
        cfg["base_resolution"],
        mode,
    )


def evaluation_folder(model, baseline, dataset, cfg):
    """
    Keep evaluator CSVs/plots separate from source prediction arrays.
    """

    return os.path.join(
        EVALUATION_RESULTS_PATH,
        "Baseline_Evaluation",
        model,
        baseline,
        dataset,
        cfg["base_resolution"],
    )


# ============================================================
# ARRAY LOADING
# ============================================================

def load_independent_outputs(model, dataset, cfg):
    """
    Independent models are trained separately at every resolution.

    Each test loader advances by one native-resolution timestep, so
    each resolution must be sampled using its own one-day horizon:

        horizon_r = daily_stride / factor_r
    """

    preds = {}
    trues = {}

    for factor, resolution in cfg["factor_to_res"].items():

        folder = source_folder(
            model=model,
            baseline="independent",
            dataset=dataset,
            resolution=resolution,
        )

        pred_path = os.path.join(folder, "pred.npy")
        true_path = os.path.join(folder, "true.npy")

        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        if not os.path.exists(true_path):
            raise FileNotFoundError(f"Missing truth: {true_path}")

        pred = squeeze_last_dim(np.load(pred_path))
        true = squeeze_last_dim(np.load(true_path))

        if pred.shape != true.shape:
            raise RuntimeError(
                f"{model}/{dataset}/{resolution}: "
                f"pred={pred.shape}, true={true.shape}"
            )

        horizon = cfg["daily_stride"] // int(factor)

        if pred.shape[1] != horizon:
            raise RuntimeError(
                f"{model}/{dataset}/{resolution}: expected one-day "
                f"horizon={horizon}, got {pred.shape[1]}."
            )

        # Native-resolution daily forecast origins.
        pred = pred[::horizon]
        true = true[::horizon]

        preds[factor] = pred
        trues[factor] = true

        print(
            f"{resolution:>5s} | factor={factor:<4d} | "
            f"daily pred={pred.shape} | true={true.shape}"
        )

    n_days = min(len(x) for x in trues.values())

    if n_days < 1:
        raise RuntimeError("No common independent daily windows recovered.")

    preds = {
        factor: pred[:n_days]
        for factor, pred in preds.items()
    }

    trues = {
        factor: true[:n_days]
        for factor, true in trues.items()
    }

    return preds, trues


def load_factor_outputs(model, baseline, dataset, cfg):
    """
    Load factor-based Bottom-Up / reconciled outputs.
    """

    folder = source_folder(
        model=model,
        baseline=baseline,
        dataset=dataset,
        cfg=cfg,
    )

    preds = {}
    trues = {}

    for factor, resolution in cfg["factor_to_res"].items():

        pred_path = os.path.join(
            folder,
            f"pred_factor_{factor}.npy",
        )

        true_path = os.path.join(
            folder,
            f"true_factor_{factor}.npy",
        )

        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        if not os.path.exists(true_path):
            raise FileNotFoundError(f"Missing truth: {true_path}")

        pred = squeeze_last_dim(np.load(pred_path))
        true = squeeze_last_dim(np.load(true_path))

        if pred.shape != true.shape:
            raise RuntimeError(
                f"{model}/{baseline}/{dataset}/{resolution}: "
                f"pred={pred.shape}, true={true.shape}"
            )

        preds[factor] = pred
        trues[factor] = true

        print(
            f"{resolution:>5s} | factor={factor:<4d} | "
            f"pred={pred.shape} | true={true.shape}"
        )

    return preds, trues, folder


def recover_factor_daily_windows(preds, trues, baseline, cfg):
    """
    Bottom-Up files are generated from every sliding window of the
    finest-resolution model, so they share the base-resolution stride.

    WLS-Struct and OLS files already use daily forecast origins, as saved by
    reconcile_baselines.py. Use those aligned arrays directly.
    """

    if baseline == "bottomup":

        stride = cfg["daily_stride"]

        available = {
            factor: len(true[::stride])
            for factor, true in trues.items()
        }

        n_days = min(available.values())

        daily_pred = {
            factor: pred[::stride][:n_days]
            for factor, pred in preds.items()
        }

        daily_true = {
            factor: true[::stride][:n_days]
            for factor, true in trues.items()
        }

    elif baseline in {
        "mint_struct",
        "ols",
    }:

        n_days = min(
            len(true)
            for true in trues.values()
        )

        daily_pred = {
            factor: pred[:n_days]
            for factor, pred in preds.items()
        }

        daily_true = {
            factor: true[:n_days]
            for factor, true in trues.items()
        }

    else:
        raise ValueError(
            f"recover_factor_daily_windows does not support {baseline}"
        )

    if n_days < 1:
        raise RuntimeError("No common daily windows recovered.")

    print(f"[ALIGN] common daily origins = {n_days}")

    return daily_pred, daily_true


def load_baseline_daily_outputs(model, baseline, dataset, cfg):

    if baseline == "independent":
        return load_independent_outputs(
            model=model,
            dataset=dataset,
            cfg=cfg,
        )

    preds, trues, _ = load_factor_outputs(
        model=model,
        baseline=baseline,
        dataset=dataset,
        cfg=cfg,
    )

    return recover_factor_daily_windows(
        preds=preds,
        trues=trues,
        baseline=baseline,
        cfg=cfg,
    )


# ============================================================
# RESOLUTION ACCURACY
# ============================================================

def build_resolution_accuracy(
    model,
    baseline,
    dataset,
    cfg,
    daily_pred,
    daily_true,
):

    rows = []

    for factor, resolution in cfg["factor_to_res"].items():

        pred = daily_pred[factor]
        true = daily_true[factor]

        mae, mse_v, rmse, mape, mspe = metric(
            pred,
            true,
        )

        rows.append({
            "Forecaster": model,
            "Baseline": baseline,
            "Baseline_label": BASELINE_LABELS[baseline],
            "Dataset": dataset,
            "Factor": int(factor),
            "Resolution": resolution,
            "MSE": float(mse_v),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "MSPE": float(mspe),
            "NMSE": float(nmse(pred, true)),
        })

    return pd.DataFrame(rows)


# ============================================================
# FULL BASELINE EVALUATION
# ============================================================

def evaluate_baseline(
    model,
    baseline,
    dataset,
    road_data_root="./data/Road",
):

    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model}'. "
            f"Choose from {SUPPORTED_MODELS}."
        )

    if baseline not in SUPPORTED_BASELINES:
        raise ValueError(
            f"Unsupported baseline '{baseline}'. "
            f"Choose from {SUPPORTED_BASELINES}."
        )

    cfg = get_dataset_config(
        dataset,
        road_data_root=road_data_root,
    )

    factor_to_res = cfg["factor_to_res"]
    pairs = cfg["pairs"]

    output_folder = evaluation_folder(
        model=model,
        baseline=baseline,
        dataset=dataset,
        cfg=cfg,
    )

    os.makedirs(
        output_folder,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print(f"Forecaster: {model}")
    print(
        f"Baseline:   {baseline} "
        f"({BASELINE_LABELS[baseline]})"
    )
    print(f"Dataset:    {dataset}")
    print(
        "Hierarchy:  "
        + " -> ".join(
            factor_to_res[f]
            for f in factor_to_res
        )
    )
    print(f"Output:     {output_folder}")
    print("=" * 72)

    daily_pred, daily_true = load_baseline_daily_outputs(
        model=model,
        baseline=baseline,
        dataset=dataset,
        cfg=cfg,
    )

    # This is the strongest alignment check. It catches incorrect
    # daily-origin selection before any metric is reported.
    verify_ground_truth_coherence(
        daily_true,
        pairs,
        factor_to_res,
    )

    road_q_cap = None
    road_threshold = None

    if cfg["decision"] == "road_capacity":

        road_q_cap, road_threshold = compute_road_threshold(
            cfg["decision_params"]
        )

        print(
            f"\n[ROAD] Sensor {cfg['sensor']} | "
            f"Qcap={road_q_cap:.2f} veh/h | "
            f"threshold={road_threshold:.2f} veh/h"
        )

    rows = []

    for fine, coarse in pairs:

        ratio = coarse // fine

        fine_pred = daily_pred[fine]
        fine_true = daily_true[fine]

        direct_pred = daily_pred[coarse]
        coarse_true = daily_true[coarse]

        agg_pred = aggregate_sum(
            fine_pred,
            ratio,
        )

        if agg_pred.shape != coarse_true.shape:
            raise RuntimeError(
                f"Aggregated shape mismatch "
                f"{factor_to_res[fine]}->{factor_to_res[coarse]}: "
                f"{agg_pred.shape} vs {coarse_true.shape}"
            )

        if direct_pred.shape != coarse_true.shape:
            raise RuntimeError(
                f"Direct shape mismatch for "
                f"{factor_to_res[coarse]}: "
                f"{direct_pred.shape} vs {coarse_true.shape}"
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

        row = {
            "Forecaster": model,
            "Baseline": baseline,
            "Baseline_label": BASELINE_LABELS[baseline],
            "Dataset": dataset,
            "Pair": (
                f"{factor_to_res[fine]}"
                f"->{factor_to_res[coarse]}"
            ),
            "Fine_factor": int(fine),
            "Coarse_factor": int(coarse),
            "E_fine": float(e_fine),
            "E_agg": float(e_agg),
            "E_dir": float(e_dir),
            "C_rs": float(c_rs),
            # Positive = aggregate route has lower error.
            "Delta_E": float(e_dir - e_agg),
            "Abs_Delta_E": float(abs(e_dir - e_agg)),
        }

        if "sensor" in cfg:
            row["Sensor"] = cfg["sensor"]
            row["Q_cap_veh_per_h"] = road_q_cap
            row["Decision_threshold_veh_per_h"] = road_threshold

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

        rows.append(row)

    full_table = pd.DataFrame(rows)

    evaluation_csv = os.path.join(
        output_folder,
        f"{dataset}_{baseline}_evaluation.csv",
    )

    full_table.to_csv(
        evaluation_csv,
        index=False,
    )

    accuracy_table = build_resolution_accuracy(
        model=model,
        baseline=baseline,
        dataset=dataset,
        cfg=cfg,
        daily_pred=daily_pred,
        daily_true=daily_true,
    )

    accuracy_csv = os.path.join(
        output_folder,
        f"{dataset}_{baseline}_resolution_accuracy.csv",
    )

    accuracy_table.to_csv(
        accuracy_csv,
        index=False,
    )

    print("\n=== BASELINE PAIRWISE RESULTS ===")
    print(
        full_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n=== RESOLUTION ACCURACY ===")
    print(
        accuracy_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSaved:")
    print(f"  {evaluation_csv}")
    print(f"  {accuracy_csv}")

    return (
        full_table,
        accuracy_table,
        output_folder,
        cfg,
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate independent, Bottom-Up, and post-hoc "
            "reconciliation baselines using the same LACE metrics."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
    )

    parser.add_argument(
        "--baseline",
        required=True,
        choices=SUPPORTED_BASELINES,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--road_data_root",
        default="./data/Road",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save the same relationship diagnostic used by evaluate_lace.",
    )

    args = parser.parse_args()

    full_table, _, output_folder, cfg = evaluate_baseline(
        model=args.model,
        baseline=args.baseline,
        dataset=args.dataset,
        road_data_root=args.road_data_root,
    )

    if args.plot:
        plot_relationships(
            full_table=full_table,
            output_folder=output_folder,
            dataset=args.dataset,
            mode=f"{args.model}_{args.baseline}",
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
