"""Collect baseline evaluation tables across models and datasets."""
# src/evaluator/collect_baseline_results.py

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EVALUATION_RESULTS_PATH
from src.evaluator.evaluate_lace import get_dataset_config
from src.evaluator.evaluate_baseline import (
    SUPPORTED_MODELS,
    SUPPORTED_BASELINES,
    BASELINE_LABELS,
    evaluation_folder,
)


DEFAULT_DATASETS = [
    "c_traffic",
    "k_traffic",
    "ETTm1_HUFL",
    "road_sensor_62",
    "road_sensor_103",
    "road_sensor_240",
]

DEFAULT_MODELS = [
    "DLinear",
    "LSTM",
]

DEFAULT_BASELINES = [
    "independent",
    "bottomup",
    "mint_struct",
]


# ============================================================
# HELPERS
# ============================================================

def expected_files(
    model,
    baseline,
    dataset,
    road_data_root,
):

    cfg = get_dataset_config(
        dataset,
        road_data_root=road_data_root,
    )

    folder = evaluation_folder(
        model=model,
        baseline=baseline,
        dataset=dataset,
        cfg=cfg,
    )

    evaluation_csv = os.path.join(
        folder,
        f"{dataset}_{baseline}_evaluation.csv",
    )

    accuracy_csv = os.path.join(
        folder,
        f"{dataset}_{baseline}_resolution_accuracy.csv",
    )

    return (
        cfg,
        folder,
        evaluation_csv,
        accuracy_csv,
    )


def safe_mean(series):

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = np.isfinite(x)

    if not finite.any():
        return np.nan

    return float(
        np.mean(x[finite])
    )


# ============================================================
# COLLECTION
# ============================================================

def collect_results(
    models,
    baselines,
    datasets,
    road_data_root,
):

    pairwise_tables = []
    accuracy_tables = []
    status_rows = []

    for model in models:

        for baseline in baselines:

            for dataset in datasets:

                _, folder, evaluation_csv, accuracy_csv = expected_files(
                    model=model,
                    baseline=baseline,
                    dataset=dataset,
                    road_data_root=road_data_root,
                )

                eval_exists = os.path.isfile(
                    evaluation_csv
                )

                accuracy_exists = os.path.isfile(
                    accuracy_csv
                )

                complete = (
                    eval_exists
                    and accuracy_exists
                )

                status_rows.append({
                    "Forecaster": model,
                    "Baseline": baseline,
                    "Baseline_label": BASELINE_LABELS.get(
                        baseline,
                        baseline,
                    ),
                    "Dataset": dataset,
                    "Complete": bool(complete),
                    "Evaluation_csv": evaluation_csv,
                    "Accuracy_csv": accuracy_csv,
                    "Folder": folder,
                })

                if not complete:
                    continue

                pairwise = pd.read_csv(
                    evaluation_csv
                )

                accuracy = pd.read_csv(
                    accuracy_csv
                )

                pairwise_tables.append(
                    pairwise
                )

                accuracy_tables.append(
                    accuracy
                )

    pairwise_master = (
        pd.concat(
            pairwise_tables,
            ignore_index=True,
        )
        if pairwise_tables
        else pd.DataFrame()
    )

    accuracy_master = (
        pd.concat(
            accuracy_tables,
            ignore_index=True,
        )
        if accuracy_tables
        else pd.DataFrame()
    )

    status = pd.DataFrame(
        status_rows
    )

    return (
        pairwise_master,
        accuracy_master,
        status,
    )


# ============================================================
# SUMMARIES
# ============================================================

def build_dataset_summary(pairwise_master):

    if pairwise_master.empty:
        return pd.DataFrame()

    rows = []

    group_cols = [
        "Forecaster",
        "Baseline",
        "Baseline_label",
        "Dataset",
    ]

    for keys, df in pairwise_master.groupby(
        group_cols,
        dropna=False,
    ):

        (
            model,
            baseline,
            baseline_label,
            dataset,
        ) = keys

        rows.append({
            "Forecaster": model,
            "Baseline": baseline,
            "Baseline_label": baseline_label,
            "Dataset": dataset,
            "Pairs": int(len(df)),
            "Mean_C_rs": safe_mean(df["C_rs"]),
            "Mean_E_fine": safe_mean(df["E_fine"]),
            "Mean_E_agg": safe_mean(df["E_agg"]),
            "Mean_E_dir": safe_mean(df["E_dir"]),
            "Mean_Abs_Delta_E": safe_mean(df["Abs_Delta_E"]),
            "Mean_Decision_mismatch": (
                safe_mean(df["Decision_mismatch"])
                if "Decision_mismatch" in df.columns
                else np.nan
            ),
            "Mean_Delta_D": (
                safe_mean(df["Delta_D"])
                if "Delta_D" in df.columns
                else np.nan
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "Forecaster",
                "Baseline",
                "Dataset",
            ]
        )
        .reset_index(drop=True)
    )


def build_accuracy_summary(accuracy_master):

    if accuracy_master.empty:
        return pd.DataFrame()

    metric_cols = [
        "MSE",
        "MAE",
        "RMSE",
        "MAPE",
        "MSPE",
        "NMSE",
    ]

    rows = []

    group_cols = [
        "Forecaster",
        "Baseline",
        "Baseline_label",
        "Dataset",
    ]

    for keys, df in accuracy_master.groupby(
        group_cols,
        dropna=False,
    ):

        (
            model,
            baseline,
            baseline_label,
            dataset,
        ) = keys

        row = {
            "Forecaster": model,
            "Baseline": baseline,
            "Baseline_label": baseline_label,
            "Dataset": dataset,
            "Resolutions": int(len(df)),
        }

        for col in metric_cols:
            row[f"Mean_{col}"] = safe_mean(
                df[col]
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "Forecaster",
                "Baseline",
                "Dataset",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# SAVE
# ============================================================

def save_table(df, path):

    df.to_csv(
        path,
        index=False,
    )

    print(
        "Saved:",
        path,
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Collect baseline evaluator CSVs across conventional "
            "forecasters, datasets, and reconciliation methods."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=SUPPORTED_MODELS,
    )

    parser.add_argument(
        "--baselines",
        nargs="+",
        default=DEFAULT_BASELINES,
        choices=SUPPORTED_BASELINES,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
    )

    parser.add_argument(
        "--road_data_root",
        default="./data/Road",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    args = parser.parse_args()

    if args.output_dir is None:

        output_dir = os.path.join(
            EVALUATION_RESULTS_PATH,
            "Baseline_Evaluation",
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

    (
        pairwise_master,
        accuracy_master,
        status,
    ) = collect_results(
        models=args.models,
        baselines=args.baselines,
        datasets=args.datasets,
        road_data_root=args.road_data_root,
    )

    status_path = os.path.join(
        output_dir,
        "BASELINE_run_status.csv",
    )

    save_table(
        status,
        status_path,
    )

    complete_n = int(
        status["Complete"].sum()
    )

    print(
        f"\nComplete baseline evaluations: "
        f"{complete_n}/{len(status)}"
    )

    missing = status[
        ~status["Complete"]
    ]

    if not missing.empty:

        print(
            "\n=== MISSING BASELINE EVALUATIONS ==="
        )

        print(
            missing[
                [
                    "Forecaster",
                    "Baseline",
                    "Dataset",
                ]
            ].to_string(
                index=False
            )
        )

    if pairwise_master.empty:

        print(
            "\nNo completed baseline evaluation CSVs found."
        )

        return

    pairwise_path = os.path.join(
        output_dir,
        "BASELINE_all_results.csv",
    )

    save_table(
        pairwise_master,
        pairwise_path,
    )

    accuracy_path = os.path.join(
        output_dir,
        "BASELINE_resolution_accuracy.csv",
    )

    save_table(
        accuracy_master,
        accuracy_path,
    )

    dataset_summary = build_dataset_summary(
        pairwise_master
    )

    dataset_summary_path = os.path.join(
        output_dir,
        "BASELINE_dataset_summary.csv",
    )

    save_table(
        dataset_summary,
        dataset_summary_path,
    )

    accuracy_summary = build_accuracy_summary(
        accuracy_master
    )

    accuracy_summary_path = os.path.join(
        output_dir,
        "BASELINE_accuracy_summary.csv",
    )

    save_table(
        accuracy_summary,
        accuracy_summary_path,
    )

    print(
        "\n=== BASELINE DATASET SUMMARY ==="
    )

    print(
        dataset_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        "\nSummary directory:"
    )

    print(
        f"  {output_dir}"
    )


if __name__ == "__main__":
    main()
