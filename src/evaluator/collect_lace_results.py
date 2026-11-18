"""Collect LACE accuracy, coherence, and decision results across datasets and backbones.

Run from the repository root:
    python -m src.evaluator.collect_lace_results --help"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import EVALUATION_RESULTS_PATH
from src.evaluator.evaluate_lace import (
    get_dataset_config,
    load_lace_outputs,
    recover_daily_windows,
    nmse,
)
from src.utils.metrics import metric


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

CORE_COLUMNS = [
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
]


# ============================================================
# PATHS
# ============================================================

def expected_result_path(
    dataset,
    mode,
    road_data_root,
    model_dir="LACE_moirai1",
):
    cfg = get_dataset_config(
        dataset,
        road_data_root=road_data_root,
    )

    cfg = cfg.copy()
    cfg["model_dir"] = model_dir

    folder = os.path.join(
        EVALUATION_RESULTS_PATH,
        model_dir,
        dataset,
        cfg["base_resolution"],
        mode,
    )

    csv_path = os.path.join(
        folder,
        f"{dataset}_{mode}_evaluation.csv",
    )

    return cfg, folder, csv_path


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_result_table(df, dataset, mode, csv_path):
    missing = [
        col
        for col in CORE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            f"{csv_path} contains no evaluation rows."
        )

    datasets_in_file = set(
        df["Dataset"].astype(str).unique()
    )

    modes_in_file = set(
        df["Mode"].astype(str).unique()
    )

    if datasets_in_file != {dataset}:
        raise ValueError(
            f"{csv_path}: expected Dataset={dataset}, "
            f"found {sorted(datasets_in_file)}."
        )

    if modes_in_file != {mode}:
        raise ValueError(
            f"{csv_path}: expected Mode={mode}, "
            f"found {sorted(modes_in_file)}."
        )

    duplicated = df.duplicated(
        subset=[
            "Dataset",
            "Mode",
            "Pair",
        ],
        keep=False,
    )

    if duplicated.any():
        duplicate_pairs = (
            df.loc[
                duplicated,
                "Pair",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"{csv_path}: duplicate pair rows found: "
            f"{duplicate_pairs}"
        )


# ============================================================
# COLLECT COMPLETED RUNS
# ============================================================

def collect_results(
    datasets,
    modes,
    road_data_root,
    model_dir,
):
    tables = []
    status_rows = []

    for dataset in datasets:
        for mode in modes:

            try:
                cfg, folder, csv_path = expected_result_path(
                    dataset,
                    mode,
                    road_data_root,
                    model_dir=model_dir,
                )

            except Exception as exc:
                status_rows.append({
                    "Dataset": dataset,
                    "Backbone": model_dir,
                    "Mode": mode,
                    "Label": MODE_LABELS.get(
                        mode,
                        mode,
                    ),
                    "Status": "CONFIG_ERROR",
                    "Rows": 0,
                    "Path": "",
                    "Message": str(exc),
                })
                continue

            if not os.path.exists(csv_path):
                status_rows.append({
                    "Dataset": dataset,
                    "Backbone": model_dir,
                    "Mode": mode,
                    "Label": MODE_LABELS.get(
                        mode,
                        mode,
                    ),
                    "Status": "MISSING",
                    "Rows": 0,
                    "Path": csv_path,
                    "Message": "",
                })
                continue

            try:
                df = pd.read_csv(
                    csv_path
                )

                validate_result_table(
                    df,
                    dataset,
                    mode,
                    csv_path,
                )

                df = df.copy()

                df["Backbone"] = model_dir

                df["Mode_label"] = MODE_LABELS.get(
                    mode,
                    mode,
                )

                df["Base_resolution"] = cfg[
                    "base_resolution"
                ]

                df["Source_file"] = csv_path

                tables.append(
                    df
                )

                status_rows.append({
                    "Dataset": dataset,
                    "Backbone": model_dir,
                    "Mode": mode,
                    "Label": MODE_LABELS.get(
                        mode,
                        mode,
                    ),
                    "Status": "COMPLETE",
                    "Rows": len(df),
                    "Path": csv_path,
                    "Message": "",
                })

            except Exception as exc:
                status_rows.append({
                    "Dataset": dataset,
                    "Backbone": model_dir,
                    "Mode": mode,
                    "Label": MODE_LABELS.get(
                        mode,
                        mode,
                    ),
                    "Status": "INVALID",
                    "Rows": 0,
                    "Path": csv_path,
                    "Message": str(exc),
                })

    status = pd.DataFrame(
        status_rows
    )

    if not tables:
        return (
            pd.DataFrame(),
            status,
        )

    master = pd.concat(
        tables,
        ignore_index=True,
        sort=False,
    )

    dataset_order = {
        name: i
        for i, name in enumerate(
            datasets
        )
    }

    mode_order = {
        name: i
        for i, name in enumerate(
            modes
        )
    }

    master["_dataset_order"] = (
        master["Dataset"]
        .map(dataset_order)
        .fillna(999)
    )

    master["_mode_order"] = (
        master["Mode"]
        .map(mode_order)
        .fillna(999)
    )

    master = (
        master
        .sort_values(
            [
                "_dataset_order",
                "_mode_order",
                "Fine_factor",
                "Coarse_factor",
            ]
        )
        .drop(
            columns=[
                "_dataset_order",
                "_mode_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        master,
        status,
    )


# ============================================================
# UNIQUE PER-RESOLUTION ACCURACY
# ============================================================

def build_resolution_accuracy(
    master,
    road_data_root="./data/Road",
    model_dir="LACE_moirai1",
):
    """
    Compute one complete accuracy record per dataset / mode /
    resolution using the SAME non-overlapping daily forecast origins
    used by evaluate_lace.py.

    Use non-overlapping daily origins for comparison with the baseline forecasts.

    Alignment:
        1. Load every factor for one dataset/mode.
        2. Use the common base-resolution one-day stride.
        3. Trim every factor to the same number of recovered days.
        4. Compute MAE/MSE/RMSE/MAPE/MSPE/NMSE on those daily windows.

    Metrics:
        MAE
        MSE
        RMSE
        MAPE
        MSPE
        NMSE

    NMSE:
        MSE / Var(true)
    """
    if master.empty:
        return pd.DataFrame()

    rows = []

    completed_runs = (
        master[
            [
                "Dataset",
                "Mode",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for _, run in completed_runs.iterrows():

        dataset = str(
            run["Dataset"]
        )

        mode = str(
            run["Mode"]
        )

        cfg, folder, _ = expected_result_path(
            dataset=dataset,
            mode=mode,
            road_data_root=road_data_root,
            model_dir=model_dir,
        )

        factor_to_res = cfg[
            "factor_to_res"
        ]

        # ----------------------------------------------------

        # Reuse the exact LACE evaluator loading/alignment logic.
        #
        # Every LACE resolution head is produced at the same
        # base-resolution forecast origins, so all factors use
        # the SAME base one-day stride.
        # ----------------------------------------------------
        preds, trues = load_lace_outputs(
            folder,
            factor_to_res,
        )

        daily_pred, daily_true = recover_daily_windows(
            preds,
            trues,
            cfg["daily_stride"],
        )

        n_days = min(
            len(x)
            for x in daily_true.values()
        )

        if n_days < 1:
            raise RuntimeError(
                f"No common daily windows for {dataset}/{mode}."
            )

        for factor, resolution in factor_to_res.items():

            pred = daily_pred[
                factor
            ]

            true = daily_true[
                factor
            ]

            if pred.shape != true.shape:
                raise RuntimeError(
                    f"Shape mismatch for "
                    f"{dataset}/{mode}/{resolution}: "
                    f"pred={pred.shape}, "
                    f"true={true.shape}"
                )

            if len(pred) != n_days:
                raise RuntimeError(
                    f"Common-day mismatch for "
                    f"{dataset}/{mode}/{resolution}: "
                    f"{len(pred)} != {n_days}"
                )

            if not np.isfinite(pred).all():
                raise RuntimeError(
                    f"Non-finite predictions for "
                    f"{dataset}/{mode}/{resolution}."
                )

            if not np.isfinite(true).all():
                raise RuntimeError(
                    f"Non-finite truth values for "
                    f"{dataset}/{mode}/{resolution}."
                )

            mae, mse_v, rmse, mape, mspe = metric(
                pred,
                true,
            )

            nmse_v = nmse(
                pred,
                true,
            )

            rows.append({
                "Backbone":
                    model_dir,

                "Dataset":
                    dataset,

                "Mode":
                    mode,

                "Mode_label":
                    MODE_LABELS.get(
                        mode,
                        mode,
                    ),

                "Factor":
                    int(factor),

                "Resolution":
                    resolution,

                "N_days":
                    int(n_days),

                "MSE":
                    float(mse_v),

                "MAE":
                    float(mae),

                "RMSE":
                    float(rmse),

                "MAPE":
                    float(mape),

                "MSPE":
                    float(mspe),

                "NMSE":
                    float(nmse_v),
            })

    result = pd.DataFrame(
        rows
    )

    return (
        result
        .sort_values(
            [
                "Dataset",
                "Mode",
                "Factor",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# A1 vs A0 PAIRWISE COMPARISON
# ============================================================

def build_a1_vs_a0_pairwise(master):
    required_modes = {
        "none",
        "structural",
    }

    if master.empty:
        return pd.DataFrame()

    if not required_modes.issubset(
        set(master["Mode"].unique())
    ):
        return pd.DataFrame()

    metrics = [
        "E_fine",
        "E_agg",
        "E_dir",
        "C_rs",
        "Decision_mismatch",
        "Delta_E",
        "Delta_D",
    ]

    # Dataset-specific correctness metrics.
    # These exist for road datasets and will be NaN elsewhere.
    for metric in [
        "Agg_decision_accuracy",
        "Direct_decision_accuracy",
    ]:
        if metric in master.columns:
            metrics.append(metric)

    base_columns = [
        "Backbone",
        "Dataset",
        "Pair",
        "Fine_factor",
        "Coarse_factor",
    ]

    a0 = (
        master[
            master["Mode"] == "none"
        ][
            base_columns
            + metrics
        ]
        .copy()
    )

    a1 = (
        master[
            master["Mode"] == "structural"
        ][
            base_columns
            + metrics
        ]
        .copy()
    )

    a0 = a0.rename(
        columns={
            metric:
                f"A0_{metric}"
            for metric in metrics
        }
    )

    a1 = a1.rename(
        columns={
            metric:
                f"A1_{metric}"
            for metric in metrics
        }
    )

    paired = a0.merge(
        a1,
        on=base_columns,
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        return paired

    # --------------------------------------------------------
    # Signed changes:
    #
    # For C_rs, errors, and decision mismatch:
    #   negative Change_* = A1 is lower/better.
    #
    # Improvement_* is defined so:
    #   positive = A1 improved.
    # --------------------------------------------------------

    lower_is_better = [
        "E_fine",
        "E_agg",
        "E_dir",
        "C_rs",
        "Decision_mismatch",
    ]

    for metric in lower_is_better:
        paired[
            f"Change_{metric}"
        ] = (
            paired[f"A1_{metric}"]
            -
            paired[f"A0_{metric}"]
        )

        paired[
            f"Improvement_{metric}"
        ] = (
            paired[f"A0_{metric}"]
            -
            paired[f"A1_{metric}"]
        )

    # --------------------------------------------------------
    # Higher-is-better decision correctness metrics
    # --------------------------------------------------------
    higher_is_better = [
        "Agg_decision_accuracy",
        "Direct_decision_accuracy",
    ]

    for metric in higher_is_better:
        a0_col = f"A0_{metric}"
        a1_col = f"A1_{metric}"

        if (
            a0_col in paired.columns
            and a1_col in paired.columns
        ):
            paired[
                f"Change_{metric}"
            ] = (
                paired[a1_col]
                -
                paired[a0_col]
            )

            # Positive = A1 improved correctness.
            paired[
                f"Improvement_{metric}"
            ] = (
                paired[a1_col]
                -
                paired[a0_col]
            )

    paired[
        "C_rs_reduction_pct"
    ] = (
        100.0
        * (
            paired["A0_C_rs"]
            -
            paired["A1_C_rs"]
        )
        /
        paired["A0_C_rs"].replace(
            0,
            np.nan,
        )
    )

    paired[
        "Decision_mismatch_reduction_pct"
    ] = (
        100.0
        * (
            paired["A0_Decision_mismatch"]
            -
            paired["A1_Decision_mismatch"]
        )
        /
        paired[
            "A0_Decision_mismatch"
        ].replace(
            0,
            np.nan,
        )
    )

    # Delta_E and Delta_D are signed route-advantage quantities,
    # not "lower is better" model metrics. Preserve only raw change.
    paired[
        "Change_Delta_E"
    ] = (
        paired["A1_Delta_E"]
        -
        paired["A0_Delta_E"]
    )

    paired[
        "Change_Delta_D"
    ] = (
        paired["A1_Delta_D"]
        -
        paired["A0_Delta_D"]
    )

    return paired


# ============================================================
# A1 vs A0 UNIQUE-RESOLUTION ACCURACY
# ============================================================

def build_a1_vs_a0_accuracy(resolution_accuracy):
    if resolution_accuracy.empty:
        return pd.DataFrame()

    a0 = (
        resolution_accuracy[
            resolution_accuracy["Mode"] == "none"
        ][
            [
                "Backbone",
                "Dataset",
                "Factor",
                "Resolution",
                "NMSE",
            ]
        ]
        .rename(
            columns={
                "NMSE": "A0_NMSE",
            }
        )
    )

    a1 = (
        resolution_accuracy[
            resolution_accuracy["Mode"] == "structural"
        ][
            [
                "Backbone",
                "Dataset",
                "Factor",
                "Resolution",
                "NMSE",
            ]
        ]
        .rename(
            columns={
                "NMSE": "A1_NMSE",
            }
        )
    )

    paired = a0.merge(
        a1,
        on=[
            "Backbone",
            "Dataset",
            "Factor",
            "Resolution",
        ],
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        return paired

    paired[
        "Change_NMSE"
    ] = (
        paired["A1_NMSE"]
        -
        paired["A0_NMSE"]
    )

    paired[
        "Improvement_NMSE"
    ] = (
        paired["A0_NMSE"]
        -
        paired["A1_NMSE"]
    )

    paired[
        "NMSE_change_pct"
    ] = (
        100.0
        * paired["Change_NMSE"]
        /
        paired["A0_NMSE"].replace(
            0,
            np.nan,
        )
    )

    return paired


# ============================================================
# DATASET-LEVEL A1 vs A0 DIAGNOSTIC SUMMARY
# ============================================================

def build_dataset_summary(pairwise):
    if pairwise.empty:
        return pd.DataFrame()

    rows = []

    for dataset, group in pairwise.groupby(
        "Dataset",
        sort=False,
    ):
        decision_mask = (
            group[
                "A0_Decision_mismatch"
            ].notna()
            &
            group[
                "A1_Decision_mismatch"
            ].notna()
        )

        a0_mean_c = group["A0_C_rs"].mean()
        a1_mean_c = group["A1_C_rs"].mean()

        row = {
            "Backbone":
                group["Backbone"].iloc[0]
                if "Backbone" in group.columns
                else None,

            "Dataset": dataset,

            "Pairs_compared":
                len(group),

            "Decision_pairs_compared":
                int(decision_mask.sum()),

            "A0_mean_C_rs":
                a0_mean_c,

            "A1_mean_C_rs":
                a1_mean_c,

            "Mean_C_rs_improvement":
                a0_mean_c - a1_mean_c,

            # Dataset-level reduction, not mean of
            # pairwise percentages.
            "Mean_C_rs_reduction_pct":
                (
                    100.0
                    * (a0_mean_c - a1_mean_c)
                    / a0_mean_c
                    if a0_mean_c != 0
                    else np.nan
                ),

            "Mean_pairwise_E_dir_change":
                group[
                    "Change_E_dir"
                ].mean(),
        }

        if decision_mask.any():
            decision_group = group.loc[
                decision_mask
            ]

            a0_mismatch = (
                decision_group[
                    "A0_Decision_mismatch"
                ].mean()
            )

            a1_mismatch = (
                decision_group[
                    "A1_Decision_mismatch"
                ].mean()
            )

            row.update({
                "A0_mean_decision_mismatch":
                    a0_mismatch,

                "A1_mean_decision_mismatch":
                    a1_mismatch,

                "Mean_decision_mismatch_improvement":
                    a0_mismatch - a1_mismatch,

                # Dataset-level percentage.
                "Mean_decision_mismatch_reduction_pct":
                    (
                        100.0
                        * (
                            a0_mismatch
                            - a1_mismatch
                        )
                        / a0_mismatch
                        if a0_mismatch != 0
                        else np.nan
                    ),
            })

            # ----------------------------------------------
            # Road decision correctness
            # ----------------------------------------------
            required_accuracy_cols = [
                "A0_Agg_decision_accuracy",
                "A1_Agg_decision_accuracy",
                "A0_Direct_decision_accuracy",
                "A1_Direct_decision_accuracy",
            ]

            if all(
                col in decision_group.columns
                for col in required_accuracy_cols
            ):
                a0_agg_acc = (
                    decision_group[
                        "A0_Agg_decision_accuracy"
                    ].mean()
                )

                a1_agg_acc = (
                    decision_group[
                        "A1_Agg_decision_accuracy"
                    ].mean()
                )

                a0_dir_acc = (
                    decision_group[
                        "A0_Direct_decision_accuracy"
                    ].mean()
                )

                a1_dir_acc = (
                    decision_group[
                        "A1_Direct_decision_accuracy"
                    ].mean()
                )

                row.update({
                    "A0_mean_agg_decision_accuracy":
                        a0_agg_acc,

                    "A1_mean_agg_decision_accuracy":
                        a1_agg_acc,

                    "Agg_decision_accuracy_change":
                        a1_agg_acc - a0_agg_acc,

                    "A0_mean_direct_decision_accuracy":
                        a0_dir_acc,

                    "A1_mean_direct_decision_accuracy":
                        a1_dir_acc,

                    "Direct_decision_accuracy_change":
                        a1_dir_acc - a0_dir_acc,
                })

        else:
            row.update({
                "A0_mean_decision_mismatch":
                    np.nan,
                "A1_mean_decision_mismatch":
                    np.nan,
                "Mean_decision_mismatch_improvement":
                    np.nan,
                "Mean_decision_mismatch_reduction_pct":
                    np.nan,
            })

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# COMPLETION REPORT
# ============================================================

def print_completion(status):
    print(
        "\n============================================================"
    )
    print(
        "LACE MASTER RESULT STATUS"
    )
    print(
        "============================================================"
    )

    if status.empty:
        print(
            "No expected runs were configured."
        )
        return

    display = status[
        [
            "Backbone",
            "Dataset",
            "Label",
            "Status",
            "Rows",
        ]
    ]

    print(
        display.to_string(
            index=False
        )
    )

    complete = (
        status["Status"]
        == "COMPLETE"
    ).sum()

    total = len(
        status
    )

    print(
        f"\nCompleted: {complete}/{total}"
    )

    problems = status[
        status["Status"]
        != "COMPLETE"
    ]

    if not problems.empty:
        print(
            "\nIncomplete / invalid runs:"
        )

        for _, row in problems.iterrows():
            message = (
                f" | {row['Message']}"
                if row["Message"]
                else ""
            )

            print(
                f"  {row['Dataset']} "
                f"{row['Label']}: "
                f"{row['Status']}"
                f"{message}"
            )


# ============================================================
# PLOTS
# ============================================================

def add_identity_line(ax, x, y):
    finite = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if not finite.any():
        return

    low = min(
        np.min(x[finite]),
        np.min(y[finite]),
    )

    high = max(
        np.max(x[finite]),
        np.max(y[finite]),
    )

    if np.isclose(
        low,
        high,
    ):
        pad = (
            abs(low) * 0.05
            + 1e-6
        )
        low -= pad
        high += pad

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1,
    )


def plot_a0_vs_a1_coherence(pairwise, output_dir):
    if pairwise.empty:
        return

    x = pairwise[
        "A0_C_rs"
    ].to_numpy(
        dtype=float
    )

    y = pairwise[
        "A1_C_rs"
    ].to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        x,
        y,
        alpha=0.8,
    )

    add_identity_line(
        ax,
        x,
        y,
    )

    ax.set_xlabel(
        "A0 cross-resolution inconsistency"
    )

    ax.set_ylabel(
        "A1 cross-resolution inconsistency"
    )

    ax.set_title(
        "A0 vs. A1 forecast inconsistency"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    path = os.path.join(
        output_dir,
        "A0_vs_A1_C_rs.png",
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        path,
    )


def plot_a0_vs_a1_decisions(pairwise, output_dir):
    if pairwise.empty:
        return

    plot_df = pairwise.dropna(
        subset=[
            "A0_Decision_mismatch",
            "A1_Decision_mismatch",
        ]
    )

    if plot_df.empty:
        return

    x = plot_df[
        "A0_Decision_mismatch"
    ].to_numpy(
        dtype=float
    )

    y = plot_df[
        "A1_Decision_mismatch"
    ].to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        x,
        y,
        alpha=0.8,
    )

    add_identity_line(
        ax,
        x,
        y,
    )

    ax.set_xlabel(
        "A0 decision mismatch"
    )

    ax.set_ylabel(
        "A1 decision mismatch"
    )

    ax.set_title(
        "A0 vs. A1 decision disagreement"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    path = os.path.join(
        output_dir,
        "A0_vs_A1_decision_mismatch.png",
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        path,
    )


def plot_coherence_accuracy_tradeoff(
    pairwise,
    output_dir,
):
    if pairwise.empty:
        return

    plot_df = pairwise.dropna(
        subset=[
            "Improvement_C_rs",
            "Change_E_dir",
        ]
    )

    if plot_df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        plot_df[
            "Improvement_C_rs"
        ],
        plot_df[
            "Change_E_dir"
        ],
        alpha=0.8,
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
        "A1 coherence improvement (A0 C_rs - A1 C_rs)"
    )

    ax.set_ylabel(
        "A1 direct-NMSE change (A1 - A0)"
    )

    ax.set_title(
        "Structural coherence vs. accuracy change"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    path = os.path.join(
        output_dir,
        "A1_coherence_accuracy_tradeoff.png",
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        path,
    )


# ============================================================
# SAVE OUTPUTS
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

def merge_existing_groups(
    new_df,
    path,
    group_columns,
):
    """
    Preserve previously collected dataset/mode groups while
    replacing groups included in the current collection.

    Replacing whole groups avoids stale pair rows after reruns.
    """
    if (
        new_df.empty
        or not os.path.exists(path)
    ):
        return new_df.copy()

    old_df = pd.read_csv(path)

    if old_df.empty:
        return new_df.copy()

    if not all(
        col in old_df.columns
        and col in new_df.columns
        for col in group_columns
    ):
        return new_df.copy()

    new_groups = set(
        map(
            tuple,
            new_df[group_columns]
            .astype(str)
            .to_numpy(),
        )
    )

    old_group_keys = (
        old_df[group_columns]
        .astype(str)
        .apply(tuple, axis=1)
    )

    old_df = old_df.loc[
        ~old_group_keys.isin(new_groups)
    ]

    return pd.concat(
        [
            old_df,
            new_df,
        ],
        ignore_index=True,
        sort=False,
    )

def merge_existing_table(
    new_df,
    path,
    key_columns,
):
    """
    Merge newly collected rows into an existing master table.

    Existing rows with the same key are replaced by the new rows.
    Previously collected datasets are preserved.
    """
    if not os.path.exists(path):
        return new_df.copy()

    old_df = pd.read_csv(path)

    combined = pd.concat(
        [
            old_df,
            new_df,
        ],
        ignore_index=True,
        sort=False,
    )

    valid_keys = [
        col
        for col in key_columns
        if col in combined.columns
    ]

    if valid_keys:
        combined = combined.drop_duplicates(
            subset=valid_keys,
            keep="last",
        )

    return combined.reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect LACE evaluator outputs and build "
            "A0/A1 master comparisons."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=(
            "Datasets to scan. Defaults to the full current "
            "LACE experiment set."
        ),
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=[
            "none",
            "structural",
        ],
        help=(
            "Modes to scan. Defaults to none structural."
        ),
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
            "LACE result/model directory to collect."
        ),
    )

    parser.add_argument(
        "--road_data_root",
        default="./data/Road",
        help=(
            "Passed to the shared road dataset configuration."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Summary directory. Default: "
            "EVALUATION_RESULTS_PATH/<model_dir>/_summary"
        ),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Create pooled A0-vs-A1 diagnostic figures."
        ),
    )

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = os.path.join(
            EVALUATION_RESULTS_PATH,
            args.model_dir,
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

    master, status = collect_results(
        datasets=args.datasets,
        modes=args.modes,
        road_data_root=args.road_data_root,
        model_dir=args.model_dir,
    )

    print_completion(
        status
    )

    status_path = os.path.join(
        output_dir,
        "LACE_run_status.csv",
    )

    status = merge_existing_table(
        status,
        status_path,
        key_columns=[
            "Backbone",
            "Dataset",
            "Mode",
        ],
    )

    save_table(
        status,
        status_path,
    )

    if master.empty:
        print(
            "\nNo completed valid evaluation CSVs were found."
        )
        return

    master_path = os.path.join(
        output_dir,
        "LACE_all_results.csv",
    )

    master = merge_existing_table(
        master,
        master_path,
        key_columns=[
            "Backbone",
            "Dataset",
            "Mode",
            "Pair",
        ],
    )

    master = merge_existing_groups(
        master,
        master_path,
        group_columns=[
            "Backbone",
            "Dataset",
            "Mode",
        ],
    )

    save_table(
        master,
        master_path,
    )

    resolution_accuracy = (
        build_resolution_accuracy(
            master,
            road_data_root=args.road_data_root,
            model_dir=args.model_dir,
        )
    )

    resolution_path = os.path.join(
        output_dir,
        "LACE_resolution_accuracy.csv",
    )

    save_table(
        resolution_accuracy,
        resolution_path,
    )

    pairwise = (
        build_a1_vs_a0_pairwise(
            master
        )
    )

    if not pairwise.empty:
        pairwise_path = os.path.join(
            output_dir,
            "LACE_A1_vs_A0_pairwise.csv",
        )

        save_table(
            pairwise,
            pairwise_path,
        )

        dataset_summary = (
            build_dataset_summary(
                pairwise
            )
        )

        summary_path = os.path.join(
            output_dir,
            "LACE_A1_vs_A0_dataset_summary.csv",
        )

        save_table(
            dataset_summary,
            summary_path,
        )

        accuracy_comparison = (
            build_a1_vs_a0_accuracy(
                resolution_accuracy
            )
        )

        accuracy_path = os.path.join(
            output_dir,
            "LACE_A1_vs_A0_accuracy.csv",
        )

        save_table(
            accuracy_comparison,
            accuracy_path,
        )

        print(
            "\n=== A1 vs A0 DATASET SUMMARY ==="
        )

        print(
            dataset_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        if args.plot:
            plot_a0_vs_a1_coherence(
                pairwise,
                output_dir,
            )

            plot_a0_vs_a1_decisions(
                pairwise,
                output_dir,
            )

            plot_coherence_accuracy_tradeoff(
                pairwise,
                output_dir,
            )

    else:
        print(
            "\nA0/A1 paired comparison is not available yet. "
            "The master/status files were still generated."
        )

    print(
        "\nSummary directory:"
    )

    print(
        f"  {output_dir}"
    )


if __name__ == "__main__":
    main()
