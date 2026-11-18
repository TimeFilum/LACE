"""Summarize A0-vs-A1 coherence and decision changes for each backbone."""
import os
import numpy as np
import pandas as pd

from src.config import EVALUATION_RESULTS_PATH


MODELS = [
    "LACE_moirai1",
    "LACE_moirai2",
    "LACE_chronos2",
    "LACE_timesfm2",
]


def main():
    rows = []

    for model in MODELS:

        path = os.path.join(
            EVALUATION_RESULTS_PATH,
            model,
            "_summary",
            "LACE_A1_vs_A0_dataset_summary.csv",
        )

        if not os.path.exists(path):
            print(f"[MISSING SUMMARY] {model}")
            continue

        df = pd.read_csv(path)

        for _, row in df.iterrows():

            a0_c = row["A0_mean_C_rs"]
            a1_c = row["A1_mean_C_rs"]

            a0_d = row["A0_mean_decision_mismatch"]
            a1_d = row["A1_mean_decision_mismatch"]

            coherence_improved = (
                np.isfinite(a0_c)
                and np.isfinite(a1_c)
                and a1_c < a0_c
            )

            decision_improved = (
                np.isfinite(a0_d)
                and np.isfinite(a1_d)
                and a1_d < a0_d
            )

            rows.append({
                "Backbone": model,
                "Dataset": row["Dataset"],

                "A0_C": a0_c,
                "A1_C": a1_c,
                "C_change": a1_c - a0_c,
                "C_reduction_pct":
                    row["Mean_C_rs_reduction_pct"],

                "Coherence_improved":
                    coherence_improved,

                "A0_decision_mismatch": a0_d,
                "A1_decision_mismatch": a1_d,
                "Decision_change": a1_d - a0_d,
                "Decision_reduction_pct":
                    row[
                        "Mean_decision_mismatch_reduction_pct"
                    ],

                "Decision_agreement_improved":
                    decision_improved,

                "Both_improved":
                    coherence_improved
                    and decision_improved,
            })


    result = pd.DataFrame(rows)
    if result.empty:
        print("No summaries found; collect LACE results first.")
        return

    print(
        result[
            [
                "Backbone",
                "Dataset",
                "C_reduction_pct",
                "Coherence_improved",
                "Decision_reduction_pct",
                "Decision_agreement_improved",
                "Both_improved",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


    # ------------------------------------------------------------
    # Per-backbone success counts
    # ------------------------------------------------------------

    if not result.empty:

        summary = (
            result
            .groupby(
                "Backbone",
                as_index=False,
            )
            .agg(
                Datasets=("Dataset", "count"),

                Coherence_improved=(
                    "Coherence_improved",
                    "sum",
                ),

                Decision_improved=(
                    "Decision_agreement_improved",
                    "sum",
                ),

                Both_improved=(
                    "Both_improved",
                    "sum",
                ),
            )
        )

        print(
            "\n=== BACKBONE SUMMARY ==="
        )

        print(
            summary.to_string(
                index=False
            )
        )


    output_path = os.path.join(
        EVALUATION_RESULTS_PATH,
        "LACE_backbone_main_effects.csv",
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved:",
        output_path,
    )


if __name__ == "__main__":
    main()
