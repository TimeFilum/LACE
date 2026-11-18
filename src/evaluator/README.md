# Evaluation

Run commands from the repository root. Evaluate saved LACE forecasts with:

```bash
python -m src.evaluator.evaluate_lace --dataset ETTm1_HUFL --mode structural --model_dir LACE_moirai1 --plot
```

| File | Purpose |
|---|---|
| `evaluate_lace.py` | LACE accuracy, consistency, and decision metrics |
| `evaluate_baseline.py` | Independent and reconciled baseline evaluation |
| `reconcile_controls_lace.py` | Bottom-Up and WLS-Struct for zero-shot and A0 forecasts |
| `reconcile_baselines.py` | WLS-Struct and OLS for supervised forecasts |
| `collect_lace_results.py` | LACE result tables and comparisons |
| `collect_baseline_results.py` | Baseline result tables |
| `collect_final_plot_matrix.py` | Paper comparison tables and plotting data |
| `bootstrap_lace_significance.py` | Paired moving-block bootstrap intervals |
| `check_lace_integrity.py` | Forecast, metric, log, and checkpoint checks |
| `plot_accuracy_coherence_frontier.py` | Accuracy–consistency comparison plots |
| `compare_lace_backbones.py` | Backbone-level A0-vs-A1 summary |

The reconciliation and analysis commands expose their options through `--help`.
For the experiment sequence and settings, see the
[reproduction guide](../../docs/REPRODUCIBILITY.md).
