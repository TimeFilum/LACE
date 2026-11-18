# Reproduction guide

Run commands from the repository root. The `.sh` launchers use Bash and set their
import paths automatically; on Windows, you can run them in Git Bash or WSL.

## Environment

The tested environment uses **Python 3.11.15**, **PyTorch 2.6.0**, and
**uni2ts 2.0.0**. Package versions are listed in
[`configs/research_environment.json`](../configs/research_environment.json).
Install the PyTorch 2.6.0 build that matches your CPU or CUDA setup, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-backbones.txt
```

`requirements-research.txt` adds the Lightning, GluonTS, and JAX version pins from
the research environment. Record the CUDA runtime, GPU driver, and pretrained
checkpoint revisions alongside your results when comparing runs across machines.

LACE loads the package for the selected backbone:

| Backbone | Package |
|---|---|
| Moirai-1 and Moirai-2 | `uni2ts` |
| Chronos-2 | `chronos-forecasting` |
| TimesFM-2 | `transformers` |

The independent TimesFM baseline uses `timesfm`. For experiment tracking with
`--use_wandb`, install the optional dependency:

```bash
python -m pip install -r requirements-optional.txt
```

Capture your installed package versions with:

```bash
python tools/capture_environment.py > research_environment.json
```

## Data

Follow [DATASETS.md](DATASETS.md) to obtain and prepare the input CSVs. The
launchers read from `data/Network/`, `data/Electricity/`, and `data/Road/`.
The LACE queues cover the full study grid by default; use `ONLY_DATASET` to select
a dataset you have prepared.

## Train LACE

The paper compares accuracy-only adaptation (A0) with structural alignment (A1):

| Mode | Arguments |
|---|---|
| A0: accuracy only | `--lace_coherence_mode none --lace_lambda 0` |
| A1: structural alignment | `--lace_coherence_mode structural --lace_lambda 1` |

The runner defaults to `none`. Set the mode to `structural` to enable alignment.

For ETTm1 with Moirai-1, run both modes followed by evaluation:

```bash
ONLY_DATASET=ETTm1_HUFL MODEL_NAME=LACE_moirai1 bash scripts/run_lace_overnight.sh
```

Add `ONLY_MODE=structural` to run A1 alone. Set `MODEL_NAME` to
`LACE_moirai2`, `LACE_chronos2`, or `LACE_timesfm2` to select another backbone.
`GPU_ID` selects the GPU index (default `0`), and `PYTHON_BIN` selects the Python
executable.

Preview the selected commands with `--dry-run`:

```bash
ONLY_DATASET=ETTm1_HUFL MODEL_NAME=LACE_moirai1 bash scripts/run_lace_overnight.sh --dry-run
```

To run the full study grid:

```bash
bash scripts/run_lace_overnight.sh
bash scripts/run_lace_backbones_overnight.sh
```

The first launcher runs Moirai-1 across six datasets in A0 and A1. The second
runs the other three backbones. The defaults are Adam with learning rate `1e-5`,
batch size `32`, up to `50` epochs, patience `5`, adapter dimension `128`,
non-affine BatchNorm, and sum aggregation.
[`configs/paper_protocol.json`](../configs/paper_protocol.json) lists the 48
training configurations. The offline tests check their agreement with the launchers.

**Seeds:** `src/run.py` initializes Python, NumPy, and Torch with `2021`. Importing
the LACE experiment then sets Torch and NumPy to `42`. These explicit
initializations determine the run seeds; the parsed `--seed` argument leaves them
unchanged. Update the initialization points when running additional seeds.

## Independent forecasts and reconciliation

Select the dataset and model entries in the zero-shot launchers, then run the
script for your domain:

```bash
bash scripts/run_foundation_models_network.sh
bash scripts/run_foundation_models_electricity.sh
bash scripts/run_foundation_models_road.sh
```

The independent daily forecasts use `28` observations of history. At the finest
C-traffic resolution, the zero-shot context is `1008` observations and the LACE
context is `1024`.

Inspect the zero-shot and A0 reconciliation controls with:

```bash
python -m src.evaluator.reconcile_controls_lace --help
```

Use `--source phase0` for independent TSFM forecasts or `--source a0` for
accuracy-only LACE forecasts. For A0, WLS-Struct is the default; pass
`--methods bottomup mint_struct` to generate both controls.

For the supervised baselines:

```bash
MODEL=DLinear bash scripts/run_deep_learning_baselines.sh
MODEL=LSTM bash scripts/run_deep_learning_baselines.sh
python -m src.evaluator.reconcile_baselines --help
python -m src.evaluator.evaluate_baseline --help
```

In supervised reconciliation, `--method mint_struct` selects WLS-Struct with
structural scaling; `--method ols` selects OLS.

## Evaluation and tables

Evaluate the ETTm1 structural-alignment forecasts with:

```bash
python -m src.evaluator.evaluate_lace --dataset ETTm1_HUFL --mode structural --model_dir LACE_moirai1 --plot
```

The evaluation and analysis entry points provide their options through `--help`:

```bash
python -m src.evaluator.collect_lace_results --help
python -m src.evaluator.collect_baseline_results --help
python -m src.evaluator.collect_final_plot_matrix --help
python -m src.evaluator.bootstrap_lace_significance --help
python -m src.evaluator.check_lace_integrity --help
python -m src.evaluator.plot_accuracy_coherence_frontier --help
```

The evaluator uses the resolution pairs configured for each dataset. The final
collector produces the paper's adjacent-edge summaries. Road daily accuracy is
included, while road decision summaries use the finer-resolution edges.
The bootstrap resamples paired test windows to estimate uncertainty.

LACE and baseline forecasts and summaries go to `results/Evaluation/`;
independent TSFM predictions go to `results/phase0/`. Use `--inverse` to save the
physical-unit predictions used by the decision policies. The launch scripts
enable this option. For sum aggregation, inverse scaling is
`std * prediction + factor * mean`, using the finest-resolution training statistics.

## Alignment-weight sensitivity

Run the configured sensitivity experiments with:

```bash
bash scripts/run_lace_lambda_sensitivity.sh --dry-run
bash scripts/run_lace_lambda_sensitivity.sh
```

The launcher covers TimesFM-2 on C-traffic and Moirai-2 on road sensor 62, each
with alignment weights `0.25` and `0.5`.

## Checks

Run the offline regression tests and Python syntax checks:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q src tools tests
```

The offline tests use deterministic frozen encoders to check the adaptation and
training pipeline, aggregation, gradients, and checkpoint handling.

The backbone integration checks load pretrained models and may download weights.
In Bash, add `src` to the Python import path, then run:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tests/integration/test_moirai1_mean_last_context.py
python tests/integration/test_moirai2_mean_last_context.py
python tests/integration/test_chronos2_mean_last_context.py
python tests/integration/test_timesfm2_mean_last_context.py
```
