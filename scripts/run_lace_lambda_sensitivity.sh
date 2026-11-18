#!/usr/bin/env bash

# Resolve paths relative to this checkout, even when invoked elsewhere.
LACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${LACE_ROOT}" || exit 1
export PYTHONPATH="${LACE_ROOT}/src:${LACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Run from the repository root:
#   bash scripts/run_lace_lambda_sensitivity.sh --dry-run
#   bash scripts/run_lace_lambda_sensitivity.sh
#
# Four alignment-weight experiments:
#   LACE_timesfm2 x c_traffic       x lambda {0.25, 0.5}
#   LACE_moirai2  x road_sensor_62  x lambda {0.25, 0.5}
# Use structural alignment with the optimizer, hierarchy, contexts, splits,
# bias weight, and seeds configured in the main LACE experiments.
# Each launch has a new output directory, including isolated evaluation arrays.

set -u
set -o pipefail

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
        printf 'Run four structural-alignment sensitivity experiments.\nUsage: bash scripts/run_lace_lambda_sensitivity.sh [--dry-run]\n'
        printf '\nOptional environment settings: GPU_ID, PYTHON_BIN, OUTPUT_BASE, RUN_TAG,\nONLY_DATASET, ONLY_LAMBDA. Run from the project root.\n'
        exit 0 ;;
    "") ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
esac
if (( $# )); then printf 'Unexpected extra arguments. Use --help.\n' >&2; exit 2; fi

PROJECT_ROOT="$(pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
ONLY_DATASET="${ONLY_DATASET:-}"
ONLY_LAMBDA="${ONLY_LAMBDA:-}"
OUTPUT_BASE="${OUTPUT_BASE:-./results/lace_lambda_sensitivity}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"

case "${ONLY_DATASET}" in
    ""|c_traffic|road_sensor_62) ;;
    *) printf 'ONLY_DATASET must be c_traffic or road_sensor_62.\n' >&2; exit 2 ;;
esac
case "${ONLY_LAMBDA}" in
    ""|0.25|0.5) ;;
    *) printf 'ONLY_LAMBDA must be 0.25 or 0.5.\n' >&2; exit 2 ;;
esac
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    printf 'RUN_TAG must contain only letters, digits, dots, underscores or hyphens.\n' >&2
    exit 2
fi
case "${OUTPUT_BASE}" in
    /*) QUEUE_DIR="${OUTPUT_BASE}/${RUN_TAG}" ;;
    *) QUEUE_DIR="${PROJECT_ROOT}/${OUTPUT_BASE}/${RUN_TAG}" ;;
esac

DATASETS=(c_traffic road_sensor_62)
LAMBDAS=(0.25 0.5)
MODE="structural"

configure_dataset() {
    case "$1" in
        c_traffic)
            MODEL_NAME="LACE_timesfm2"
            FACTORS=(1 6 36 144)
            TEST_TIMESTAMP="2024-04-07 00:00:00+0000"
            ROOT_PATH="./data/Network/"
            DATA_PATH="c10m_ideal.csv"
            TARGET="total_traffic_kbps"
            DATA_RESOLUTION="10m"
            BASE_MODEL_ID="c_traffic_144"
            SEQ_LEN=1024
            LABEL_LEN=512
            PRED_LEN=144
            ;;
        road_sensor_62)
            MODEL_NAME="LACE_moirai2"
            FACTORS=(1 3 12 288)
            TEST_TIMESTAMP="2018-02-17 00:00:00"
            ROOT_PATH="./data/Road/"
            DATA_PATH="road_sensor_62_5m.csv"
            TARGET="value"
            DATA_RESOLUTION="5m"
            BASE_MODEL_ID="road_sensor_62_288"
            SEQ_LEN=2016
            LABEL_LEN=1008
            PRED_LEN=288
            ;;
        *) return 2 ;;
    esac
}

build_train_args() {
    local dataset="$1" lambda="$2" tag="${2/./p}"
    RUN_ID="${MODEL_NAME}_${dataset}_lambda${tag}"
    DESCRIPTION="LACE_lambda${tag}_${dataset}_${MODEL_NAME}_${RUN_TAG}"
    TRAIN_ARGS=(
        --task_name multiscale_lace_forecast
        --is_training 1
        --inverse
        --data fixed_split
        --lace_factors "${FACTORS[@]}"
        --lace_adapter_dim 128
        --lace_representation_norm batch
        --lace_reduction sum
        --lace_coherence_mode "${MODE}"
        --lace_lambda "${lambda}"
        --data_name "${dataset}"
        --data_resolution "${DATA_RESOLUTION}"
        --test_split_timestamp "${TEST_TIMESTAMP}"
        --root_path "${ROOT_PATH}"
        --data_path "${DATA_PATH}"
        --target "${TARGET}"
        --model_id "${BASE_MODEL_ID}_lambda${tag}_${RUN_TAG}"
        --model "${MODEL_NAME}"
        --use_gpu
        --features S
        --patience 5
        --seq_len "${SEQ_LEN}"
        --label_len "${LABEL_LEN}"
        --pred_len "${PRED_LEN}"
        --batch_size 32
        --learning_rate 1e-5
        --train_epochs 50
        --des "${DESCRIPTION}"
        --itr 1
    )
}

# The uploaded experiment writes arrays under a constant imported from config,
# and its evaluator imports the same constant from src.config. Override only
# this output constant inside each child process; no project file is edited.
# Unlike changing data_name or coherence_mode, this keeps scientific settings
# and evaluator dataset dispatch unchanged.
scoped_python() {
    local phase="$1" evaluation_root="$2"
    shift 2
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u - \
        "${PROJECT_ROOT}" "${phase}" "${evaluation_root}" "$@" <<'PY'
import importlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
phase = sys.argv[2]
destination = str(Path(sys.argv[3]).resolve())
arguments = sys.argv[4:]
sys.path.insert(0, str(project))
if (project / 'src').is_dir():
    sys.path.append(str(project / 'src'))

patched = []
for name in ('config', 'src.config'):
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError as exc:
        if exc.name == 'src':
            continue
        raise
    if spec is None:
        continue
    if not spec.origin or not Path(spec.origin).resolve().is_relative_to(project):
        raise RuntimeError(f'{name} resolves outside this project; refusing ambiguous output configuration')
    module = importlib.import_module(name)
    if hasattr(module, 'EVALUATION_RESULTS_PATH'):
        module.EVALUATION_RESULTS_PATH = destination
        patched.append(name)
if not patched:
    raise RuntimeError('No project config with EVALUATION_RESULTS_PATH found; no training started')

print('Scoped output configuration:', json.dumps({'modules': patched, 'evaluation_root': destination}), flush=True)
if phase == 'train':
    # Preserve the entry point's own seed and argument handling.
    sys.argv = [str(project / 'run.py'), *arguments]
    runpy.run_path(str(project / 'run.py'), run_name='__main__')
elif phase == 'evaluate':
    module = importlib.import_module('evaluator.evaluate_lace')
    module.EVALUATION_RESULTS_PATH = destination
    sys.argv = ['evaluator.evaluate_lace', *arguments]
    module.main()
else:
    raise ValueError(f'Unknown phase: {phase}')
PY
}

printf 'LACE lambda sensitivity: four-run target queue\n'
printf 'Outputs: %s\n' "${QUEUE_DIR}"
printf 'Contexts and split timestamps match the main LACE launcher.\n'
printf 'Use existing lambda=0/1 controls only when their protocol matches.\n'

# Check all selected input files before any training begins.
if (( ! DRY_RUN )); then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { printf 'Python not found: %s\n' "${PYTHON_BIN}" >&2; exit 1; }
    [[ -f run.py ]] || { printf 'Run this script from the project root containing run.py.\n' >&2; exit 1; }
    "${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3,9), "Python 3.9+ required"' || exit 1
    for dataset in "${DATASETS[@]}"; do
        [[ -z "${ONLY_DATASET}" || "${dataset}" == "${ONLY_DATASET}" ]] || continue
        configure_dataset "${dataset}" || exit 1
        [[ -f "${ROOT_PATH}${DATA_PATH}" ]] || { printf 'Missing input: %s%s\nNo experiments started.\n' "${ROOT_PATH}" "${DATA_PATH}" >&2; exit 1; }
    done
    mkdir -p "$(dirname "${QUEUE_DIR}")" || exit 1
    mkdir "${QUEUE_DIR}" || { printf 'Use a new RUN_TAG; the queue folder must not already exist.\n' >&2; exit 1; }
    printf 'run_id\tmodel\tdataset\tlambda\tmode\tstatus\texit_code\toutput_folder\n' > "${QUEUE_DIR}/status.tsv"
fi

successes=()
failures=()
for dataset in "${DATASETS[@]}"; do
    [[ -z "${ONLY_DATASET}" || "${dataset}" == "${ONLY_DATASET}" ]] || continue
    configure_dataset "${dataset}" || exit 1
    for lambda in "${LAMBDAS[@]}"; do
        [[ -z "${ONLY_LAMBDA}" || "${lambda}" == "${ONLY_LAMBDA}" ]] || continue
        build_train_args "${dataset}" "${lambda}"
        run_dir="${QUEUE_DIR}/${RUN_ID}"
        evaluation_root="${run_dir}/evaluation_results"
        output_folder="${evaluation_root}/${MODEL_NAME}/${dataset}/${DATA_RESOLUTION}/${MODE}"
        printf '\n%s | lambda=%s | mode=%s | context=%s\n' "${MODEL_NAME}/${dataset}" "${lambda}" "${MODE}" "${SEQ_LEN}"
        printf '  Split: %s\n  Arrays/metrics: %s\n' "${TEST_TIMESTAMP}" "${output_folder}"
        printf '  Training arguments: '
        printf '%q ' "${PYTHON_BIN}" -u run.py "${TRAIN_ARGS[@]}"
        printf '\n'
        if (( DRY_RUN )); then continue; fi

        mkdir "${run_dir}" || exit 1
        {
            printf 'CUDA_VISIBLE_DEVICES=%q\n' "${GPU_ID}"
            printf 'evaluation_root=%q\n' "${evaluation_root}"
            printf 'command='; printf '%q ' "${PYTHON_BIN}" -u run.py "${TRAIN_ARGS[@]}"; printf '\n'
        } > "${run_dir}/command.txt"

        scoped_python train "${evaluation_root}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${run_dir}/training.log"
        pipeline_status=("${PIPESTATUS[@]}")
        status="trained"
        code="${pipeline_status[0]}"
        if (( code != 0 )); then
            status="training_failed"
        elif (( pipeline_status[1] != 0 )); then
            status="log_failed"; code="${pipeline_status[1]}"
        else
            for factor in "${FACTORS[@]}"; do
                for kind in pred true; do
                    if [[ ! -s "${output_folder}/${kind}_factor_${factor}.npy" ]]; then
                        printf 'Missing isolated output: %s/%s_factor_%s.npy\n' "${output_folder}" "${kind}" "${factor}"
                        status="outputs_missing"; code=1
                    fi
                done
            done
        fi

        if [[ "${status}" == "trained" ]]; then
            scoped_python evaluate "${evaluation_root}" \
                --dataset "${dataset}" --mode "${MODE}" --model_dir "${MODEL_NAME}" --plot \
                2>&1 | tee "${run_dir}/evaluation.log"
            pipeline_status=("${PIPESTATUS[@]}")
            code="${pipeline_status[0]}"
            if (( code != 0 )); then
                status="evaluation_failed"
            elif (( pipeline_status[1] != 0 )); then
                status="log_failed"; code="${pipeline_status[1]}"
            elif [[ ! -s "${output_folder}/${dataset}_${MODE}_evaluation.csv" ]]; then
                status="evaluation_outputs_missing"; code=1
            else
                status="complete"
            fi
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${RUN_ID}" "${MODEL_NAME}" "${dataset}" "${lambda}" "${MODE}" "${status}" "${code}" "${output_folder}" >> "${QUEUE_DIR}/status.tsv"
        if [[ "${status}" == "complete" ]]; then
            successes+=("${RUN_ID}")
        else
            failures+=("${RUN_ID}:${status}")
        fi
    done
done

if (( DRY_RUN )); then
    printf '\nDry run only: no files written and no training launched.\nInput/dependency checks run on launch.\n'
    exit 0
fi
printf '\nComplete: %s | Incomplete: %s\n' "${#successes[@]}" "${#failures[@]}"
for item in "${failures[@]}"; do printf '  %s\n' "${item}"; done
printf 'Status table and logs: %s\n' "${QUEUE_DIR}"
if (( ${#failures[@]} )); then exit 1; fi
exit 0
