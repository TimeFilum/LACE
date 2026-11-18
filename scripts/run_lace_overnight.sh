#!/usr/bin/env bash

# Resolve paths relative to this checkout, even when invoked elsewhere.
LACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${LACE_ROOT}" || exit 1
export PYTHONPATH="${LACE_ROOT}/src:${LACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# ============================================================
# LACE OVERNIGHT A0/A1 RUNNER
# ============================================================
#
# Run from the project root:
#   bash scripts/run_lace_overnight.sh
#
# Sequentially runs:
#   A0 = accuracy only      (none,       lambda=0)
#   A1 = structural LACE    (structural, lambda=1)
#
# for:
#   c_traffic
#   k_traffic
#   ETTm1_HUFL
#   road_sensor_62
#   road_sensor_103
#   road_sensor_240
#
# Every run gets its own log. A failed run is recorded and the
# queue continues instead of killing the whole overnight batch.
# ============================================================

set -u

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    "") ;;
    *) echo "Usage: bash scripts/run_lace_overnight.sh [--dry-run]" >&2; exit 2 ;;
esac

MODEL_NAME="${MODEL_NAME:-LACE_moirai1}"
GPU_ID="${GPU_ID:-0}"

# Optional selective-run filters.
# Leave empty to run the full DATASETS x MODES matrix.
ONLY_DATASET="${ONLY_DATASET:-}"
ONLY_MODE="${ONLY_MODE:-}"

BATCH_SIZE=32
LEARNING_RATE="1e-5"
TRAIN_EPOCHS=50
PATIENCE=5

ADAPTER_DIM=128
REPRESENTATION_NORM="batch"
REDUCTION="sum"

LOG_DIR="${LOG_DIR:-./logs/lace_overnight/${MODEL_NAME}}"
if (( ! DRY_RUN )); then mkdir -p "${LOG_DIR}"; fi


# ============================================================
# DATASETS / MODES
# ============================================================

DATASETS=(
    "c_traffic"
    "k_traffic"
    "ETTm1_HUFL"
    "road_sensor_62"
    "road_sensor_103"
    "road_sensor_240"
)

MODES=(
    "none"
    "structural"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================
#
# Each dataset sets:
#   FACTORS
#   TEST_TIMESTAMP
#   ROOT_PATH
#   DATA_PATH
#   TARGET
#   DATA_NAME
#   DATA_RESOLUTION
#   MODEL_ID
#   SEQ_LEN
#   LABEL_LEN
#   PRED_LEN
#
# ============================================================

configure_dataset() {

    local dataset="$1"

    case "${dataset}" in

        # ----------------------------------------------------
        # Network: 10m -> 1h -> 6h -> 1d
        # ----------------------------------------------------
        c_traffic)

            FACTORS=(1 6 36 144)

            TEST_TIMESTAMP="2024-04-07 00:00:00+0000"

            ROOT_PATH="./data/Network/"
            DATA_PATH="c10m_ideal.csv"
            TARGET="total_traffic_kbps"

            DATA_NAME="c_traffic"

            DATA_RESOLUTION="10m"

            MODEL_ID="c_traffic_144"

            SEQ_LEN=1024
            LABEL_LEN=512
            PRED_LEN=144
            ;;


        # ----------------------------------------------------
        # Network: 5m -> 15m -> 1h -> 1d
        # ----------------------------------------------------
        k_traffic)

            FACTORS=(1 3 12 288)

            # Fill these from the exact K-traffic prepared file.
            TEST_TIMESTAMP="2025-07-01 00:00:00+0000"

            ROOT_PATH="./data/Network/"
            DATA_PATH="k5m_ideal.csv"
            TARGET="total_traffic_kbps"

            DATA_NAME="k_traffic"
            DATA_RESOLUTION="5m"

            MODEL_ID="k_traffic_288"

            # One physical week at 5-minute resolution.
            SEQ_LEN=2016
            LABEL_LEN=1008
            PRED_LEN=288
            ;;


        # ----------------------------------------------------
        # Electricity: 15m -> 1h -> 6h -> 1d
        # ----------------------------------------------------
        ETTm1_HUFL)

            FACTORS=(1 4 24 96)

            # Fixed test boundary for the ETTm1 experiments.
            TEST_TIMESTAMP="2018-01-31 00:00:00"

            # Change ROOT_PATH only if your prepared ETT files
            # live elsewhere.
            ROOT_PATH="./data/Electricity/"
            DATA_PATH="ETTm1_HUFL_15m.csv"
            TARGET="value"

            DATA_NAME="ETTm1_HUFL"
            DATA_RESOLUTION="15m"

            MODEL_ID="ETTm1_HUFL_96"

            # One physical week at 15-minute resolution.
            SEQ_LEN=672
            LABEL_LEN=336
            PRED_LEN=96
            ;;


        # ----------------------------------------------------
        # Road: 5m -> 15m -> 1h -> 1d
        # ----------------------------------------------------
        road_sensor_62|road_sensor_103|road_sensor_240)

            FACTORS=(1 3 12 288)

            # Fixed test boundary for the road-traffic experiments.
            TEST_TIMESTAMP="2018-02-17 00:00:00"

            ROOT_PATH="./data/Road/"
            DATA_PATH="${dataset}_5m.csv"
            TARGET="value"

            DATA_NAME="${dataset}"
            DATA_RESOLUTION="5m"

            MODEL_ID="${dataset}_288"

            # One physical week at 5-minute resolution.
            SEQ_LEN=2016
            LABEL_LEN=1008
            PRED_LEN=288
            ;;


        *)
            echo "Unknown dataset: ${dataset}"
            return 1
            ;;
    esac
}


# ============================================================
# MODE CONFIGURATION
# ============================================================

configure_mode() {

    local mode="$1"

    case "${mode}" in

        none)
            LAMBDA="0.0"
            RUN_LABEL="A0"
            ;;

        structural)
            LAMBDA="1.0"
            RUN_LABEL="A1"
            ;;

        *)
            echo "Unknown mode: ${mode}"
            return 1
            ;;
    esac
}


# ============================================================
# CONFIG VALIDATION
# ============================================================

validate_config() {

    local dataset="$1"

    local fields=(
        "${TEST_TIMESTAMP}"
        "${ROOT_PATH}"
        "${DATA_PATH}"
        "${TARGET}"
        "${DATA_NAME}"
        "${DATA_RESOLUTION}"
        "${MODEL_ID}"
    )

    for value in "${fields[@]}"; do
        if [[ "${value}" == "__SET_ME__" ]]; then
            echo
            echo "ERROR: ${dataset} still contains __SET_ME__."
            echo "Edit its configuration before launching the overnight queue."
            echo
            return 1
        fi
    done

    if [[ ! -f "${ROOT_PATH}${DATA_PATH}" ]]; then
        echo
        echo "ERROR: dataset file does not exist:"
        echo "  ${ROOT_PATH}${DATA_PATH}"
        echo
        return 1
    fi

    return 0
}


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

run_one() {

    local dataset="$1"
    local mode="$2"

    configure_dataset "${dataset}" || return 1
    configure_mode "${mode}" || return 1

    if (( ! DRY_RUN )); then validate_config "${dataset}" || return 1; fi

    local description="${RUN_LABEL}_${dataset}_${MODEL_NAME}"
    local factor_tag
    factor_tag=$(IFS=-; echo "${FACTORS[*]}")

    local log_file="${LOG_DIR}/${description}_${mode}.log"

    echo
    echo "============================================================"
    echo "STARTING"
    echo "  dataset      : ${dataset}"
    echo "  mode         : ${mode}"
    echo "  lambda       : ${LAMBDA}"
    echo "  factors      : ${FACTORS[*]}"
    echo "  timestamp    : ${TEST_TIMESTAMP}"
    echo "  root/file    : ${ROOT_PATH}${DATA_PATH}"
    echo "  target       : ${TARGET}"
    echo "  data name    : ${DATA_NAME}"
    echo "  resolution   : ${DATA_RESOLUTION}"
    echo "  seq/label    : ${SEQ_LEN}/${LABEL_LEN}"
    echo "  pred_len     : ${PRED_LEN}"
    echo "  description  : ${description}"
    echo "  log          : ${log_file}"
    echo "============================================================"

    # Do not use `set -e`: if one experiment fails, record it
    # and continue to the next overnight job.
    command=( "${PYTHON_BIN:-python}" -u run.py \
        --task_name multiscale_lace_forecast \
        --is_training 1 \
        --inverse \
        --data fixed_split \
        --lace_factors "${FACTORS[@]}" \
        --lace_adapter_dim "${ADAPTER_DIM}" \
        --lace_representation_norm "${REPRESENTATION_NORM}" \
        --lace_reduction "${REDUCTION}" \
        --lace_coherence_mode "${mode}" \
        --lace_lambda "${LAMBDA}" \
        --data_name "${DATA_NAME}" \
        --data_resolution "${DATA_RESOLUTION}" \
        --test_split_timestamp "${TEST_TIMESTAMP}" \
        --root_path "${ROOT_PATH}" \
        --data_path "${DATA_PATH}" \
        --target "${TARGET}" \
        --model_id "${MODEL_ID}" \
        --model "${MODEL_NAME}" \
        --use_gpu \
        --features S \
        --patience "${PATIENCE}" \
        --seq_len "${SEQ_LEN}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${PRED_LEN}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --train_epochs "${TRAIN_EPOCHS}" \
        --des "${description}" \
        --itr 1
    )
    if (( DRY_RUN )); then
        printf 'COMMAND: '
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" \
        2>&1 | tee "${log_file}"

    local train_status=${PIPESTATUS[0]}

    if [[ ${train_status} -ne 0 ]]; then
        echo
        echo "FAILED: ${dataset} / ${mode}"
        echo "See: ${log_file}"
        return ${train_status}
    fi

    echo
    echo "TRAINING FINISHED: ${dataset} / ${mode}"

    # --------------------------------------------------------
    # Automatically evaluate a successful run.
    # Evaluation failure does NOT erase successful training.
    # --------------------------------------------------------

    local eval_log="${LOG_DIR}/${description}_${mode}_evaluation.log"

    echo "Running evaluator..."

    python -m evaluator.evaluate_lace \
        --dataset "${DATA_NAME}" \
        --mode "${mode}" \
        --model_dir "${MODEL_NAME}" \
        --plot \
        2>&1 | tee "${eval_log}"

    local eval_status=${PIPESTATUS[0]}

    if [[ ${eval_status} -ne 0 ]]; then
        echo
        echo "WARNING: training succeeded but evaluation failed:"
        echo "  ${dataset} / ${mode}"
        echo "See: ${eval_log}"
        return "${eval_status}"
    else
        echo "EVALUATION FINISHED: ${dataset} / ${mode}"
    fi

    return 0
}


# ============================================================
# PRE-FLIGHT CHECK
# ============================================================
#
# Validate EVERY dataset before starting anything.
# This prevents discovering at 3 a.m. that job #5 still had a
# missing timestamp or filename.
# ============================================================

echo
echo "============================================================"
echo "LACE OVERNIGHT PRE-FLIGHT CHECK"
echo "============================================================"

preflight_ok=1

for dataset in "${DATASETS[@]}"; do

    if [[ -n "${ONLY_DATASET}" && "${dataset}" != "${ONLY_DATASET}" ]]; then
        continue
    fi

    configure_dataset "${dataset}" || {
        preflight_ok=0
        continue
    }

    echo
    echo "[${dataset}]"
    echo "  factors      = ${FACTORS[*]}"
    echo "  timestamp    = ${TEST_TIMESTAMP}"
    echo "  file         = ${ROOT_PATH}${DATA_PATH}"
    echo "  target       = ${TARGET}"
    echo "  data_name    = ${DATA_NAME}"
    echo "  resolution   = ${DATA_RESOLUTION}"
    echo "  seq_len      = ${SEQ_LEN}"
    echo "  pred_len     = ${PRED_LEN}"

    if (( ! DRY_RUN )) && ! validate_config "${dataset}"; then
        preflight_ok=0
    fi
done

if [[ ${preflight_ok} -ne 1 ]]; then
    echo
    echo "============================================================"
    echo "PRE-FLIGHT FAILED"
    echo "Fix the configuration above. No experiments were started."
    echo "============================================================"
    exit 1
fi


# ============================================================
# OVERNIGHT QUEUE
# ============================================================

echo
echo "============================================================"
echo "PRE-FLIGHT PASSED"
echo "Starting overnight queue."
echo "============================================================"

successes=()
failures=()

for dataset in "${DATASETS[@]}"; do

    if [[ -n "${ONLY_DATASET}" && "${dataset}" != "${ONLY_DATASET}" ]]; then
        continue
    fi

    for mode in "${MODES[@]}"; do

        if [[ -n "${ONLY_MODE}" && "${mode}" != "${ONLY_MODE}" ]]; then
            continue
        fi

        if run_one "${dataset}" "${mode}"; then
            successes+=("${dataset}:${mode}")
        else
            failures+=("${dataset}:${mode}")
        fi

    done
done


# ============================================================
# FINAL SUMMARY
# ============================================================

echo
echo "============================================================"
echo "OVERNIGHT QUEUE COMPLETE"
echo "============================================================"

echo
echo "Successful runs (${#successes[@]}):"
for item in "${successes[@]}"; do
    echo "  ${item}"
done

echo
echo "Failed runs (${#failures[@]}):"
for item in "${failures[@]}"; do
    echo "  ${item}"
done

echo
echo "Logs:"
echo "  ${LOG_DIR}/"
echo

if (( ${#failures[@]} )); then exit 1; fi
