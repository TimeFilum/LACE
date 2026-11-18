#!/usr/bin/env bash

# Resolve paths relative to this checkout, even when invoked elsewhere.
LACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${LACE_ROOT}" || exit 1
export PYTHONPATH="${LACE_ROOT}/src:${LACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ============================================================
# LACE MULTI-BACKBONE OVERNIGHT QUEUE
# ============================================================
#
# Runs the LACE experiment protocol for:
#
#   1. Moirai-2
#   2. Chronos-2
#   3. TimesFM-2
#
# Dataset configuration, hierarchy, optimizer settings,
# A0/A1 definitions, and evaluation remain in
# run_lace_overnight.sh.
# ============================================================

set -u

MODELS=(
    "LACE_moirai2"
    "LACE_chronos2"
    "LACE_timesfm2"
)

BASE_LOG_DIR="./logs/lace_backbones"

mkdir -p "${BASE_LOG_DIR}"

successes=()
failures=()

echo
echo "============================================================"
echo "LACE MULTI-BACKBONE OVERNIGHT QUEUE"
echo "============================================================"

for model in "${MODELS[@]}"; do

    echo
    echo "============================================================"
    echo "STARTING BACKBONE: ${model}"
    echo "============================================================"

    MODEL_NAME="${model}" \
    LOG_DIR="${BASE_LOG_DIR}/${model}" \
    bash scripts/run_lace_overnight.sh "$@"

    status=$?

    if [[ ${status} -eq 0 ]]; then
        successes+=("${model}")
    else
        failures+=("${model}")
    fi

    echo
    echo "============================================================"
    echo "FINISHED BACKBONE: ${model}"
    echo "============================================================"

done


echo
echo "============================================================"
echo "MULTI-BACKBONE QUEUE COMPLETE"
echo "============================================================"

echo
echo "Completed backbone queues (${#successes[@]}):"
for model in "${successes[@]}"; do
    echo "  ${model}"
done

echo
echo "Failed backbone queues (${#failures[@]}):"
for model in "${failures[@]}"; do
    echo "  ${model}"
done

echo
echo "Logs:"
echo "  ${BASE_LOG_DIR}/"
echo
