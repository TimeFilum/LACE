#!/bin/bash

# Resolve paths relative to this checkout, even when invoked elsewhere.
LACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${LACE_ROOT}" || exit 1
export PYTHONPATH="${LACE_ROOT}/src:${LACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES=0

models=( "Moirai1" "Moirai2" "TimesFM2" "Chronos2" )

configs=(
    "ETTm1_HUFL_15m,ETTm1_HUFL,15m,672,336,96"
    "ETTm1_HUFL_1h,ETTm1_HUFL,1h,168,12,24"
    "ETTm1_HUFL_6h,ETTm1_HUFL,6h,28,14,4"
    "ETTm1_HUFL_1d,ETTm1_HUFL,1d,28,3,1"
)

for config in "${configs[@]}"; do

    IFS=',' read -r dataset data_name data_resolution seq_len label_len pred_len <<< "$config"

    for model in "${models[@]}"; do

        model_id="${data_name}_${data_resolution}"

        echo "Running $model on $dataset"
        echo "Context=$seq_len | Horizon=$pred_len"

        python -u run.py \
            --task_name zero_shot_forecast \
            --is_training 0 \
            --inverse \
            --root_path ./data/Electricity/ \
            --data fixed_split \
            --test_split_timestamp '2018-01-31 00:00:00' \
            --data_path "${dataset}.csv" \
            --data_name "${data_name}" \
            --data_resolution "${data_resolution}" \
            --target value \
            --model "${model}" \
            --model_id "${model_id}" \
            --features S \
            --seq_len "${seq_len}" \
            --label_len "${label_len}" \
            --pred_len "${pred_len}" \
            --patch_len 32 \
            --factor 3 \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --batch_size 16 \
            --learning_rate 0.001 \
            --train_epochs 100 \
            --des "phase0" \
            --itr 1

        echo "Finished $model on $dataset"
        echo "--------------------------------------"

    done
done
