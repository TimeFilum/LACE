#!/bin/bash

# Resolve paths relative to this checkout, even when invoked elsewhere.
LACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${LACE_ROOT}" || exit 1
export PYTHONPATH="${LACE_ROOT}/src:${LACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES=0

models=( "Moirai1" "Moirai2" "TimesFM2" "Chronos2" )

configs=(
    "k5m_ideal,k_traffic,5m,2016,288,288"
    "k15m_ideal,k_traffic,15m,672,96,96"
    "k1h_ideal,k_traffic,1h,168,24,24"
    "k1d_ideal,k_traffic,1d,28,1,1"
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
            --root_path ./data/Network/ \
            --data fixed_split \
            --test_split_timestamp '2025-07-01 00:00:00+0000' \
            --data_path "${dataset}.csv" \
            --data_name "${data_name}" \
            --data_resolution "${data_resolution}" \
            --target 'total_traffic_kbps' \
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
            --batch_size 32 \
            --learning_rate 0.001 \
            --train_epochs 100 \
            --des "phase0" \
            --itr 1

        echo "Finished $model on $dataset"
        echo "--------------------------------------"

    done
done



models=( "Moirai1" "Moirai2" "TimesFM2" "Chronos2" )

configs=(
    "c10m_ideal,c_traffic,10m,1008,144,144"
    "c1h_ideal,c_traffic,1h,168,24,24"
    "c6h_ideal,c_traffic,6h,28,14,4"
    "c1d_ideal,c_traffic,1d,28,1,1"
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
            --root_path ./data/Network/ \
            --data fixed_split \
            --test_split_timestamp '2024-04-07 00:00:00+0000' \
            --data_path "${dataset}.csv" \
            --data_name "${data_name}" \
            --data_resolution "${data_resolution}" \
            --target 'total_traffic_kbps' \
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
            --batch_size 32 \
            --learning_rate 0.001 \
            --train_epochs 100 \
            --des "phase0" \
            --itr 1

        echo "Finished $model on $dataset"
        echo "--------------------------------------"

    done
done
