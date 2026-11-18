#!/bin/bash

# Resolve paths relative to this checkout, even when invoked elsewhere.
LACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${LACE_ROOT}" || exit 1
export PYTHONPATH="${LACE_ROOT}/src:${LACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES=0

models=( "Moirai1" "Moirai2" "TimesFM2" "Chronos2" )

configs=(
    "road_sensor_62_5m,road_sensor_62,5m,2016,288,288"
    "road_sensor_62_15m,road_sensor_62,15m,672,96,96"
    "road_sensor_62_1h,road_sensor_62,1h,168,24,24"
    "road_sensor_62_1d,road_sensor_62,1d,28,1,1"

    "road_sensor_103_5m,road_sensor_103,5m,2016,288,288"
    "road_sensor_103_15m,road_sensor_103,15m,672,96,96"
    "road_sensor_103_1h,road_sensor_103,1h,168,24,24"
    "road_sensor_103_1d,road_sensor_103,1d,28,1,1"

    "road_sensor_240_5m,road_sensor_240,5m,2016,288,288"
    "road_sensor_240_15m,road_sensor_240,15m,672,96,96"
    "road_sensor_240_1h,road_sensor_240,1h,168,24,24"
    "road_sensor_240_1d,road_sensor_240,1d,28,1,1"
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
            --root_path ./data/Road/ \
            --data fixed_split \
            --test_split_timestamp '2018-02-17 00:00:00' \
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
            --batch_size 1 \
            --learning_rate 0.001 \
            --train_epochs 100 \
            --des "phase0" \
            --itr 1

        echo "Finished $model on $dataset"
        echo "--------------------------------------"

    done
done
