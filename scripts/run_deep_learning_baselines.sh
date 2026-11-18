#!/usr/bin/env bash

# Resolve paths relative to this checkout, even when invoked elsewhere.
LACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${LACE_ROOT}" || exit 1
export PYTHONPATH="${LACE_ROOT}/src:${LACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Supervised baselines at all resolutions
#
# Purposes:
#   1. Independent per-resolution forecasting baseline
#   2. Bottom-Up: use finest-resolution forecast + aggregation
#   3. WLS-Struct: reconcile independently generated forecasts
# ============================================================

MODEL="${MODEL:-LSTM}"

configs=(

    # ========================================================
    # Network C
    # 10m -> 1h -> 6h -> 1d
    # ========================================================

    "c10m_ideal,c_traffic,10m,./data/Network/,total_traffic_kbps,2024-04-07 00:00:00+0000,1024,512,144"
    "c1h_ideal,c_traffic,1h,./data/Network/,total_traffic_kbps,2024-04-07 00:00:00+0000,168,24,24"
    "c6h_ideal,c_traffic,6h,./data/Network/,total_traffic_kbps,2024-04-07 00:00:00+0000,28,14,4"
    "c1d_ideal,c_traffic,1d,./data/Network/,total_traffic_kbps,2024-04-07 00:00:00+0000,7,1,1"


    # ========================================================
    # Network K
    # 5m -> 15m -> 1h -> 1d
    # ========================================================

    "k5m_ideal,k_traffic,5m,./data/Network/,total_traffic_kbps,2025-07-01 00:00:00+0000,2016,1008,288"
    "k15m_ideal,k_traffic,15m,./data/Network/,total_traffic_kbps,2025-07-01 00:00:00+0000,672,96,96"
    "k1h_ideal,k_traffic,1h,./data/Network/,total_traffic_kbps,2025-07-01 00:00:00+0000,168,24,24"
    "k1d_ideal,k_traffic,1d,./data/Network/,total_traffic_kbps,2025-07-01 00:00:00+0000,7,1,1"


    # ========================================================
    # Electricity
    # 15m -> 1h -> 6h -> 1d
    # Target column: "value"
    # ========================================================

    "ETTm1_HUFL_15m,ETTm1_HUFL,15m,./data/Electricity/,value,2018-01-31 00:00:00,672,336,96"
    "ETTm1_HUFL_1h,ETTm1_HUFL,1h,./data/Electricity/,value,2018-01-31 00:00:00,168,12,24"
    "ETTm1_HUFL_6h,ETTm1_HUFL,6h,./data/Electricity/,value,2018-01-31 00:00:00,28,14,4"
    "ETTm1_HUFL_1d,ETTm1_HUFL,1d,./data/Electricity/,value,2018-01-31 00:00:00,7,3,1"


    # ========================================================
    # Road sensor 62
    # 5m -> 15m -> 1h -> 1d
    # ========================================================

    "road_sensor_62_5m,road_sensor_62,5m,./data/Road/,value,2018-02-17 00:00:00,2016,1008,288"
    "road_sensor_62_15m,road_sensor_62,15m,./data/Road/,value,2018-02-17 00:00:00,672,96,96"
    "road_sensor_62_1h,road_sensor_62,1h,./data/Road/,value,2018-02-17 00:00:00,168,24,24"
    "road_sensor_62_1d,road_sensor_62,1d,./data/Road/,value,2018-02-17 00:00:00,7,1,1"


    # ========================================================
    # Road sensor 103
    # ========================================================

    "road_sensor_103_5m,road_sensor_103,5m,./data/Road/,value,2018-02-17 00:00:00,2016,1008,288"
    "road_sensor_103_15m,road_sensor_103,15m,./data/Road/,value,2018-02-17 00:00:00,672,96,96"
    "road_sensor_103_1h,road_sensor_103,1h,./data/Road/,value,2018-02-17 00:00:00,168,24,24"
    "road_sensor_103_1d,road_sensor_103,1d,./data/Road/,value,2018-02-17 00:00:00,7,1,1"


    # ========================================================
    # Road sensor 240
    # ========================================================

    "road_sensor_240_5m,road_sensor_240,5m,./data/Road/,value,2018-02-17 00:00:00,2016,1008,288"
    "road_sensor_240_15m,road_sensor_240,15m,./data/Road/,value,2018-02-17 00:00:00,672,96,96"
    "road_sensor_240_1h,road_sensor_240,1h,./data/Road/,value,2018-02-17 00:00:00,168,24,24"
    "road_sensor_240_1d,road_sensor_240,1d,./data/Road/,value,2018-02-17 00:00:00,7,1,1"
)


for config in "${configs[@]}"; do

    IFS=',' read -r \
        dataset \
        data_name \
        data_resolution \
        root_path \
        target \
        test_timestamp \
        seq_len \
        label_len \
        pred_len <<< "${config}"


    model_id="${data_name}_${data_resolution}"


    echo
    echo "============================================================"
    echo "${MODEL}"
    echo "Dataset    : ${data_name}"
    echo "File       : ${dataset}.csv"
    echo "Resolution : ${data_resolution}"
    echo "Context    : ${seq_len}"
    echo "Label      : ${label_len}"
    echo "Horizon    : ${pred_len}"
    echo "============================================================"


    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --inverse \
        --root_path "${root_path}" \
        --data fixed_split \
        --test_split_timestamp "${test_timestamp}" \
        --data_path "${dataset}.csv" \
        --data_name "${data_name}" \
        --data_resolution "${data_resolution}" \
        --target "${target}" \
        --model "${MODEL}" \
        --model_id "${model_id}" \
        --features S \
        --seq_len "${seq_len}" \
        --label_len "${label_len}" \
        --pred_len "${pred_len}" \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --train_epochs 100 \
        --patience 5 \
        --des "${MODEL}_multires_baseline" \
        --itr 1


    status=$?


    if [[ ${status} -ne 0 ]]; then

        echo
        echo "FAILED:"
        echo "  ${data_name} / ${data_resolution}"

    else

        echo
        echo "FINISHED:"
        echo "  ${data_name} / ${data_resolution}"

    fi

done


echo
echo "============================================================"
echo "DLINEAR ALL-RESOLUTION RUNS COMPLETE"
echo "============================================================"
