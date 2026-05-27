#!/bin/bash

MODELS=(
    "InternVL3_8B_DAfinetuned"
    # "InternVL3_5-8B"
    # "InternVL3-8B"
)
CLOTHES=(
    "sameclothes"
    # "crossclothes"
)
HOUSEHOLDS=(
    "singleton"
    "family"
)
CAMERAS=(
    "samecamera"
    "crosscamera"
)

declare -A DATA_FOLDER_MAP
DATA_FOLDER_MAP["crossclothes"]="/home/xin.liang/dataset/wyze_person_v2/cross_clothes/"
DATA_FOLDER_MAP["sameclothes"]="/home/xin.liang/dataset/wyze_person_v2/same_clothes/"


RESULTS_FILE="results_internvl.csv"

for MODEL in "${MODELS[@]}"; do
    for CLOTHES_TYPE in "${CLOTHES[@]}"; do
        DATA_FOLDER="${DATA_FOLDER_MAP[$CLOTHES_TYPE]}"
        for HOUSEHOLD in "${HOUSEHOLDS[@]}"; do
            for CAMERA in "${CAMERAS[@]}"; do
                SCENARIO="${HOUSEHOLD}_${CLOTHES_TYPE}_${CAMERA}"
                TEST_FILE="../prepare_dataset/household-based/${SCENARIO}.json"
                echo "Running: model=$MODEL scenario=$SCENARIO"
                LAST_LINE=$(python eval_internvl.py --test_file $TEST_FILE --data_folder $DATA_FOLDER --model_name $MODEL | tee /dev/tty | tail -1)
                echo "$MODEL,$SCENARIO,$LAST_LINE" | tee -a $RESULTS_FILE
            done
        done
    done
done


