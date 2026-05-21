#!/bin/bash

MODELS=(
    # "Qwen2.5-VL-3B-Instruct"
    "Qwen2.5-VL-7B-Instruct"
    # "Qwen3-VL-8B-Instruct"
)

# Benchmarks live under prepare_dataset/06_annotated_abcd/. Two flavors:
#   benchmarks/                  realistic (gallery = household members only)
#   benchmarks_hardnegatives/    stress test (galleries padded with hard negs)
# Edit the line below to switch flavors. Hardcoded (not env-driven) on
# purpose — a stale TEST_FOLDER export in the user's shell otherwise
# silently overrides this script's default.
TEST_FOLDER="/home/tian.liu/IDA-VLM/prepare_dataset/06_annotated_abcd/benchmarks"
# TEST_FOLDER="/home/tian.liu/IDA-VLM/prepare_dataset/06_annotated_abcd/benchmarks_hardnegatives"

# .jsonl files carry the lettered-options format (prepared by
# prepare_jsonl.py). eval_qwen.py auto-detects .jsonl vs .json.
TEST_FILES=(
    "cropped_sameclothes_singleton_samecamera.jsonl"
    "cropped_sameclothes_singleton_crosscamera.jsonl"
    "cropped_sameclothes_family_samecamera.jsonl"
    "cropped_sameclothes_family_crosscamera.jsonl"
    "cropped_crossclothes_singleton_samecamera.jsonl"
    "cropped_crossclothes_singleton_crosscamera.jsonl"
    "cropped_crossclothes_family_samecamera.jsonl"
    "cropped_crossclothes_family_crosscamera.jsonl"
    "distractor_cropped_singleton.jsonl"
    "distractor_cropped_family.jsonl"
)

# Tag the results file with the benchmark folder so realistic vs.
# hard-negatives runs don't overwrite each other.
TEST_FOLDER_TAG="$(basename "$TEST_FOLDER")"
RESULTS_FILE="results_qwen_${TEST_FOLDER_TAG}.csv"

for TEST_FILE_NAME in "${TEST_FILES[@]}"; do
    # Strip .jsonl first (longest match), then .json — supports both shapes.
    SCENARIO="${TEST_FILE_NAME%.jsonl}"
    SCENARIO="${SCENARIO%.json}"
    TEST_FILE="${TEST_FOLDER}/${TEST_FILE_NAME}"

    echo "Testing scenario=$SCENARIO"
    for MODEL in "${MODELS[@]}"; do
        LAST_LINE=$(python eval_qwen.py --test_file "$TEST_FILE" --model_name "$MODEL" | tee /dev/tty | tail -1)
        echo "$MODEL,$SCENARIO,$LAST_LINE" | tee -a "$RESULTS_FILE"
    done
done
