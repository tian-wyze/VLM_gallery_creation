#!/bin/bash

MODELS=(
    # "gemini-2.5-flash-lite"
    # "gemini-2.5-flash"
    "gemini-2.5-pro"
    # "gemini-3.1-flash-lite-preview"
    # "gemini-3-flash-preview"
)

# Benchmarks live under prepare_dataset/06_annotated_abcd/. Two flavors:
#   benchmarks/                  realistic (gallery = household members only)
#   benchmarks_hardnegatives/    stress test (galleries padded with hard negs)
# Edit the line below to switch flavors. Hardcoded (not env-driven) on
# purpose — a stale TEST_FOLDER export in the user's shell otherwise
# silently overrides this script's default.
TEST_FOLDER="/home/xin.liang/code/VLM_gallery_creation/IDA-VLM/prepare_dataset/06_annotated_abcd/benchmarks"
# TEST_FOLDER="/home/xin.liang/code/VLM_gallery_creation/IDA-VLM/prepare_dataset/06_annotated_abcd/benchmarks_hardnegatives"

# .jsonl files carry the lettered-options format (prepared by
# prepare_jsonl.py). Eval_gemini.py auto-detects .jsonl vs .json.
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
RESULTS_FILE="results_gemini_${TEST_FOLDER_TAG}.csv"

for TEST_FILE_NAME in "${TEST_FILES[@]}"; do
    # Strip .jsonl first (longest match), then .json — supports both shapes.
    SCENARIO="${TEST_FILE_NAME%.jsonl}"
    SCENARIO="${SCENARIO%.json}"
    TEST_FILE="${TEST_FOLDER}/${TEST_FILE_NAME}"

    echo "Testing scenario=$SCENARIO"
    for MODEL in "${MODELS[@]}"; do
        LAST_LINE=$(python eval_gemini.py --test_file "$TEST_FILE" --model_name "$MODEL" | tail -1)
        echo "$MODEL,$SCENARIO,$LAST_LINE" | tee -a "$RESULTS_FILE"
    done
done
