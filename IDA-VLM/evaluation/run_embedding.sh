#!/bin/bash

# Force torch to use its bundled cuBLAS (pip nvidia-cublas-cu12) instead of the
# system /usr/local/cuda libcublasLt, which is ABI-mismatched and causes
# "Invalid handle. Cannot load symbol cublasLtCreate".
# NV_LIB_BASE=/usr/local/lib/python3.10/dist-packages/nvidia
# if [[ -d "$NV_LIB_BASE/cublas/lib" ]]; then
#     export LD_LIBRARY_PATH="$NV_LIB_BASE/cublas/lib:$NV_LIB_BASE/cuda_runtime/lib:$NV_LIB_BASE/cudnn/lib:$NV_LIB_BASE/cufft/lib:$NV_LIB_BASE/curand/lib:$NV_LIB_BASE/cusolver/lib:$NV_LIB_BASE/cusparse/lib:$NV_LIB_BASE/nvjitlink/lib:$LD_LIBRARY_PATH"
# fi

MODELS=(
    #"DINOv2"
    "WYZE_embedding"
    #"PLIP"
)
WYZE_VARIANTS=(
    #"50k"
    # "v02_02_reid"
     "v03_23_token"
    #"v04_15_token"
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
# prepare_jsonl.py). eval_embedding.py auto-detects .jsonl vs .json.
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

# Stranger threshold: predictions with max cosine < this become the stranger
# option. Tune per backbone — Wyze v04_15_token tends to peak high on true
# matches (0.4–0.5 work well), DINOv2 is closer to 0.6.
# Hardcoded (not env-driven) on purpose — a stale STRANGER_THRESHOLD export
# in the user's shell otherwise silently overrides this script's value.
STRANGER_THRESHOLD="0.0"
echo "Using stranger threshold: $STRANGER_THRESHOLD"

# Tag the results file with the benchmark folder so realistic vs.
# hard-negatives runs don't overwrite each other.
TEST_FOLDER_TAG="$(basename "$TEST_FOLDER")"
RESULTS_FILE="results_embedding_${TEST_FOLDER_TAG}.csv"

for MODEL in "${MODELS[@]}"; do
    if [[ "$MODEL" == "WYZE_embedding" ]]; then
        VARIANTS=("${WYZE_VARIANTS[@]}")
    else
        VARIANTS=("")
    fi
    for VARIANT in "${VARIANTS[@]}"; do
        VARIANT_FLAG=""
        LABEL="$MODEL"
        if [[ -n "$VARIANT" ]]; then
            VARIANT_FLAG="--wyze_variant $VARIANT"
            LABEL="${MODEL}_${VARIANT}"
        fi

        for TEST_FILE_NAME in "${TEST_FILES[@]}"; do
            # Strip .jsonl first (longest match), then .json — supports both shapes.
            SCENARIO="${TEST_FILE_NAME%.jsonl}"
            SCENARIO="${SCENARIO%.json}"
            TEST_FILE="${TEST_FOLDER}/${TEST_FILE_NAME}"

            echo "Running: model=$LABEL scenario=$SCENARIO threshold=$STRANGER_THRESHOLD"
            LAST_LINE=$(python eval_embedding.py \
                --test_file "$TEST_FILE" \
                --model "$MODEL" $VARIANT_FLAG \
                --stranger_threshold "$STRANGER_THRESHOLD" \
                | tee /dev/tty | tail -1)
            echo "$LABEL,$SCENARIO,$LAST_LINE" | tee -a "$RESULTS_FILE"
        done
    done
done
