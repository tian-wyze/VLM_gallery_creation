#!/bin/bash

MODELS=(
    # Liang's trained Person ReID model
    # "/home/tian.liu/liang_data/VLMID_ckpt/20251028071005-qwen2-5-3b-person-ReID-k=5-f=0.5-expert-0.0002_expert_and_image_attn_batch_size_4_captions_False_DINO/connector.pt"
    # "/home/tian.liu/liang_data/VLMID_ckpt/20251029071022-qwen2-5-3b-person-ReID-k=5-f=0.5-expert-0.0002_expert_and_image_attn_batch_size_4_captions_True_DINO/connector.pt"

    # data leakaged
    # "/home/tian.liu/VLMID/runs/20260401052652-qwen2-5-3b-person-ReID-k=5-f=0.5-expert-0.0002_expert_and_image_attn_batch_size_4_captions_True_DINO/connector.pt"
    # "/home/tian.liu/VLMID/runs/20260401153856-qwen2-5-3b-person-ReID-k=5-f=0.5-expert-0.0002_expert_and_image_attn_batch_size_4_captions_False_DINO/connector.pt"

    # newly trained with PLIP feature, no data leakage
    # "/home/tian.liu/VLMID/runs/20260402060313-qwen2-5-3b-person-ReID-expert-0.0002_expert_and_image_attn_batch_size_4_captions_False/connector.pt"

    # newly trained with wyze v3 feature, no data leakage
    # "/home/tian.liu/VLMID/runs/20260402171159-qwen2-5-3b-person-ReID-expert-0.0002_expert_and_image_attn_batch_size_4_captions_False/connector.pt"

    # trained without expert feature. no data leakage, FT on wyze data
    # "/home/tian.liu/VLMID/runs/20260402224943-qwen2-5-3b-person-ReID-vanilla-0.0002_image_only_batch_size_4_captions_False/connector.pt"

    # Qwen-3b FT on wyze data
    # "/home/tian.liu/IDA-VLM/training/runs/20260407_211451_sft-qwen3b-noexpert_Qwen2.5-VL-3B-Instruct_person_vanilla_image_only_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260408_032711_sft-qwen3b-PLIPexpert_Qwen2.5-VL-3B-Instruct_person_expert_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260408_135528_sft-qwen3b-WYZEv0202reidexpert_Qwen2.5-VL-3B-Instruct_person_expert_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260408_191025_sft-qwen3b-WYZEv03_23_token_Qwen2.5-VL-3B-Instruct_person_expert_wyzev0323token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260410_151521_sft-qwen3b-DINOv2_Qwen2.5-VL-3B-Instruct_person_expert_DINOv2_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"

    # "/home/tian.liu/IDA-VLM/training/runs/20260414_220913_sft-qwen7b-noexpert_Qwen2.5-VL-7B-Instruct_person_vanilla_None_image_only_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260414_034926_sft-qwen7b-DINOv2_Qwen2.5-VL-7B-Instruct_person_expert_DINOv2_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260414_154442_sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260416_002842_sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"

    # distractor varying gallery length
    # "/home/tian.liu/IDA-VLM/training/runs/20260417_214931_distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260420_203233_distractor-sft-qwen7b-noexpert_Qwen2.5-VL-7B-Instruct_person_vanilla_None_image_only_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260422_012930_distractor-sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260423_023310_crossatten-distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_cross_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "runs/20260423_160230_crossattennullslot-distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_cross_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "runs/20260423_204958_qformer-distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_qformer_lr_0.0002_bs_4_captions_False/connector.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260424_005636_crossattennullslot-distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_cross_attn_lr_0.0002_bs_4_captions_False/connector.pt"

    # more annotated data
    # "runs/20260427_212918_annotated-distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "runs/20260428_172402_annotated-distractor-sft-qwen7b-noexpert_Qwen2.5-VL-7B-Instruct_person_vanilla_None_image_only_lr_0.0002_bs_4_captions_False/connector.pt"
    # "runs/20260429_170847_annotated-distractor-sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
    # "runs/20260430_040726_annotated-distractor-sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/ckpts/connector_epoch_2.pt"

    # ABCD, family to singleton augmentation
    # "/home/tian.liu/IDA-VLM/training/runs/20260502_014222_annotated-distractor-sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/ckpts/connector_best_step_38000.pt"
    # "/home/tian.liu/IDA-VLM/training/runs/20260504_003210_annotated-distractor-sft-qwen7b-noexpert_Qwen2.5-VL-7B-Instruct_person_vanilla_None_image_only_lr_0.0002_bs_4_captions_False/ckpts/connector_best_step_18000.pt"
    "/home/tian.liu/IDA-VLM/training/runs/20260504_191143_annotated-distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/ckpts/connector_best_step_34000.pt"
)

# RESULTS_FILE="results_Qwen2.5-VL-3B-PLIP.csv"
# RESULTS_FILE="results_Qwen2.5-VL-3B-PLIP_liang.csv"
# RESULTS_FILE="results/results_Qwen2.5-VL-3B_FTwyze.csv"
# RESULTS_FILE="results/results_Qwen2.5-VL-7B_FTwyze.csv"
# RESULTS_FILE="results/results_distractor_Qwen2.5-VL-7B_FTwyze.csv"
# RESULTS_FILE="results/results_annotated_Qwen2.5-VL-7B_FTwyze.csv"
RESULTS_FILE="results/results_annotated_abcd_Qwen2.5-VL-7B_FTwyze.csv"


# RESULTS_FILE="results_Qwen2.5-VL-3B-vanilla.csv"
# RESULTS_FILE="results_Qwen2.5-VL-7B-vanilla.csv"

BATCH_SIZE=4

# TEST_FOLDER="../prepare_dataset/04_varying_gallery_length_distractors/benchmarks"
# TEST_FOLDER="/home/tian.liu/IDA-VLM/prepare_dataset/05_annotated_train/benchmarks"
TEST_FOLDER="/home/tian.liu/IDA-VLM/prepare_dataset/06_annotated_abcd/benchmarks"

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

# Create results folder if it doesn't exist
mkdir -p results

for MODEL in "${MODELS[@]}"; do
    # Three layouts are accepted:
    #   1. <run_dir>/connector.pt                            (final checkpoint, original layout)
    #   2. <run_dir>/ckpts/connector_epoch_<N>.pt            (per-epoch checkpoint)
    #   3. <run_dir>/ckpts/connector_best_step_<N>.pt        (best-eval-loss checkpoint)
    # We need to (a) point parsing at the run folder (not the ckpts/ folder),
    # and (b) capture an epoch / step tag when present so PREFIX includes it,
    # otherwise different checkpoints would overwrite each other's prediction CSVs.
    CKPT_BASENAME="$(basename "$MODEL")"
    CKPT_TAG=""
    if [[ "$CKPT_BASENAME" =~ ^connector_epoch_([0-9]+)\.pt$ ]]; then
        CKPT_TAG="epoch${BASH_REMATCH[1]}"
        RUN_DIR="$(dirname "$(dirname "$MODEL")")"
    elif [[ "$CKPT_BASENAME" =~ ^connector_best_step_([0-9]+)\.pt$ ]]; then
        CKPT_TAG="best_step${BASH_REMATCH[1]}"
        RUN_DIR="$(dirname "$(dirname "$MODEL")")"
    else
        RUN_DIR="$(dirname "$MODEL")"
    fi

    IFS='_' read -ra PARTS <<< "$(basename "$RUN_DIR")"
    echo ""
    # Format: {date}_{time}_{prefix}_{model_short_name}_{object_type}_{feature_mode}[_{expert_feature}]_{input_mode}_lr_{lr}_bs_{bs}_captions_{captions}
    # Anchor on model_short_name (e.g. Qwen2.5-VL-3B-Instruct): uses hyphens only, starts with "Qwen"
    MODEL_SHORT_IDX=3
    for i in "${!PARTS[@]}"; do
        if [[ "${PARTS[$i]}" == Qwen* ]]; then MODEL_SHORT_IDX=$i; break; fi
    done
    PREFIX=$(IFS='_'; echo "${PARTS[*]:2:$((MODEL_SHORT_IDX-2))}")
    if [[ -n "$CKPT_TAG" ]]; then
        PREFIX="${PREFIX}_${CKPT_TAG}"
    fi
    MODEL_SHORT="${PARTS[$MODEL_SHORT_IDX]}"
    MODEL_ID="Qwen/${MODEL_SHORT}"
    OBJECT_TYPE="${PARTS[$((MODEL_SHORT_IDX+1))]}"
    FEATURE_MODE="${PARTS[$((MODEL_SHORT_IDX+2))]}"

    # Find LR_IDX
    LR_IDX=$((MODEL_SHORT_IDX+3))
    for i in "${!PARTS[@]}"; do
        if [[ "${PARTS[$i]}" == "lr" ]]; then LR_IDX=$i; break; fi
    done

    # Parse input_mode backward from lr by matching against the known set.
    # Input_modes have 2, 3, or 4 tokens; try longest first so e.g.
    # "expert_and_image_attn" (4) wins over any 3-token suffix that happens to end in "attn".
    INPUT_START=""
    for n in 4 3 2; do
        START=$((LR_IDX-n))
        CAND=$(IFS='_'; echo "${PARTS[*]:$START:$n}")
        case "$CAND" in
            image_only|expert_only|expert_qformer|expert_and_image_attn|expert_and_image_concat|expert_and_image_add|expert_cross_attn)
                INPUT_START=$START
                break
                ;;
        esac
    done
    if [[ -z "$INPUT_START" ]]; then
        echo "Could not parse input_mode from $(basename $(dirname $MODEL))"
        # exit 1
    fi
    INPUT_MODE=$(IFS='_'; echo "${PARTS[*]:$INPUT_START:$((LR_IDX-INPUT_START))}")

    # expert_feature spans between feature_mode and input_mode (may be empty)
    EXPERT_FEATURE_START=$((MODEL_SHORT_IDX+3))
    EXPERT_FEATURE_LEN=$((INPUT_START-EXPERT_FEATURE_START))
    if [[ $EXPERT_FEATURE_LEN -gt 0 ]]; then
        EXPERT_FEATURE=$(IFS='_'; echo "${PARTS[*]:$EXPERT_FEATURE_START:$EXPERT_FEATURE_LEN}")
    else
        EXPERT_FEATURE="None"
    fi

    CAPTIONS="${PARTS[-1]}"

    echo "PREFIX=$PREFIX"
    echo "MODEL_ID=$MODEL_ID"
    echo "FEATURE_MODE=$FEATURE_MODE"
    echo "EXPERT_FEATURE=$EXPERT_FEATURE"
    echo "INPUT_MODE=$INPUT_MODE"
    echo "CAPTIONS=$CAPTIONS"
    echo ""
    # exit 0

    for TEST_FILE_NAME in "${TEST_FILES[@]}"; do
        # Strip .jsonl first (longest match), then .json — supports both shapes.
        SCENARIO="${TEST_FILE_NAME%.jsonl}"
        SCENARIO="${SCENARIO%.json}"
        TEST_FILE="${TEST_FOLDER}/${TEST_FILE_NAME}"

        echo "Testing scenario=$SCENARIO"
        LAST_LINE=$(python test.py --connector_path $MODEL --model_id $MODEL_ID --batch_size $BATCH_SIZE \
        --object_type person --feature_mode $FEATURE_MODE --expert_feature $EXPERT_FEATURE --prefix $PREFIX \
        --input_mode $INPUT_MODE --test_file $TEST_FILE --captions $CAPTIONS | tee /dev/tty | tail -1)
        echo "$PREFIX,$MODEL_ID,$FEATURE_MODE,$INPUT_MODE,$CAPTIONS,$SCENARIO,$LAST_LINE" | tee -a $RESULTS_FILE

        # exit program for quick testing
        # exit 0
    done

done


