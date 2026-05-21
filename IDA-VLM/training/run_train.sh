
#!/bin/bash

# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# PREFIX="sft-qwen3b-noexpert"
# PREFIX="sft-qwen7b-noexpert"
# PREFIX="sft-qwen3b-PLIPexpert"
# PREFIX="sft-qwen3b-WYZEv0202reidexpert"
# PREFIX="sft-qwen3b-WYZEv03_23_token"
# PREFIX="sft-qwen7b-WYZEv03_23_token"
# PREFIX="sft-qwen7b-WYZEv04_15_token"
# PREFIX="sft-qwen7b-DINOv2"

# PREFIX="distractor-sft-qwen7b-WYZEv03_23_token"
# PREFIX="crossattennullslot-distractor-sft-qwen7b-WYZEv03_23_token"
# PREFIX="qformer-distractor-sft-qwen7b-WYZEv03_23_token"
# PREFIX="distractor-sft-qwen7b-WYZEv04_15_token"
# PREFIX="distractor-sft-qwen7b-noexpert"

# PREFIX="annotated-distractor-sft-qwen7b-WYZEv04_15_token"
PREFIX="annotated-distractor-sft-qwen7b-WYZEv03_23_token"
# PREFIX="annotated-distractor-sft-qwen7b-noexpert"


# EXPERT_FEATURE="wyzev0202reid"
EXPERT_FEATURE="wyzev0323token"
# EXPERT_FEATURE="wyzev0415token"
# EXPERT_FEATURE="DINOv2"
# EXPERT_FEATURE="PLIP"
# EXPERT_FEATURE="None"


# TRAIN_FILE="/home/tian.liu/IDA-VLM/prepare_dataset/02_household-based/train_data.json"
# TRAIN_FILE="/home/tian.liu/IDA-VLM/prepare_dataset/04_varying_gallery_length_distractors/train_data.json"
# TRAIN_FILE="/home/tian.liu/IDA-VLM/prepare_dataset/05_annotated_train/train_data_20distractor.json"
TRAIN_FILE="/home/tian.liu/IDA-VLM/prepare_dataset/06_annotated_abcd/train_data.jsonl"


# Input mode — how the expert feature is fused with Qwen ViT tokens.
# Ignored when EXPERT_FEATURE="None" (always image_only in that case).
# INPUT_MODE="expert_qformer"
# INPUT_MODE="expert_cross_attn"
INPUT_MODE="expert_and_image_attn"
# INPUT_MODE="expert_and_image_add"
# INPUT_MODE="expert_only"


# Trainable modules — which parts of the model get `requires_grad=True`.
# - `merger`, `expert_projector`: shared defaults for every expert run.
# - `expert_fuser`: the learnable fuser (ExpertCrossAttention or ExpertQFormer).
#   Must be present for expert_cross_attn / expert_qformer or the fuser stays
#   frozen at its init and the run does nothing new. Harmless for legacy
#   modes (expert_and_image_attn / add / only / concat), since they don't
#   instantiate a fuser — the substring match in setup_trainable_parameters
#   finds no matching param names.
TRAINING_PARAMETERS="merger expert_projector expert_fuser"


# Number of training epochs. A connector checkpoint is saved at the end of every
# epoch under <run_name>/ckpts/connector_epoch_<N>.pt, plus the final
# <run_name>/connector.pt mirrors the last epoch.
NUM_TRAIN_EPOCHS=3


# Optional warm-start: path to a previously-saved connector.pt whose expert_projector
# weights will be loaded before training starts. Only 'expert_projector' keys are loaded
# so a freshly-initialised expert_fuser (cross-attn or Q-Former) keeps its fresh init.
# Leave empty to skip.

# WARMUP_CONNECTOR_PATH="/home/tian.liu/IDA-VLM/training/runs/20260417_214931_distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt"
WARMUP_CONNECTOR_PATH=""

WARMUP_ARG=""
if [ -n "${WARMUP_CONNECTOR_PATH}" ]; then
    WARMUP_ARG="--warmup_connector_path ${WARMUP_CONNECTOR_PATH}"
fi


if [ "${EXPERT_FEATURE}" == "None" ]; then
    ## Training without any expert feature — pure VLM fine-tune.
    ## No fuser exists, so `expert_fuser` in TRAINING_PARAMETERS is a no-op.
    python train.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_file ${TRAIN_FILE} \
        --object_type person \
        --feature_mode vanilla \
        --input_mode image_only \
        --training_parameters ${TRAINING_PARAMETERS} \
        --num_train_epochs ${NUM_TRAIN_EPOCHS} \
        --learning_rate 2e-4 \
        --captions False \
        --prefix ${PREFIX} \
        ${WARMUP_ARG}
else
    ## Training with an expert feature. INPUT_MODE (set at the top of this file)
    ## decides which fusion variant is installed on model.visual.expert_fuser.
    ##   - expert_cross_attn: per-patch cross-attn over experts + null slot.
    ##   - expert_qformer   : BLIP-2-style two-stage learnable-query fuser.
    ##   - expert_and_image_attn / add / only / concat: legacy, no fuser module.
    python train.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_file ${TRAIN_FILE} \
        --object_type person \
        --feature_mode expert \
        --expert_feature ${EXPERT_FEATURE} \
        --input_mode ${INPUT_MODE} \
        --training_parameters ${TRAINING_PARAMETERS} \
        --num_train_epochs ${NUM_TRAIN_EPOCHS} \
        --learning_rate 2e-4 \
        --captions False \
        --prefix ${PREFIX} \
        ${WARMUP_ARG}
fi