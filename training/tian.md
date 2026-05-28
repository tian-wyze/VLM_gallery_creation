
1. Inference using saved checkpoints

```bash
cd /home/tian.liu/VLMID
python run.py \
    --connector_path /home/tian.liu/liang_data/VLMID_ckpt/20251029071022-qwen2-5-3b-person-ReID-k=5-f=0.5-expert-0.0002_expert_and_image_attn_batch_size_4_captions_True_DINO/connector.pt \
    --test_size 300 \
    --gallery_size 5 \
    --filter 0.5 \
    --object_type person \
    --captions True


# python run.py \
#     --connector_path /home/tian.liu/liang_data/VLMID_ckpt/20251028071005-qwen2-5-3b-person-ReID-k=5-f=0.5-expert-0.0002_expert_and_image_attn_batch_size_4_captions_False_DINO/connector.pt \
#     --test_size 300 \
#     --gallery_size 5 \
#     --filter 0.5 \
#     --object_type person \
#     --captions False
```

2. Training

```bash

# stage-1 training with sentence-format targets (e.g. "The person in the query image matches the person in gallery image X.")
python train.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --train_file /home/tian.liu/IDA-VLM/prepare_dataset/household-based/train_data.json \
    --learning_rate 2e-4 \
    --gallery_size 5 \
    --filter 0.5 \
    --feature_mode expert \
    --input_mode expert_and_image_attn \
    --object_type person \
    --captions False


python train.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --train_file /home/tian.liu/IDA-VLM/prepare_dataset/household-based/train_data.json \
    --learning_rate 2e-4 \
    --gallery_size 5 \
    --filter 0.5 \
    --feature_mode expert \
    --input_mode expert_and_image_attn \
    --object_type person \
    --captions False

```

```bash
# pip install wandb
# wandb server start --port 8081


# use tensorboard
pip install tensorboard
cd /home/tian.liu/IDA-VLM/training

tensorboard --logdir runs/ --port 6006
# then open http://localhost:6006

```