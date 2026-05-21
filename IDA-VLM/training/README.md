# Training — VLM Person Re-Identification

This folder holds everything for fine-tuning a Qwen2.5-VL model on the person-ReID task: given one query image and a variable-sized list of N+1 lettered options (N gallery images + a text-only "stranger / not in the gallery" placeholder inserted at a randomized letter slot), pick the matching letter `A`, `B`, `C`, ... If the query person is not in any gallery image, the correct answer is the letter of the stranger placeholder.

## What this folder does

1. Loads the train/test **JSONL** files produced by [`prepare_dataset/06_annotated_abcd/prepare_jsonl.py`](../prepare_dataset/06_annotated_abcd/) (which adds the randomized stranger slot and `answer_letter` to each case). Legacy JSON inputs from earlier folders still work for the digit/`-1` answer format — the loaders auto-detect by file extension.
2. Wraps Qwen2.5-VL with optional **expert-feature integration** (DINOv2 / PLIP / Wyze embeddings) that injects domain-specific embeddings into the vision-language fusion.
3. Fine-tunes via SFT (next-token cross-entropy) with tight **answer-token-only** loss masking — only the single answer-letter token contributes to the loss.
4. Runs evaluation over the 10 test benchmarks and writes accuracy per scenario.

## Folder layout

```
training/
├── train.py               # Training entry (argparse + TrainingConfig + run_training)
├── test.py                # Benchmark entry (per-benchmark accuracy + CSV records)
├── inference.py           # Single-query inference API (IDAVLM.predict(query, gallery))
├── example_inference.py   # Minimal end-to-end example that calls IDAVLM
├── data.py                # Conversation formatter (format_data_reid) + system prompts
├── replace.py             # Monkey-patches Qwen2.5-VL to inject expert features
├── run_train.sh           # Convenience launcher for train.py
├── run_test.sh            # Sweep launcher over all 10 benchmarks × connectors
├── cross_attention.md     # Full design doc for the expert_cross_attn input mode
├── qformer.md             # Full design doc for the expert_qformer input mode
├── experts/               # Expert backbone code (PLIP, Wyze embeddings)
├── utils/
│   ├── data_utils.py      # Loaders: load_wyze_train_dataset, load_wyze_dataset, …
│   ├── model_utils.py     # load_model_and_processor, expert loading, trainable params
│   ├── collate_utils.py   # collate_fn + label masking (answer-token-only)
│   ├── training_utils.py  # run_training, training args, seed setup
│   └── logging_utils.py   # Tee stdout/stderr to terminal + log file
└── runs/                  # Per-run outputs (connector.pt, train.log, test.log, …)
```

## Model architecture

The base model is Qwen2.5-VL (3B or 7B) — a vision-language model where patches from each image are projected and merged into a shared sequence that the LLM attends to, alongside text tokens. Two fine-tuning configurations are supported:

### 1. `image_only` + `vanilla` — pure VLM fine-tuning

No expert features. Standard VLM: each image is patchified, encoded by the vision transformer, merged, and passed to the LLM. Only the **vision–language merger** (and optionally the LLM) is trained. This is the simplest baseline.

### 2. `expert_and_image_attn` + `expert` — expert-feature fusion

An additional pretrained image encoder (the "expert") produces a domain-specific global embedding per image, which is fused into the standard image tokens through attention.

Data flow per image:
```
                  ┌──────────────┐
     raw image ──►│ Qwen2.5 ViT  │──► patch tokens ─┐
                  └──────────────┘                  │
                                                    ├─► merged, attention-fused
                  ┌──────────────┐                  │   image_embeds → LLM
     raw image ──►│   Expert     │──► 1 vec (768D) ─┘
                  │  (DINOv2 /   │       │
                  │   PLIP /     │       ▼
                  │   Wyze)      │   expert_projector
                  └──────────────┘       (trainable)
```

The fusion happens in `My_Qwen2_5_VLModel_forward` in [replace.py](replace.py):

1. Standard image tokens flow through Qwen's vision transformer → `image_embeds`.
2. The same raw images (pre-processed differently) are sent through the expert backbone → one pooled vector per image, then through a trainable `expert_projector` → same hidden dim as `image_embeds`.
3. The expert vector is broadcast to every patch token belonging to its source image → `broadcasted_expert_feature`.
4. An attention mask computes how much each image token should listen to the expert:
   ```
   image_embeds = (1 − α) * image_embeds + α * broadcasted_expert_feature
   ```
   where `α` is produced by `get_expert_attention_mask` (depends on `input_mode`).
5. The fused `image_embeds` are scattered into the LLM's input embeddings at the image-token positions, replacing the original image tokens.

The rest of the forward pass is unchanged — the LLM sees a sequence that now carries both generic VLM features and domain-specific expert signal.

**Trainable parameters** are controlled by `--training_parameters` in `train.py`:
- `merger` — Qwen's vision-language merger (default, essential)
- `expert_projector` — the projection from expert-dim → Qwen-hidden (default)
- `expert_fuser` — the learnable fuser module (ExpertCrossAttention or ExpertQFormer, depending on `input_mode`)
- `expert` — expert backbone itself (optional; usually frozen)
- `llm` — full LLM (optional; very expensive)

Everything else stays frozen. Trained weights are saved to `runs/<run_name>/connector.pt`.

### 3. `expert_cross_attn` + `expert` — per-sample learnable cross-attention

A small **learnable multi-head cross-attention** layer where each image patch token (Q) attends to the expert descriptors of **all** images in the same sample (K/V), plus a learnable **null slot** that serves as an abstain path for stranger queries. More expressive than `expert_and_image_attn`; patches still do the matching.

Recommended starting setup:

```
--input_mode expert_cross_attn --feature_mode expert
--training_parameters merger expert_projector expert_fuser
--warmup_connector_path <path_to_prior_run>/connector.pt   # loads only 'expert_projector'
```

**Full design doc — motivation, Q/K/V roles, what the attention output means, null-slot behavior, scoping, implementation choices — lives in [cross_attention.md](cross_attention.md).**

### 4. `expert_qformer` + `expert` — Q-Former-style two-stage learnable-query fuser

A BLIP-2-inspired two-stage fuser. **K learnable query tokens** cross-attend to the per-sample expert set (stage 1) to produce K "digest" tokens, then image patch tokens cross-attend to those digests (stage 2) and get a residual-added fusion. Unlike cross-attn, matching is done globally once per sample with specialized learnable queries — patches just consume the digests.

Recommended starting setup:

```
--input_mode expert_qformer --feature_mode expert
--qformer_num_queries 8 --qformer_num_heads 8
--training_parameters merger expert_projector expert_fuser
--warmup_connector_path <path_to_prior_run>/connector.pt   # loads only 'expert_projector'
```

**Full design doc — motivation vs. cross-attn, learnable-query roles, two-stage math, how queries specialize under loss (stranger detector, best-match extractor, etc.), implementation choices, and parameter/compute cost — lives in [qformer.md](qformer.md).**

All existing input modes (`image_only`, `expert_and_image_attn`, `expert_and_image_add`, `expert_only`, `expert_and_image_concat`, `expert_cross_attn`) are untouched; the new path is fully gated by `input_mode == 'expert_qformer'`.

### Which monkey-patches are active

[replace.py](replace.py) swaps these Qwen2.5-VL methods on import (so merely `import replace` installs them):

- `Qwen2_5_VLProcessor.__call__` — passes expert info through
- `Qwen2_5_VisionTransformerPretrainedModel.forward` — standard ViT path
- `Qwen2_5_VLModel.forward` — wires the expert fusion
- `Qwen2_5_VLModel.get_image_features`, `get_expert_feature`
- `Qwen2_5_VLForConditionalGeneration.forward`

## Data flow

### Prompt / response format

The system prompt lives in [`../evaluation/prompt.txt`](../evaluation/prompt.txt). It's generic across gallery sizes and explains the lettered-options layout. For each case, `format_data_reid` (specifically `_format_data_reid_letter` for letter-format samples) in [data.py](data.py) builds a 3-turn conversation:

```
system:    <prompt.txt contents>
user:      Query: <image>
           A: <image>                                  # gallery option
           B: (stranger / not in the gallery)         # stranger slot — text only, randomized position per case
           C: <image>                                  # gallery option
           D: <image>                                  # gallery option
           ... (N+1 options total: N images + 1 stranger slot)
assistant: Answer: B.
```

The case dict (from `prepare_jsonl.py`) carries `stranger_letter_pos` (where the stranger slot is inserted, 0..N) and `answer_letter` (the letter the assistant must output). Letter assignment is fixed at prep time and stored in the JSONL — re-running training does not re-randomize.

Legacy fallback: if a sample lacks `answer_letter` (e.g. an old `.json` file from `04_*/`), `format_data_reid` falls back to the original `Gallery 1:`/`Gallery 2:` layout with `Answer: 2` / `Answer: -1` digit responses. Both formats are supported in the same training run.

### Loss

Standard causal LM cross-entropy, but with **tight label masking** ([collate_utils.py](utils/collate_utils.py)):

1. `mask_between_tokens` — everything outside the assistant response (system/user prompts, image tokens, padding) set to `-100`.
2. `restrict_to_answer_tokens` — within the response, only the answer-letter tokens (A-Z, with and without a leading space) and `<|im_end|>` are kept. The `Answer:` prefix and the trailing period tokenize as multi-character pieces and are correctly masked out as boilerplate.

Net effect: gradient is spent on exactly the single answer-letter token plus EOS. No capacity wasted memorizing template phrasing.

A one-time example dump on the first batch prints the full templated text and the supervised-token list so you can visually verify the masking.

## Dependencies

In addition to the usual Python packages (`torch`, `transformers`, `qwen_vl_utils`, `Pillow`, …), the expert backbones require the Wyze instance-embedding library. It is vendored at [`experts/wyze_embedding/`](experts/wyze_embedding/) and must be checked out to the branch that ships the `v03_23_token` / `v04_15_token` / `v02_02_reid` expert models:

```bash
# From IDA-VLM/training/experts/wyze_embedding/
# repo: git@github.com:wyzelabs-inc/WyzeInstanceEmbeddingLib.git
git fetch origin
git checkout dev/neil/petreid
git pull origin dev/neil/petreid
```

If you clone fresh instead:

```bash
cd IDA-VLM/training/experts
git clone -b dev/neil/petreid git@github.com:wyzelabs-inc/WyzeInstanceEmbeddingLib.git wyze_embedding
```

This branch is required for training and inference with any of the `wyzev*` expert features. DINOv2-only and vanilla runs do not use this library at inference time, but `utils/model_utils.py` still imports it at module load.

## How to run

### Training

Edit `run_train.sh` (top of file) to pick:
- `MODEL_NAME_OR_PATH` — `Qwen/Qwen2.5-VL-3B-Instruct` or `Qwen/Qwen2.5-VL-7B-Instruct`
- `PREFIX` — descriptive tag for the run folder
- `EXPERT_FEATURE` — `None` | `DINOv2` | `PLIP` | `wyzev0202reid` | `wyzev0323token` | `wyzev0415token`
- `TRAIN_FILE` — path to `train_data.jsonl` (the lettered-options JSONL produced by `prepare_jsonl.py`); legacy `.json` files are accepted for the digit/`-1` format

Then:

```bash
cd IDA-VLM/training
bash run_train.sh
```

The script dispatches to one of two `train.py` configurations based on `EXPERT_FEATURE`:

- `EXPERT_FEATURE=None` → `--feature_mode vanilla --input_mode image_only`
- otherwise → `--feature_mode expert --input_mode expert_and_image_attn --expert_feature <...>`

To train with the new `expert_cross_attn` fusion instead, change the `--input_mode` line in the second branch of `run_train.sh` to `expert_cross_attn` and add `expert_fuser` to `--training_parameters`.

Optionally, set `WARMUP_CONNECTOR_PATH` near the top of `run_train.sh` to the `connector.pt` of a prior run — only its `expert_projector.*` keys are loaded, so a freshly-initialised `expert_fuser` keeps its zero-init while the projector starts already aligned to Qwen's hidden space.

Run output lands in `runs/<timestamp>_<prefix>_<model>_<object>_<feature_mode>_<expert>_<input>_lr_...`. Every print is tee'd to `runs/<run>/train.log`.

### Testing

Edit `run_test.sh` to set the `MODELS` array (paths to `connector.pt` files) and a `RESULTS_FILE` path. Then:

```bash
cd IDA-VLM/training
bash run_test.sh
```

The script:
1. Parses the run folder name to recover training config (model, feature_mode, expert_feature, input_mode, captions).
2. Sweeps all combinations of `{sameclothes, crossclothes} × {singleton, family} × {samecamera, crosscamera}` benchmarks.
3. For each scenario, runs `test.py` with the connector and appends a one-line result (`prefix,model,feature_mode,input_mode,captions,scenario,accuracy`) to `RESULTS_FILE`.

Output from each test run is tee'd to `<connector_dir>/test.log`. Per-case predictions are written to `<connector_dir>/predictions_<prefix>_<scenario>.csv`.

### Running `train.py` / `test.py` directly

If you want finer control (e.g., alternate datasets, custom flags):

```bash
# Train
python train.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --train_file ../prepare_dataset/06_annotated_abcd/train_data.jsonl \
  --object_type person \
  --feature_mode expert --expert_feature DINOv2 \
  --input_mode expert_and_image_attn \
  --learning_rate 2e-4 --batch_size 4 \
  --captions False \
  --prefix my-run-tag

# Test
python test.py \
  --connector_path runs/<run_name>/connector.pt \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --object_type person \
  --feature_mode expert --expert_feature DINOv2 \
  --input_mode expert_and_image_attn \
  --batch_size 16 \
  --captions False \
  --test_file ../prepare_dataset/06_annotated_abcd/benchmarks/distractor_cropped_family.jsonl \
  --prefix my-run-tag
```

## Inference on a single query (shareable API)

Use [inference.py](inference.py) when you want to run the fine-tuned model on an ad-hoc query image + a variable-length gallery, without the benchmark machinery. It exposes a small `IDAVLM` class whose `predict(query_image, gallery_images)` returns the matched option letter (or the letter of the stranger slot if no gallery image matches) plus the raw model response.

### Quick start — run the example script

[example_inference.py](example_inference.py) is a minimal end-to-end example: it loads a trained connector, sends one query + five gallery images, and prints the parsed answer and full model response.

```bash
cd IDA-VLM/training
python example_inference.py
```

Edit the `connector_path`, `query_image`, and `gallery_images` at the top of the file to point at your own checkpoint and images. The rest of this section explains how to adapt it for library or CLI use.

### Library usage

```python
from inference import IDAVLM

model = IDAVLM(
    connector_path="runs/<run_name>/connector.pt",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",       # must match the base model used during training
    feature_mode="expert",                         # "vanilla" if trained with no expert
    expert_feature="wyzev0323token",              # ignored when feature_mode="vanilla"
    input_mode="expert_and_image_attn",            # "image_only" for vanilla runs
)

result = model.predict(
    query_image="/path/to/query.jpg",
    gallery_images=[
        "/path/to/gallery1.jpg",
        "/path/to/gallery2.jpg",
        "/path/to/gallery3.jpg",
    ],
)

print(result["answer"])    # "B"     (single letter; or int 1..N / -1 for legacy digit-format models, None if unparsable)
print(result["raw_text"])  # full model response, e.g. "Answer: B."
```

The gallery list can be any length ≥ 1; the prompt format (from [`../evaluation/prompt.txt`](../evaluation/prompt.txt)) and processing match training exactly. Construct the `IDAVLM` once and call `predict(...)` repeatedly — model + expert weights stay resident on GPU.

Note: for letter-format models, the API caller is responsible for tracking which letter corresponds to which gallery position when calling `predict` (the inference path inserts the stranger slot at a default location internally; for analysis-grade letter→position mapping use the lower-level message builder).

### CLI usage

```bash
cd IDA-VLM/training
python inference.py \
  --connector_path runs/<run_name>/connector.pt \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --feature_mode expert \
  --expert_feature wyzev0323token \
  --input_mode expert_and_image_attn \
  --query /path/to/query.jpg \
  --gallery /path/to/gallery1.jpg /path/to/gallery2.jpg /path/to/gallery3.jpg
```

### Picking the right `feature_mode` / `expert_feature` / `input_mode`

These must match the values the connector was trained with. The run folder name encodes them — for a folder like
`20260417_214931_distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_and_image_attn_lr_0.0002_bs_4_captions_False`, the relevant fields are `expert` → `feature_mode`, `wyzev0323token` → `expert_feature`, and `expert_and_image_attn` → `input_mode`. For vanilla (no-expert) runs, use `--feature_mode vanilla --input_mode image_only` and the `expert_feature` flag is ignored.

## Stranger (distractor) support

Both training and evaluation handle stranger queries natively. In the lettered-options format there is no special `-1` sentinel — the stranger case is just one where the correct answer letter happens to land on the stranger slot.

- Training: `_format_data_reid_letter` always inserts a text-only "stranger / not in the gallery" option at `stranger_letter_pos` (sampled per case in `prepare_jsonl.py`). For a distractor case, `answer_letter` is the letter of that slot; for a non-distractor case it's the letter of the matching gallery image.
- Loss: only the answer-letter token is preserved in the label mask, regardless of whether it points at a gallery image or the stranger slot — the model learns both behaviors uniformly.
- Eval: `extract_answer_from_caption` in `test.py` parses `"Answer: <letter>."` from both ground-truth and model output and compares letter-to-letter. Legacy digit/`-1` parsing is preserved for old-format test files.

## Outputs per run

```
runs/<run_name>/
├── connector.pt                              # Final trainable weights (mirrors the last epoch's checkpoint)
├── ckpts/
│   ├── connector_epoch_<N>.pt                # Saved at the end of every epoch (PerEpochCheckpointCallback)
│   └── connector_best_step_<N>.pt            # Lowest-eval_loss checkpoint to date (BestCheckpointCallback);
│                                             # only the current best is kept on disk, prior best is removed
├── train.log                                 # Full stdout/stderr from training
├── test.log                                  # Full stdout/stderr from evaluation runs (appended per invocation)
├── predictions_<prefix>_<scenario>.csv       # Per-case idx, label, prediction, model_output
└── (tensorboard events, if enabled)
```
