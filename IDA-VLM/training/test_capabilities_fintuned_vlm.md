# Capability Comparison: Fine-tuned IDA-VLM vs. Base Qwen2.5-VL

Documentation for [`test_capabilities_fintuned_vlm.py`](test_capabilities_fintuned_vlm.py).

## What this script does

Runs a fixed list of probe questions on **one** test image with two models, prints the answers side by side, and (optionally) writes a report file:

1. **Base Qwen2.5-VL-7B** — the pretrained model, no expert wiring, no fine-tune weights. Establishes the "before" baseline for general image-description ability.
2. **Fine-tuned IDA-VLM ReID model** — the same Qwen with the `wyzev0415token` expert wired in (`expert_and_image_attn`) and the trained `connector.pt` loaded on top (trained `merger` + `expert_projector`). This is the deployed configuration.
3. *(Optional, `--include_no_expert`)* — the fine-tuned model run **without** expert injection. Skips the convex mix of expert features into image tokens, so the only difference from base Qwen is the fine-tuned `merger`. Useful for isolating *merger drift* from *expert injection* as the cause of any quality drop.

The two models are loaded **sequentially**, not simultaneously, so a single 7B-capable GPU is enough. Each model loads → answers all questions → frees memory before the next loads.

## Requirements

- Run from `IDA-VLM/training/` (the script imports `replace`, `utils`, and `experts/` via relative paths).
- A working IDA-VLM training environment (same as training/eval — see [README.md § Dependencies](README.md#dependencies)). In particular: the vendored `experts/wyze_embedding` checked out to the `dev/neil/petreid` branch so the `wyzev0415token` weights can load.
- A GPU with enough memory for Qwen2.5-VL-7B in bf16 (≈ 16 GB).
- The fine-tuned `connector.pt` checkpoint at the path passed to `--connector_path` (defaults to the `wyzev0415token` ABCD run; edit the constant at the top of the script or pass `--connector_path` to use a different one).

## Default test image

The default image is checked into [`examples/`](examples/) so the script runs out of the box:

```
examples/9991418_9996314_1_D03F278DB00F_D03F278DB00F131749415111_000000_005_fullframe.jpg
```

A full-frame Wyze camera capture showing a person in a white t-shirt, smoking a cigarette, wearing a black watch, walking next to a parked car. The default question list probes multiple visual concepts present in this image (person attributes, clothing, accessories, action, vehicle, scene). Other useable images in `examples/`:

- `9991418_9996314_1_D03F278DB00F_..._005.jpg` — cropped version of the same scene (person only).
- `10868404_0_D03F275D4BB3_..._001.jpg`, `1960853_1_2CAA8E379D04_..._004.jpg`, `7205943_7216348_..._011.jpg` — other cropped person captures from the Wyze dataset.

## Default probe questions

The script ships with two groups of questions targeting different aspects of visual understanding.

**Open-ended / scene-level** — free-form generation; sensitive to overall captioning quality:

| # | Question | What it probes |
|---|----------|----------------|
| 1 | Describe this image in detail. | Free-form captioning |
| 2 | What is the person in the image doing? | Action recognition |
| 3 | What is the person wearing? | Clothing description |
| 4 | Is the person holding or smoking anything? Describe it. | Fine-grained interaction |
| 5 | Is the person wearing any accessories such as a watch or jewelry? | Small-object recall (the watch) |
| 6 | Is there a vehicle in the image? If so, describe it. | Non-person object |
| 7 | What objects can you see in this image besides the person? | Scene parsing |
| 8 | How many people are visible in this image? | Counting / localization |

**Fine-grained attribute probes** — short-answer / yes-no; mix of true positives and true negatives so you can spot hallucination in either direction. These are the most diagnostic when comparing models:

| # | Question | Expected | What it probes |
|---|----------|----------|----------------|
|  9 | Is the man wearing glasses? Answer yes or no. | no  | True-negative — hallucination check |
| 10 | Is the man smoking? Answer yes or no.         | yes | Action / object-in-hand detection |
| 11 | Is the man wearing a hat? Answer yes or no.   | no  | True-negative — hallucination check |
| 12 | Is the man wearing a watch? Answer yes or no. | yes | Small-object recall |
| 13 | Is the man indoors or outdoors?               | outdoors | Scene context |
| 14 | What color is the man's t-shirt?              | white | Color attention |
| 15 | What color are the man's pants?               | (depends) | Color attention |
| 16 | What color is the man's watch?                | black | Color of small object |
| 17 | What is the man holding in his hand?          | cigarette | Held-object identification |
| 18 | What color is the car in the image?           | (depends) | Color attention on background object |

Edit the `QUESTIONS` list at the top of the script if you want different probes, or override on the CLI with `--questions`. Yes/no and color questions are most useful for objective grading — the answer is short and a wrong answer is unambiguous.

## Running

All commands assume you are in `IDA-VLM/training/`.

### Quick start — default two-way comparison

```bash
python test_capabilities_fintuned_vlm.py
```

Loads base Qwen → answers all 9 questions, loads fine-tuned (with expert) → answers all 9 questions, prints both side by side. Output stays in the terminal.

### Three-way comparison (recommended)

```bash
python test_capabilities_fintuned_vlm.py --include_no_expert
```

Adds a third column for the fine-tuned model with **expert injection bypassed**. This is the most informative run: it lets you tell *why* any degradation happens.

- If the bypass output is close to base Qwen → merger drift is mild; expert injection is the main contributor to quality drop.
- If the bypass output is also degraded → the merger itself has drifted enough to hurt general VL chat.

### Save the report to a file

```bash
python test_capabilities_fintuned_vlm.py \
    --output_file capability_report.txt
```

Writes the side-by-side report to `capability_report.txt` (in addition to printing it). Easier to share or diff across checkpoints.

### Use a different image from `examples/`

```bash
# Cropped person crop (same scene, no surroundings)
python test_capabilities_fintuned_vlm.py \
    --image examples/9991418_9996314_1_D03F278DB00F_D03F278DB00F131749415111_000000_005.jpg

# A different person crop
python test_capabilities_fintuned_vlm.py \
    --image examples/10868404_0_D03F275D4BB3_D03F275D4BB3131711462612_000072_001.jpg
```

### Override the question list

```bash
python test_capabilities_fintuned_vlm.py \
    --questions \
        "Describe the scene in two sentences." \
        "What is the dominant color in the image?" \
        "Is this taken indoors or outdoors?"
```

You can pass any number of questions; each becomes one row in the side-by-side report.

### Use a different connector / model

```bash
# Different checkpoint (e.g. the wyzev03_23_token run)
python test_capabilities_fintuned_vlm.py \
    --connector_path runs/<some_other_run>/ckpts/connector_best_step_XXXXX.pt

# Different base model size (3B)
python test_capabilities_fintuned_vlm.py \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --connector_path runs/<a_3b_run>/connector.pt
```

The script does **not** auto-parse the expert/feature config from the connector path (`run_test.sh` does). The defaults in the script assume `expert_and_image_attn + wyzev0415token`. If you point it at a connector trained with a different expert or input_mode, also edit the `load_finetuned` function near the bottom of the script (`expert_feature=...`, `input_mode=...`).

### Longer / shorter responses

```bash
# Cap each response at 100 new tokens (fast)
python test_capabilities_fintuned_vlm.py --max_new_tokens 100

# Allow longer descriptions (default is 300)
python test_capabilities_fintuned_vlm.py --max_new_tokens 600
```

## Output format

The terminal first streams each `Q/A` pair as it runs, then at the end prints a consolidated side-by-side report. With `--include_no_expert`, each block contains three answers:

```
--- Q1: Describe this image in detail. ---

[base Qwen]
<long descriptive caption>

[fine-tuned + expert (deployed)]
<may be terser / more identity-flavored>

[fine-tuned, expert injection bypassed]
<usually close to base Qwen if merger drift is mild>
```

## Interpreting results — quick guide

1. **Fluent text on all three columns** → LLM is intact (it's frozen, so this is expected).
2. **Base ≈ fine-tuned (with expert)** → minimal degradation; the fine-tune barely touched general VL ability.
3. **Fine-tuned with expert is shorter / fixates on identity / mentions clothing or hair too much for unrelated prompts** → expert injection is biasing image tokens toward person-ID semantics. The `--include_no_expert` column should look better.
4. **Both fine-tuned columns are degraded** → merger drift is also non-trivial. Consider lowering the merger learning rate next run, or co-training with a captioning mix to preserve general VL.
5. **Non-person questions (Q7 vehicle, Q8 objects) are much worse on fine-tuned columns than person questions (Q2-Q6)** → expected: the ReID fine-tune specialized the visual stream around persons. Co-training is the durable fix.

## Caveats

- **Quantitative comparison requires more than one image.** This script is a fast qualitative probe. For systematic evaluation, run on a held-out captioning benchmark (e.g. a subset of VQA / COCO captions) and grade with GPT-4 or human judgement.
- **The script does not call the `format_data_reid` ReID prompt.** It uses a plain one-turn user message. This means you are testing the model in a *different* prompt regime than it was fine-tuned on, which is exactly what we want here — but it also means a "good" result on this script does *not* guarantee good ReID performance, and a "bad" result does not necessarily hurt ReID accuracy. Use [`test.py`](test.py) for ReID benchmark evaluation.
- **Greedy decoding only.** `do_sample=False` is hard-coded so outputs are deterministic across runs. To inspect sampling variance, edit `ask()` in the script.
