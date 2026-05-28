# Evaluation

For InternVL,

```bash
conda activate internvl_py310

. run_internvl.sh

# download the in-house DA model for evaluation
gcloud storage rsync -r gs://xlong-model-exports/2026/03/18/internvl3-8b/internvl3_8b_llm_lora_mlp_full_vqa_annotated_qa_v2_260318/internvl3_8b_llm_lora_mlp_full_vqa_annotated_qa_v2_260318_ckpt800 model_ckpt/internvl3_8b_llm_lora_mlp_full_vqa_annotated_qa_v2_260318/internvl3_8b_llm_lora_mlp_full_vqa_annotated_qa_v2_260318_ckpt800

. run_dinov2.sh

# install conda env for qwen
conda create --name qwen --clone ida-vlm
conda activate qwen
pip install transformers==4.51.3 accelerate
pip install qwen-vl-utils[decord]
```

## Embedding-based Evaluation

`eval_embedding.py` evaluates embedding models on the person ReID benchmark by computing cosine similarity between query and gallery embeddings, then deciding the option letter (or -1 for legacy) per the prompt format the rest of the pipeline expects.

### Input formats

The script auto-detects the benchmark format from the file extension:

- **`.jsonl` (preferred)** — output of `prepare_jsonl.py` from `prepare_dataset/06_annotated_abcd/`. Each line is a case carrying `stranger_letter_pos` and `answer_letter`. The script produces letter predictions (A, B, C, ...) that match the lettered-options prompt used by `eval_qwen.py` / `eval_gemini.py`.
- **`.json` (legacy)** — older `{eval_cases: [...]}` envelope with integer `label`. Predictions are integers (1..N for a gallery hit, -1 for stranger).

### How predictions are made

The embedding model only encodes images, so the stranger "option" never has an embedding. Instead:

1. Compute cosine similarity between the query embedding and each gallery embedding → `argmax` and `max_sim`.
2. **If `max_sim < --stranger_threshold`** → predict the stranger option (letter at `stranger_letter_pos` for JSONL, or `-1` for legacy JSON).
3. **Otherwise** → predict the letter at the `argmax` gallery slot, accounting for the stranger-letter offset (`argmax < stranger_pos → letter = chr('A'+argmax)`, else `chr('A'+argmax+1)`). For legacy JSON, predict `argmax+1`.

### Usage

```bash
python eval_embedding.py \
  --test_file <path>.jsonl \
  --model <model_name> \
  [--wyze_variant <variant>] \
  [--stranger_threshold 0.5]
```

### Arguments

| Argument | Default | Choices | Description |
|---|---|---|---|
| `--test_file` | (required) | | Path to a `.jsonl` (preferred) or legacy `.json` benchmark file |
| `--model` | `WYZE_embedding` | `DINOv2`, `WYZE_embedding`, `PLIP` | Embedding model to use |
| `--wyze_variant` | `v03_23_token` | `50k`, `v02_02_reid`, `v03_23_token`, `v04_15_token` | Wyze model variant (only used when `--model=WYZE_embedding`) |
| `--stranger_threshold` | `0.5` | float | Predictions with max cosine below this become the stranger option |

### Examples

```bash
# Wyze v04_15_token on the realistic benchmark suite (lettered options)
python eval_embedding.py \
  --test_file ../prepare_dataset/06_annotated_abcd/benchmarks/cropped_sameclothes_family_crosscamera.jsonl \
  --model WYZE_embedding --wyze_variant v04_15_token

# DINOv2 on the hard-negative benchmark (galleries padded with hard negs)
python eval_embedding.py \
  --test_file ../prepare_dataset/06_annotated_abcd/benchmarks_hardnegatives/distractor_cropped_singleton.jsonl \
  --model DINOv2 --stranger_threshold 0.6

# Legacy JSON benchmark still works (digit/-1 predictions)
python eval_embedding.py \
  --test_file ../prepare_dataset/04_varying_gallery_length_distractors/benchmarks/cropped_sameclothes_singleton_samecamera.json \
  --model WYZE_embedding --wyze_variant v04_15_token
```

### Batch evaluation

`run_embedding.sh` sweeps the configured models / Wyze variants over all 10 scenarios in `TEST_FOLDER`:

```bash
bash run_embedding.sh
```

The `STRANGER_THRESHOLD` and `TEST_FOLDER` are hardcoded near the top of the script (rather than env-driven) so a stale export in your shell can't silently override them. To sweep thresholds or swap the benchmark folder, edit those two lines directly. The `RESULTS_FILE` is auto-tagged with the folder basename, so realistic vs. hard-negatives runs land in different CSVs:

```
results_embedding_benchmarks.csv
results_embedding_benchmarks_hardnegatives.csv
```

Per-scenario prediction CSVs land at `results/predictions_<model_label>/predictions_<model_label>_<scenario>.csv`. Schema: `idx,label,prediction,response,query`. For JSONL-format benchmarks the `label` and `prediction` columns are single uppercase letters; for legacy JSON they are integers.

A second sidecar JSON is also written alongside the CSV at `results/predictions_<model_label>/similarities_<model_label>_<scenario>.json` containing the per-case similarity vector at 6-digit precision. Top-level keys are `test_file, model, stranger_threshold, n_cases, accuracy, cases`; each case carries `idx, query, gallery, similarity (per-gallery cosines), max_sim, argmax_gallery, label, prediction, correct` (plus `stranger_letter_pos` and `answer_letter` for letter-format cases). This sidecar is what `poc_expert_to_gemini.py` (next section) consumes.

---

## POC: Expert-to-Gemini cascade — `poc_expert_to_gemini.py`

A small two-stage pipeline that combines the Wyze v04 embedding's strong-but-noisy ranking with Gemini's visual reasoning. The expert encodes the query and each gallery image and emits cosine similarities; the POC feeds those similarities + the rank order into Gemini's user prompt as auxiliary text, while Gemini still sees the same lettered-options layout (query + gallery images + a stranger placeholder) it gets in `eval_gemini.py`. Gemini makes the final decision.

### Why "auxiliary info" rather than "rerank by expert order"

Reordering the lettered options to match the expert's ranking risks anchoring Gemini to whatever the expert thinks — but the expert is the side we don't fully trust (it generates high cosine similarity even on stranger queries). Instead, the POC keeps Gemini in charge: the lettered options keep their JSONL-prepared positions, and the expert's similarity scores arrive as hints. Gemini can use them, ignore them, or override them on visual evidence.

### What's deliberately omitted from the auxiliary text

- **No threshold-based stranger detection from the expert.** v04 over-saturates similarity (≈0.99 even when no gallery image matches), so any "if all sims are low, query is a stranger" hint to Gemini would mis-direct it.
- **No "argmax-is-the-answer" instruction.** We surface only the numbers + their rank order; Gemini decides whether to follow them.

The stranger letter is still a regular lettered option in the prompt (`X: (stranger / not in the gallery)`), so Gemini can pick it on visual grounds.

### Auxiliary-text format

Appended to the user message after the gallery images:

```
Expert similarity scores (Wyze v04_15_token cosine, higher = more similar to query):
  B: 0.9911  ← rank 1, highest similarity
  A: 0.9898  ← rank 2
```

### Inputs (all reused, no recomputation)

1. Test JSONL — `prepare_dataset/06_annotated_abcd/benchmarks/<scenario>.jsonl` (lettered-options test data)
2. Expert similarities sidecar — `results/predictions_WYZE_embedding_v04_15_token/similarities_WYZE_embedding_v04_15_token_<scenario>.json` (output of `eval_embedding.py`)
3. Gemini baseline CSV — `results/predictions_gemini_gemini-2.5-pro_result_<scenario>.csv` (output of `eval_gemini.py`)

Only the POC variant calls Gemini; the baseline is read from disk.

### Flowchart

```
                                                                              ┌────── existing ─────┐
                                                                              │ Gemini baseline CSV │
                                                                              │ (no expert info,    │
                                                                              │  from eval_gemini)  │
                                                                              └──────────┬──────────┘
                                                                                         │
   ┌──────── existing ────────┐    ┌──────── existing ────────┐                          │
   │ Test JSONL              │    │ Expert similarities JSON │                          │
   │ benchmarks/<scen>.jsonl │    │ similarities_v04_<scen>  │                          │
   │ — query, gallery,       │    │ — per-case cosine sims,  │                          │
   │   answer_letter,        │    │   expert prediction,     │                          │
   │   stranger_letter_pos   │    │   max_sim, ...           │                          │
   └────────────┬────────────┘    └─────────────┬────────────┘                          │
                │                                │                                       │
                │ for each case (idx)            │                                       │
                ▼                                ▼                                       │
   ┌────────────────────────────────────────────────────────────────────┐                │
   │ build_aux_info()                                                   │                │
   │   - map gallery indices → option letters via stranger_letter_pos   │                │
   │   - sort (letter, sim) descending                                  │                │
   │   - format as "A: 0.991  ← rank 1, …"                              │                │
   └─────────────────────────┬──────────────────────────────────────────┘                │
                             │ aux_info text                                              │
                             ▼                                                            │
   ┌────────────────────────────────────────────────────────────────────┐                │
   │ build_poc_content()                                                │                │
   │   [ system_prompt (prompt.txt),                                    │                │
   │     "Query:", <query image>,                                       │                │
   │     "A:", <gallery image>,  …  "X: (stranger / not in gallery)",   │                │
   │     aux_info  ]                                                    │                │
   └─────────────────────────┬──────────────────────────────────────────┘                │
                             │ multi-modal content                                        │
                             ▼                                                            │
                ┌─────────────────────────┐                                               │
                │ Gemini 2.5 Pro          │                                               │
                │ (Vertex AI)             │                                               │
                │  → "Answer: <letter>."  │                                               │
                └────────────┬────────────┘                                               │
                             │                                                            │
                             ▼                                                            │
   ┌────────────────────────────────────────────────────────────────────┐                │
   │ extract_prediction(text) → poc_pred (letter)                       │                │
   └─────────────────────────┬──────────────────────────────────────────┘                │
                             │                                                            │
                             │  + answer_letter (GT)                                      │
                             │  + expert_prediction (from sidecar)                        │
                             │  + gemini_baseline_prediction ◀───────────────────────────┘
                             ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ Per-case row → results/poc_expertv04_to_gemini/poc_<scen>.csv      │
   │   columns: idx, label, expert_pred, gemini_pred, poc_pred,         │
   │            *_correct, expert_max_sim, aux_info, *_response, query  │
   └────────────────────────────────────────────────────────────────────┘
                             │
                             ▼  (after all cases done)
   ┌────────────────────────────────────────────────────────────────────┐
   │ Aggregate: 3 accuracies, 2 deltas, follow/override rates           │
   │   printed to stdout; in run_poc.sh also written to summary CSV     │
   └────────────────────────────────────────────────────────────────────┘
```

### Example prompt sent to Gemini

For one real case (`#0` of `cropped_crossclothes_family_crosscamera` — gallery size 2, stranger at slot C, GT=B, expert correctly picks B, baseline Gemini wrong with C), the full multi-modal content list passed to `client.models.generate_content` is:

```
[0] (system prompt — prompt.txt, ~1.4 kB):
    "You are an expert at person re-identification. Your task is to determine which option,
     if any, shows the same person as the query image.

     You will receive:
     - 1 query image (labelled 'Query:')
     - A variable number of options labelled 'A:', 'B:', 'C:', ... Each option is either a
       gallery image showing a person, OR a text-only placeholder reading
       '(stranger / not in the gallery)'. Exactly one option in every case is the stranger
       placeholder, and its letter slot is randomized — it can appear at any letter, not
       always the last one.
     ..."

[1] (text):  "Query:"
[2] (image): <query image, e.g. .../10026312_1_..._000405_004.jpg>

[3] (text):  "A:"
[4] (image): <gallery image 0, e.g. .../10026312_0_..._000078_002.jpg>

[5] (text):  "B:"
[6] (image): <gallery image 1, e.g. .../10026312_1_..._000324_005.jpg>

[7] (text):  "C: (stranger / not in the gallery)"

[8] (text — auxiliary info appended after the gallery):
    "Expert similarity scores (Wyze v04_15_token cosine, higher = more similar to query):
       B: 0.9911  ← rank 1, highest similarity
       A: 0.9898  ← rank 2"
```

What Gemini sees — and what it doesn't — is deliberate:
- It sees **3 images** (query + 2 gallery) and **1 text-only stranger option**.
- It sees the **system prompt** explaining the lettered-options task.
- It sees the **expert ranking + numerical scores** as auxiliary text at the end, but **no instruction to follow them**.
- It does **not** see any threshold-derived "this case looks like a stranger" hint, because the v04 expert over-saturates similarity on actual strangers (we don't trust its distractor calls).

Gemini is then asked to produce the same `Answer: <letter>.` it would produce under `eval_gemini.py` — the auxiliary block is informational, not directive.

### Usage

```bash
# Activate the env that has google-genai + Vertex auth (same one used for eval_gemini.py)
conda activate ida-vlm

# Sit in the evaluation folder so the default --prompt_file=prompt.txt resolves
cd /home/tian.liu/IDA-VLM/evaluation

# Smoke test on the first 5 cases (~30 seconds, ~5 API calls)
python poc_expert_to_gemini.py --limit 5

# Full sweep on the default scenario (cropped_crossclothes_family_crosscamera, 207 cases)
python poc_expert_to_gemini.py

# Different scenario / model / output folder
python poc_expert_to_gemini.py \
  --scenario cropped_sameclothes_family_crosscamera \
  --gemini_model_name gemini-2.5-pro \
  --output_dir results/poc_expertv04_to_gemini
```

If you haven't authed the Vertex client today:

```bash
gcloud auth application-default login
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--scenario` | `cropped_crossclothes_family_crosscamera` | Scenario name without extension |
| `--test_dir` | `../prepare_dataset/06_annotated_abcd/benchmarks` | Folder with the `.jsonl` test files |
| `--expert_label` | `WYZE_embedding_v04_15_token` | Subfolder/filename label for the expert similarities |
| `--expert_dir` | `results/predictions_WYZE_embedding_v04_15_token` | Folder with the `similarities_*.json` files |
| `--gemini_baseline_dir` | `results` | Folder with the existing `predictions_gemini_*.csv` |
| `--gemini_baseline_prefix` | `predictions_gemini_gemini-2.5-pro_result` | Prefix of the baseline CSV (filename = `<prefix>_<scenario>.csv`) |
| `--gemini_baseline_path` | (auto from above) | Override to a specific baseline CSV path |
| `--gemini_model_name` | `gemini-2.5-pro` | Gemini variant to call for the POC |
| `--output_dir` | `results/poc_expertv04_to_gemini` | Where to save the per-scenario CSV |
| `--prompt_file` | `prompt.txt` | System prompt (current letter-format prompt) |
| `--project_id` | `ai-datascience-354723` | GCP project for Vertex AI |
| `--location` | `us-central1` | GCP region |
| `--limit` | (none) | Cap on cases for fast smoke testing |

### Outputs

Per-case CSV at `results/poc_expertv04_to_gemini/poc_<scenario>.csv`. Schema:

```
idx, label,
expert_pred, gemini_pred, poc_pred,
expert_correct, gemini_correct, poc_correct,
expert_max_sim,
aux_info, gemini_baseline_response, poc_response,
query
```

Terminal summary at the end:

```
============================================================
POC Results — cropped_crossclothes_family_crosscamera  (207 cases)
============================================================
  Expert (Wyze v04_15_token):   86.0%  (178/207)
  Gemini (no expert):           52.2%  (108/207)
  POC (Gemini + expert info):   XX.X%  (XXX/207)
  Δ vs Gemini alone:  +XX.X%
  Δ vs Expert alone:  +XX.X%
  POC followed expert prediction: XXX/207 (XX.X%)
  POC overrode expert prediction: XXX/207 (XX.X%)
```

The script writes rows incrementally, so a mid-run crash leaves a usable partial CSV — re-run with `--limit` adjusted or restart from scratch.

### Sweep all 10 scenarios — `run_poc.sh`

`run_poc.sh` is the multi-scenario wrapper: it loops over the same 10 scenarios used by `run_gemini.sh` / `run_qwen.sh` / `run_embedding.sh`, calls `poc_expert_to_gemini.py` per scenario, then aggregates the per-scenario CSVs into a single summary file with one row per scenario plus an `OVERALL` row.

```bash
# Smoke-test first — open run_poc.sh and uncomment LIMIT_FLAG="--limit 5", then:
bash run_poc.sh

# Full sweep (~1.9k Gemini calls, ~30-60 min depending on Gemini latency):
#   re-comment LIMIT_FLAG and run again
bash run_poc.sh
```

To switch from realistic benchmarks (`benchmarks/`) to the hard-negatives variant (`benchmarks_hardnegatives/`), edit the `TEST_FOLDER` line near the top of the script. The summary filename is auto-tagged with the folder basename so the two flavors don't overwrite each other:

```
results/poc_expertv04_to_gemini/
├── poc_<scenario>.csv                         (one per scenario; per-case rows)
├── summary_benchmarks.csv                     (sweep against benchmarks/)
└── summary_benchmarks_hardnegatives.csv       (sweep against benchmarks_hardnegatives/)
```

The aggregator scans `poc_*.csv` files in `OUTPUT_DIR` directly (counts the boolean `*_correct` columns), so it's robust to partial sweeps — if you kill the run mid-way, re-running picks up where it left off conceptually (the last incomplete scenario gets re-done from scratch since the python script overwrites that scenario's CSV) and the summary will reflect whatever scenarios completed. To exclude a scenario from the sweep, comment it out in the `SCENARIOS` array.

Aggregator output (printed to stdout and saved to the summary CSV):

```
================================================================================
Summary across N scenarios  →  results/poc_expertv04_to_gemini/summary_benchmarks.csv
================================================================================
scenario                                         n_cases  expert_acc  gemini_acc  poc_acc  delta_vs_gemini  delta_vs_expert
---------------------------------------------------------------------------------------------------------------------------
cropped_crossclothes_family_crosscamera          207      86.0        52.2        XX.X     +X.X             +X.X
cropped_crossclothes_family_samecamera           150      ...
...
OVERALL                                          1987     ...
```