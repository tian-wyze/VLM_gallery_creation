# Family-Member Gallery from Video — Pipeline

End-to-end pipeline that turns a set of video clips into a **family-member gallery** — i.e. a list of unique people seen across the videos, with every person crop that belongs to each identity.

```
                                        ┌─────────────────┐
                                        │ identities.json │
                                        │ person_001 → [..]│
                                        │ person_002 → [..]│
                                        │ ...             │
                                        └─────────────────┘
                                                ▲
example_videos/*.mp4                            │
        │                                       │
        ▼ (1 fps)                               │
  YOLOv8n person detector ──► crops/crop_NNNN.jpg
        │                              │
        ▼                              ▼
 crops_metadata.jsonl    IDA-VLM ReID model (query crop vs running gallery)
                                       │
                                       ▼
                              decisions.jsonl
                                       │
                                       ▼
                            visualization.html
```

The pipeline runs in **two phases** in one process. YOLO is freed from GPU memory before the 7B VLM is loaded, so a single 40-GB-class GPU is enough.

  1. **DETECT** — Walk every video at 1 fps, run YOLOv8, save every person crop + write `crops_metadata.jsonl`.
  2. **ASSIGN** — Read the metadata in chronological order. For each crop: if the gallery is empty, open identity #1; otherwise query the fine-tuned VLM with (query=this crop, gallery=one rep per existing identity, plus a stranger slot). On `stranger`, open a new identity; on a letter match, append to that identity.

Separating the phases means you can also re-run just one (`--phase detect` or `--phase assign`), or inspect / curate crops before paying for the VLM step.

## Files

```
gallery_creation/
├── build_gallery.py     # pipeline (CLI)
├── visualize.py         # state/ → state/visualization.html (static)
├── serve.py             # tiny HTTP server for remote browser viewing
├── README.md            # this file
├── example_videos/      # three sample clips (1080p / 1536p)
│   ├── 0a865705a0d0419c89fdf491917bc89e.mp4
│   ├── 0af545942fc64f048e03b04ffaede2e5_-1.mp4
│   └── 0affb6606ee9410481c85a94474acfcc_-1.mp4
└── state/                       # written at runtime
    ├── crops/
    │   ├── crop_0001.jpg        # every YOLO crop, monotonic id across all videos
    │   └── ...
    ├── crops_metadata.jsonl     # one line per crop with video, frame, bbox, det_conf
    ├── decisions.jsonl          # one line per crop with VLM decision trace
    ├── identities.json          # final {person_id: [crop_paths]}
    └── visualization.html       # self-contained side-by-side viz
```

## Requirements

- The IDA-VLM training environment (`ida-vlm` conda env, or equivalent — same prerequisites as `IDA-VLM/training/`). In particular, the vendored `experts/wyze_embedding` checked out to the `dev/neil/petreid` branch so `wyzev0415token` loads.
- `ultralytics` (for YOLOv8). Already in the env. Weights `yolov8n.pt` (~6 MB) auto-download on first run.
- The fine-tuned IDA-VLM `connector.pt` at the path passed via `--connector_path`. Defaults to the `wyzev0415token` ABCD-format checkpoint (`runs/20260502_014222_…/ckpts/connector_best_step_38000.pt`).
- A GPU with ≥ 16 GB free (Qwen2.5-VL-7B in bf16). Tested on an A100-40GB.

## Quickstart

```bash
cd IDA-VLM/gallery_creation/

# Run the full pipeline on example_videos/ (~30s on A100-40GB).
python build_gallery.py --clean

# Generate the HTML viz.
python visualize.py
# → opens state/visualization.html
```

Expected smoke-test output on the 3 example videos: **5 identities discovered across 15 person crops** (1 + 1 + 7 + 4 + 2). Distributions vary slightly run-to-run if YOLO's NMS picks slightly different boxes.

## Command examples

All commands assume cwd is `IDA-VLM/gallery_creation/`.

### Run both phases (default)

```bash
python build_gallery.py
python build_gallery.py --clean                  # wipe state/ first
```

### Only detect crops (no VLM)

Useful if you want to inspect / curate the crops before running the slow VLM step.

```bash
python build_gallery.py --phase detect
# → state/crops/* and state/crops_metadata.jsonl
```

Open the crops in a file browser, delete any junk crops, then run:

```bash
python build_gallery.py --phase assign
```

`--phase assign` reads `crops_metadata.jsonl` and walks the crops in order; deleting a line from the metadata file (or deleting a crop file) skips that crop in the assignment pass.

### Process a single video

```bash
python build_gallery.py \
    --videos example_videos/0af545942fc64f048e03b04ffaede2e5_-1.mp4 \
    --clean
```

### Use your own videos

Point `--videos` at any directory of `.mp4 / .mov / .avi / .mkv` files, or at a single file path:

```bash
python build_gallery.py --videos /path/to/my/videos --clean
```

### Tune frame sampling / detector confidence

```bash
# Sample 2 frames per second instead of 1
python build_gallery.py --fps 2.0

# Stricter person threshold — drop low-conf false positives (default is 0.5)
python build_gallery.py --det_conf 0.8
```

### Switch person detector backend

Two interchangeable backends are wired in:

```bash
# Default — Wyze YOLOv11 (in-house, trained on Wyze footage; weights ship
# at training/experts/wyze_embedding/models/od/v11_07_25.onnx).
python build_gallery.py --detector wyze

# Fallback — off-the-shelf YOLOv8n (COCO 80-class) via ultralytics.
python build_gallery.py --detector yolov8

# Point the Wyze detector at a different weights file
python build_gallery.py --detector wyze \
    --wyze_detector_weights /path/to/v11_xx_yy.onnx
```

`--det_conf` controls the person-class threshold for both backends. The Wyze detector also auto-installs `onnxruntime-gpu` on first use if not already in the env.

### Use a different VLM connector

```bash
python build_gallery.py \
    --connector_path runs/<other_run>/connector.pt \
    --expert_feature wyzev0323token \
    --input_mode expert_and_image_attn
```

Make sure `--expert_feature` and `--input_mode` match the connector you point at (the directory name encodes them — see `IDA-VLM/training/README.md`).

### Cap videos processed (debug)

```bash
python build_gallery.py --max_videos 1
```

### Visualize a specific state directory

```bash
python visualize.py --state_dir state --out state/visualization.html
python visualize.py --state_dir state --out my_report.html
```

`visualize.py` is purely a renderer — it reads `identities.json` + `decisions.jsonl` and emits HTML. You can re-run it any time without touching the pipeline.

**Image embedding.** By default every thumbnail is **embedded into the HTML as a base64 data URI**, so the file is fully self-contained and renders correctly in VS Code's preview pane as well as in any browser. The trade-off is a larger file (~500 KB per crop). For very large runs (thousands of crops), pass `--no_embed` to use relative paths instead — that produces a tiny HTML but only renders when opened in a real browser from inside `state_dir`:

```bash
python visualize.py --no_embed       # smaller file, needs real browser
```

## Viewing the viz on a remote server

If your `state/` lives on a remote machine but you want to view it in Chrome on your laptop, the cleanest approach is to run [`serve.py`](serve.py) on the remote and tunnel its port over SSH:

```bash
# On the remote machine:
cd IDA-VLM/gallery_creation
python serve.py            # listens on 127.0.0.1:8088 by default
```

```bash
# On your laptop:
ssh -L 8088:localhost:8088 <user>@<remote-host>
# leave that ssh session open, then in Chrome:
#   http://localhost:8088/
```

`serve.py`:
- Re-renders the HTML on **every** request from the current contents of `identities.json` + `decisions.jsonl`, so refreshing the browser picks up the latest pipeline state without restarting anything. Handy while a long pipeline is still running.
- Serves images directly from disk (`/crops/crop_NNNN.jpg`) — no base64 bloat, instant load.
- Binds to `127.0.0.1` by default (only reachable via the SSH tunnel). Pass `--host 0.0.0.0` to expose it on the LAN (no auth — use with care).
- Other useful flags: `--port 8081` if 8088 is busy, `--state_dir /path/to/state` to point at a different state dir.

Alternative (no server): generate a self-contained HTML with images embedded as base64 and `scp` the single file to your laptop:

```bash
# On remote:
python visualize.py --out /tmp/gallery.html       # embedded mode, default
# On laptop:
scp <user>@<remote>:/tmp/gallery.html .
open gallery.html
```

## What the HTML viz shows

Open `state/visualization.html` in any modern browser. Two sections:

### Final family gallery

One row per `person_NNN`. The first crop (with a green `rep` badge) is the *representative image* — the one that was actually used as the gallery slot when querying the VLM for subsequent crops. Following crops are the additional matches assigned to that identity.

### Decision log

One card per crop, in chronological order. Each card shows:

| Column | What it is |
|---|---|
| **Query** | the YOLO crop we're trying to place |
| **Gallery state at query time** | letter-labelled thumbnails of identities A, B, C, … that existed *before* this crop was queried, plus a dashed "stranger" slot at the end |
| **Outcome** | the model's `Answer: X.` raw text, the parsed answer letter, the resulting verdict (matched / new), and which `person_NNN` this crop ended up in |

This is the "process visualization" — you can read top-to-bottom to see exactly how the gallery grew, and which model answer drove each decision.

## Tuning notes

- **Gallery rep choice.** The current code uses the *first* crop assigned to each identity as the gallery rep. This matches the user's spec ("simply take the first person crop as the first gallery image") and keeps each VLM call cheap, but if the first crop is unusually low-quality (e.g. truncated by frame edge) downstream matching may suffer. To experiment, edit `GalleryState.gallery_reps()` to e.g. pick the *highest-detection-conf* crop per identity.
- **Crop padding.** `--crop_pad 8` adds 8 pixels around the YOLO bbox to capture a little context (hair, shoes). Larger pads help the VLM but risk including a second person.
- **`stranger_letter_pos`.** The pipeline relies on `IDAVLM.predict(..., stranger_letter_pos=None)` which puts the stranger slot at the *last* letter (after N identities). The fine-tuned model was trained with the stranger slot at a random position per case, so this is in-distribution for it.
- **N grows unbounded.** As more identities are added, the gallery sent to each VLM call grows. Empirically this is fine for small families (≤ ~10 identities). At scale you may want to cap the gallery (e.g. send only the K most recently-seen identities) — that's a one-line change in `run_assign`.
- **Long videos.** The pipeline streams frames one at a time so memory stays flat; the only growing artefact is `crops/`. For a 1-hour video at 1 fps with one person per frame, expect ~3 600 crops (~50 MB on disk) and ~3 600 VLM calls (~30 min on A100).

## Output schema (for downstream tooling)

### `crops_metadata.jsonl`

One JSON object per line (chronological order across all videos):

```json
{"crop_id": 7, "crop_path": "crops/crop_0007.jpg",
 "video": "0af545942fc64f048e03b04ffaede2e5_-1.mp4",
 "frame_idx": 105, "timestamp_s": 7.0,
 "bbox": [12, 84, 1188, 1530], "det_conf": 0.887}
```

### `decisions.jsonl`

Same fields as above plus:

```json
{... ,
 "gallery_size_before": 3,
 "gallery_reps_before": ["crops/crop_0001.jpg", "crops/crop_0002.jpg", "crops/crop_0003.jpg"],
 "stranger_letter": "D",
 "raw_text": "Answer: C.",
 "answer_letter": "C",
 "verdict": "matched",
 "assigned_to": "person_003"}
```

`verdict` is one of: `matched`, `new (stranger)`, `new (gallery_empty)`, `new (unparsable)`.

### `identities.json`

```json
{
  "person_001": ["crops/crop_0001.jpg"],
  "person_003": ["crops/crop_0003.jpg", "crops/crop_0004.jpg", ...],
  ...
}
```

## Limitations / known caveats

- **No multi-shot gallery rep.** Each identity contributes exactly one gallery slot (the first crop). If a person's appearance changes substantially between videos (lighting, clothing, pose), the VLM may mis-classify them as a stranger and open a duplicate identity. Mitigation: extend `gallery_reps()` to send the top-K most representative crops, or pick the highest-confidence detection.
- **No verification / post-merge step.** Once an identity is opened, it's never re-merged. If two duplicate identities appear (`person_002` and `person_005` are actually the same person), nothing reconciles them. A cheap post-process: run all-pairs VLM queries on the identities' reps and merge any pair that the model classifies as a match.
- **YOLO is the bottleneck for accuracy.** A missed detection means a missed crop; a false-positive crop produces a junk gallery member. Tune `--det_conf` for your footage.
- **First crop, no model call.** When the gallery is empty, the first detected crop is added unconditionally. If that first detection happens to be a bird or a low-quality false positive, you'll start the gallery with a junk identity. Easiest fix: pre-filter by detection conf + min bbox size.
