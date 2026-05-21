#!/bin/bash

# Iterate through all 10 test scenarios in the realistic benchmark suite,
# run poc_expert_to_gemini.py on each, and aggregate per-scenario accuracies
# (expert / gemini-baseline / POC) into a single summary CSV.
#
# Per-scenario artifacts:
#   results/poc_expertv04_to_gemini/poc_<scenario>.csv     (one row per case)
# Aggregate summary:
#   results/poc_expertv04_to_gemini/summary_<benchmark_tag>.csv

# Benchmarks live under prepare_dataset/06_annotated_abcd/. Two flavors:
#   benchmarks/                  realistic (gallery = household members only)
#   benchmarks_hardnegatives/    stress test (galleries padded with hard negs)
# Edit the line below to switch flavors. Hardcoded (not env-driven) on
# purpose — a stale TEST_FOLDER export in the user's shell otherwise
# silently overrides this script's value.
TEST_FOLDER="/home/tian.liu/IDA-VLM/prepare_dataset/06_annotated_abcd/benchmarks"
# TEST_FOLDER="/home/tian.liu/IDA-VLM/prepare_dataset/06_annotated_abcd/benchmarks_hardnegatives"

# Same 10 scenarios as run_gemini.sh / run_qwen.sh / run_embedding.sh, but
# referenced by stem (no extension) — poc_expert_to_gemini.py expands to
# `<scenario>.jsonl` internally.
SCENARIOS=(
    "cropped_sameclothes_singleton_samecamera"
    "cropped_sameclothes_singleton_crosscamera"
    "cropped_sameclothes_family_samecamera"
    "cropped_sameclothes_family_crosscamera"
    "cropped_crossclothes_singleton_samecamera"
    "cropped_crossclothes_singleton_crosscamera"
    "cropped_crossclothes_family_samecamera"
    # "cropped_crossclothes_family_crosscamera"
    "distractor_cropped_singleton"
    "distractor_cropped_family"
)

# Cap on cases per scenario (for smoke testing). Empty string = no cap.
# Uncomment the second line and set a small N before paying for the full sweep.
LIMIT_FLAG=""
# LIMIT_FLAG="--limit 5"

OUTPUT_DIR="results/poc_expertv04_to_gemini"
GEMINI_MODEL_NAME="gemini-2.5-pro"
mkdir -p "$OUTPUT_DIR"

TEST_FOLDER_TAG="$(basename "$TEST_FOLDER")"
SUMMARY_FILE="$OUTPUT_DIR/summary_${TEST_FOLDER_TAG}.csv"

# Per-scenario runs — each writes poc_<scenario>.csv inside OUTPUT_DIR.
for SCENARIO in "${SCENARIOS[@]}"; do
    echo ""
    echo "================================================================"
    echo "Running scenario: $SCENARIO"
    echo "================================================================"
    python poc_expert_to_gemini.py \
        --scenario "$SCENARIO" \
        --test_dir "$TEST_FOLDER" \
        --output_dir "$OUTPUT_DIR" \
        --gemini_model_name "$GEMINI_MODEL_NAME" \
        $LIMIT_FLAG
done

# ── Aggregate ──────────────────────────────────────────────────────────────
# Read every per-scenario CSV in OUTPUT_DIR, count expert_correct /
# gemini_correct / poc_correct, and emit a summary CSV with per-scenario +
# overall accuracy. Done in python so the parsing is robust to format drift.

python - "$OUTPUT_DIR" "$SUMMARY_FILE" <<'PY'
import csv, glob, os, sys

output_dir, summary_file = sys.argv[1], sys.argv[2]

rows_out = []
totals = {'n': 0, 'e': 0, 'g': 0, 'p': 0}

for csv_path in sorted(glob.glob(os.path.join(output_dir, 'poc_*.csv'))):
    scenario = os.path.basename(csv_path)[len('poc_'):-len('.csv')]
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    n = len(rows)
    def truthy(v): return str(v).strip().lower() == 'true'
    e = sum(truthy(r['expert_correct']) for r in rows)
    g = sum(truthy(r['gemini_correct']) for r in rows)
    p = sum(truthy(r['poc_correct'])    for r in rows)
    totals['n'] += n; totals['e'] += e; totals['g'] += g; totals['p'] += p
    rows_out.append({
        'scenario':         scenario,
        'n_cases':          n,
        'expert_acc':       round(e / n * 100, 2),
        'gemini_acc':       round(g / n * 100, 2),
        'poc_acc':          round(p / n * 100, 2),
        'delta_vs_gemini':  round((p - g) / n * 100, 2),
        'delta_vs_expert':  round((p - e) / n * 100, 2),
    })

if rows_out:
    n = totals['n']
    rows_out.append({
        'scenario':         'OVERALL',
        'n_cases':          n,
        'expert_acc':       round(totals['e'] / n * 100, 2),
        'gemini_acc':       round(totals['g'] / n * 100, 2),
        'poc_acc':          round(totals['p'] / n * 100, 2),
        'delta_vs_gemini':  round((totals['p'] - totals['g']) / n * 100, 2),
        'delta_vs_expert':  round((totals['p'] - totals['e']) / n * 100, 2),
    })

with open(summary_file, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

# Pretty-print to stdout.
print()
print('=' * 80)
print(f'Summary across {len(rows_out) - 1} scenarios  →  {summary_file}')
print('=' * 80)
hdr = ['scenario', 'n_cases', 'expert_acc', 'gemini_acc', 'poc_acc',
       'delta_vs_gemini', 'delta_vs_expert']
widths = [max(len(str(r.get(h, ''))) for r in rows_out + [{h: h for h in hdr}])
          for h in hdr]
def fmt_row(r):
    return '  '.join(str(r.get(h, '')).ljust(widths[i]) for i, h in enumerate(hdr))
print(fmt_row({h: h for h in hdr}))
print('-' * (sum(widths) + 2 * (len(hdr) - 1)))
for r in rows_out:
    print(fmt_row(r))
PY
