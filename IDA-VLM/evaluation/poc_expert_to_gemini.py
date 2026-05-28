"""POC: feed Wyze v04 expert ranking + similarities as auxiliary text to
Gemini, and compare three accuracies on a single test scenario:

  - Expert alone (Wyze v04_15_token, decisions cached in similarities_*.json)
  - Gemini alone (no expert hint, decisions cached in
    predictions_gemini_gemini-2.5-pro_result_<scenario>.csv)
  - POC = Gemini + expert auxiliary info (re-runs Gemini once per case)

Defaults are tuned for `cropped_crossclothes_family_crosscamera` (the
scenario where expert and Gemini diverge the most — biggest signal). Each
case sends Gemini the same lettered-options prompt as eval_gemini.py
PLUS a short text block listing the per-option expert similarity scores
(highest first) and a note that the stranger option has no expert score.
"""

import argparse
import csv
import json
import os
import re

from PIL import Image
from google import genai
from google.genai import types
from tqdm import tqdm


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_test_cases(path):
    """Load JSONL test cases (output of prepare_jsonl.py)."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_expert_payload(path):
    """Load similarities_*.json (output of eval_embedding.py).

    Returns (cases_by_idx: {idx: case_dict}, top_level_payload).
    """
    with open(path) as f:
        payload = json.load(f)
    cases_by_idx = {c['idx']: c for c in payload['cases']}
    return cases_by_idx, payload


def load_gemini_baseline(path):
    """Read existing Gemini baseline predictions CSV.

    Schema: idx, label, prediction, response, query.
    Returns: {idx: {'prediction': str, 'response': str}}.
    """
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['idx'])
            out[idx] = {
                'prediction': row['prediction'].strip(),
                'response':   row.get('response', ''),
            }
    return out


# ── Helpers ─────────────────────────────────────────────────────────────────

def extract_prediction(text):
    """Pull the single-letter answer out of a Gemini response. Mirrors
    eval_gemini.extract_prediction (letter-format path only)."""
    m = re.search(r'Answer:\s*([A-Z])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-Z])\b', text)
    if m:
        return m.group(1)
    return None


def gallery_index_to_letter(g_idx, stranger_letter_pos):
    """Map a 0-based gallery index to its option letter, accounting for the
    stranger slot inserted at `stranger_letter_pos`."""
    opt_idx = g_idx if g_idx < stranger_letter_pos else g_idx + 1
    return chr(ord('A') + opt_idx)


def build_aux_info(case, expert_record):
    """Format the expert's per-gallery cosine similarities as auxiliary
    text Gemini will see appended to the user message.

    We surface ONLY the similarity numbers and rank order — no stranger
    guidance. Rationale: the v04 expert produces saturated similarities
    even on distractor (stranger) queries, so any threshold-based
    "low score = stranger" hint we hand Gemini would mis-direct it. The
    stranger letter still appears in the prompt as a regular lettered
    option ("X: (stranger / not in the gallery)"), so Gemini knows it's
    available; we just don't editorialize about which way to use it.

    Layout (sorted by similarity, highest first):

        Expert similarity scores (Wyze v04_15_token cosine, higher = more
        similar to query):
          B: 0.9911  ← rank 1, highest similarity
          A: 0.9898  ← rank 2
    """
    stranger_pos = case['stranger_letter_pos']
    similarity   = expert_record['similarity']

    pairs = [
        (gallery_index_to_letter(i, stranger_pos), sim)
        for i, sim in enumerate(similarity)
    ]
    pairs.sort(key=lambda x: -x[1])

    lines = [
        "Expert similarity scores (Wyze v04_15_token cosine, "
        "higher = more similar to query):",
    ]
    for rank_i, (letter, sim) in enumerate(pairs, start=1):
        suffix = "  ← rank 1, highest similarity" if rank_i == 1 else f"  ← rank {rank_i}"
        lines.append(f"  {letter}: {sim:.4f}{suffix}")
    return "\n".join(lines)


def build_poc_content(case, prompt_template, aux_info, data_folder=''):
    """Build the Gemini multi-modal content list.

    Same lettered-options layout as eval_gemini.build_letter_content
    (system prompt → query image → A: <image> / B: (stranger) / C: <image>
    ...) with the expert auxiliary info appended after the gallery.
    """
    n_gallery    = len(case['gallery'])
    stranger_pos = case['stranger_letter_pos']

    query_img = Image.open(os.path.join(data_folder, case['query']))
    content   = [prompt_template, "Query:", query_img]

    g_idx = 0
    for opt_idx in range(n_gallery + 1):
        letter = chr(ord('A') + opt_idx)
        if opt_idx == stranger_pos:
            content.append(f"{letter}: (stranger / not in the gallery)")
        else:
            img = Image.open(os.path.join(data_folder, case['gallery'][g_idx]))
            content.append(f"{letter}:")
            content.append(img)
            g_idx += 1

    content.append(aux_info)
    return content


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',
                        default='cropped_crossclothes_family_crosscamera',
                        help='Scenario name without extension')
    parser.add_argument('--test_dir',
                        default='/home/tian.liu/IDA-VLM/prepare_dataset/06_annotated_abcd/benchmarks',
                        help='Folder with the .jsonl test files')
    parser.add_argument('--expert_label',
                        default='WYZE_embedding_v04_15_token',
                        help='Folder/filename label for the expert similarities')
    parser.add_argument('--expert_dir',
                        default='/home/tian.liu/IDA-VLM/evaluation/results/predictions_WYZE_embedding_v04_15_token',
                        help='Folder containing the similarities_*.json files')
    parser.add_argument('--gemini_baseline_path',
                        default=None,
                        help='Path to the existing Gemini baseline CSV. If '
                             'omitted, builds it from --gemini_baseline_dir + '
                             '--gemini_baseline_prefix + --scenario.')
    parser.add_argument('--gemini_baseline_dir',
                        default='/home/tian.liu/IDA-VLM/evaluation/results',
                        help='Folder containing predictions_gemini_*.csv files')
    parser.add_argument('--gemini_baseline_prefix',
                        default='predictions_gemini_gemini-2.5-pro_result',
                        help='Prefix of the Gemini baseline CSV '
                             '(filename = <prefix>_<scenario>.csv).')
    parser.add_argument('--gemini_model_name', default='gemini-2.5-pro')
    parser.add_argument('--output_dir', default='results/poc_expertv04_to_gemini')
    parser.add_argument('--prompt_file', default='prompt.txt')
    parser.add_argument('--project_id',  default='ai-datascience-354723')
    parser.add_argument('--location',    default='us-central1')
    parser.add_argument('--limit', type=int, default=None,
                        help='Cap on cases (for fast smoke test). '
                             'Omit to run the whole scenario.')
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    test_file = os.path.join(args.test_dir, f'{args.scenario}.jsonl')
    expert_file = os.path.join(
        args.expert_dir,
        f'similarities_{args.expert_label}_{args.scenario}.json',
    )
    if args.gemini_baseline_path:
        gemini_baseline_file = args.gemini_baseline_path
    else:
        gemini_baseline_file = os.path.join(
            args.gemini_baseline_dir,
            f'{args.gemini_baseline_prefix}_{args.scenario}.csv',
        )

    print(f'scenario:             {args.scenario}')
    print(f'test_file:            {test_file}')
    print(f'expert_file:          {expert_file}')
    print(f'gemini_baseline_file: {gemini_baseline_file}')

    # ── Load all three sources ────────────────────────────────────────────
    cases         = load_test_cases(test_file)
    expert_by_idx, _ = load_expert_payload(expert_file)
    gemini_by_idx = load_gemini_baseline(gemini_baseline_file)

    test_indices    = set(range(len(cases)))
    expert_indices  = set(expert_by_idx.keys())
    gemini_indices  = set(gemini_by_idx.keys())
    common = test_indices & expert_indices & gemini_indices

    print(f'cases: test={len(test_indices)} expert={len(expert_indices)} '
          f'gemini_baseline={len(gemini_indices)} common={len(common)}')
    if not common:
        raise SystemExit('No overlapping case indices across the three sources.')

    work_indices = sorted(common)
    if args.limit:
        work_indices = work_indices[:args.limit]
        print(f'(limited to first {args.limit} cases)')

    # ── Prompt + Gemini client ────────────────────────────────────────────
    with open(args.prompt_file) as f:
        prompt_template = f.read()
    client = genai.Client(
        vertexai=True,
        project=args.project_id,
        location=args.location,
    )

    # ── Output CSV ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f'poc_{args.scenario}.csv')
    fieldnames = [
        'idx', 'label',
        'expert_pred', 'gemini_pred', 'poc_pred',
        'expert_correct', 'gemini_correct', 'poc_correct',
        'expert_max_sim',
        'aux_info',
        'gemini_baseline_response',
        'poc_response',
        'query',
    ]

    expert_correct = gemini_correct = poc_correct = total = 0

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in tqdm(work_indices):
            case          = cases[idx]
            gt            = case['answer_letter']
            expert_record = expert_by_idx[idx]
            expert_pred   = expert_record['prediction']
            gemini_baseline = gemini_by_idx[idx]
            gemini_pred   = gemini_baseline['prediction']

            aux_info = build_aux_info(case, expert_record)
            content  = build_poc_content(case, prompt_template, aux_info)

            try:
                response = client.models.generate_content(
                    model=args.gemini_model_name,
                    contents=content,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                    ),
                )
                poc_response = response.text or ''
                poc_pred     = extract_prediction(poc_response)
            except Exception as e:
                poc_response = f'ERROR: {e}'
                poc_pred     = None
                print(f'[idx {idx}] error: {e}')

            ec = str(expert_pred) == str(gt)
            gc = str(gemini_pred) == str(gt)
            pc = poc_pred is not None and str(poc_pred) == str(gt)
            expert_correct += int(ec)
            gemini_correct += int(gc)
            poc_correct    += int(pc)
            total += 1

            writer.writerow({
                'idx': idx,
                'label': gt,
                'expert_pred':    expert_pred,
                'gemini_pred':    gemini_pred,
                'poc_pred':       poc_pred,
                'expert_correct': ec,
                'gemini_correct': gc,
                'poc_correct':    pc,
                'expert_max_sim': expert_record.get('max_sim'),
                'aux_info':       aux_info.replace('\n', ' | '),
                'gemini_baseline_response': (
                    gemini_baseline.get('response', '')
                    .replace('\n', ' ')
                    .replace(',', ';')
                ),
                'poc_response':   (
                    poc_response.replace('\n', ' ').replace(',', ';')
                ),
                'query':          case['query'],
            })

    print(f'\nResults saved → {out_csv}')

    # ── Accuracy summary ──────────────────────────────────────────────────
    if total == 0:
        return
    e_acc = expert_correct / total * 100
    g_acc = gemini_correct / total * 100
    p_acc = poc_correct    / total * 100

    bar = '=' * 60
    print()
    print(bar)
    print(f'POC Results — {args.scenario}  ({total} cases)')
    print(bar)
    print(f'  Expert (Wyze v04_15_token):  {e_acc:5.1f}%  ({expert_correct}/{total})')
    print(f'  Gemini (no expert):          {g_acc:5.1f}%  ({gemini_correct}/{total})')
    print(f'  POC (Gemini + expert info):  {p_acc:5.1f}%  ({poc_correct}/{total})')
    print(f'  Δ vs Gemini alone:  {p_acc - g_acc:+5.1f}%')
    print(f'  Δ vs Expert alone:  {p_acc - e_acc:+5.1f}%')

    # Cross-tab: how often did the POC follow the expert vs override it
    follow_expert = override_expert = 0
    for idx in work_indices:
        e_p = expert_by_idx[idx]['prediction']
        # We need the POC pred — easier to recompute from the row we just wrote,
        # but we already have it via the loop variables only for the last case.
        # For the cross-tab, re-read the CSV (cheap).
    with open(out_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['poc_pred'] == row['expert_pred']:
                follow_expert += 1
            else:
                override_expert += 1
    print(f'  POC followed expert prediction: {follow_expert}/{total} '
          f'({follow_expert/total*100:.1f}%)')
    print(f'  POC overrode expert prediction: {override_expert}/{total} '
          f'({override_expert/total*100:.1f}%)')


if __name__ == '__main__':
    main()
