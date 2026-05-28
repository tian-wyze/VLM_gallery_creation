"""
Convert train_data.json (output of prepare_train.py) into train_data.jsonl
with letter-based options (A, B, C, ...) and a randomized "stranger" slot.

For each case, one extra option representing "the query person is not in
the gallery" is inserted at a random letter position. So a case with N
gallery images becomes N+1 lettered options. The model is expected to
output a single letter (A..) — for distractor cases the correct letter is
the stranger slot, for non-distractor cases it is the slot holding the
matching gallery image.

Per-line schema (one JSON object per line):

  {
    # carried over from prepare_train.py output
    query, gallery, label, similarity, household_id, identity_id,
    query_household_id, query_identity_id, query_status,

    # new fields
    "stranger_letter_pos": int,   # 0..len(gallery), insertion index of stranger
    "answer_letter":       str,   # "A".."<chr(ord('A')+len(gallery))>"
  }

The downstream training data loader is expected to derive the option list
by inserting a "stranger" placeholder at `stranger_letter_pos` into the
gallery sequence (image, image, ..., image), then assigning the i-th
option the letter chr(ord('A') + i). The placeholder option is rendered
text-only ("X: (the query person is not in the gallery)") since it has
no image; gallery options carry the corresponding image.

Notes
- Letter count varies with gallery size (a size-9 case yields letters
  A..J). The training loader must accept variable-length option blocks.
- Randomization is seeded for reproducibility; re-running with the same
  seed and input yields identical output.
"""

import argparse
import json
import random
from collections import Counter


def load_cases(path):
    """Load cases from train or test JSON.

    Accepted shapes:
      - flat list of cases (output of prepare_train.py: train_data.json)
      - dict with key "eval_cases" (output of prepare_test.py: benchmark
        files like .../benchmarks/<scenario>.json)
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'eval_cases' in data:
        return data['eval_cases']
    raise ValueError(
        f"Unrecognized JSON shape in {path}: expected a list of cases or "
        f"a dict with 'eval_cases'.")


def assign_letters(case, rng):
    """Pick a random stranger letter slot and compute the answer letter.

    Returns (stranger_letter_pos, answer_letter).

    stranger_letter_pos is uniform over [0, N] inclusive — i.e. the
    stranger slot can land before any gallery image, between any two
    gallery images, or after all of them, for N+1 distinct positions.
    """
    n = len(case['gallery'])
    s = rng.randint(0, n)

    if case['label'] == -1:
        # Distractor: the stranger slot is the correct answer.
        ans = chr(ord('A') + s)
    else:
        # Non-distractor: target lives at gallery index `target_idx` (0-based).
        # In the option list, gallery items before `s` keep their index;
        # items at or after `s` shift right by one to make room for stranger.
        target_idx = case['label'] - 1
        shifted = target_idx if target_idx < s else target_idx + 1
        ans = chr(ord('A') + shifted)

    return s, ans


def main():
    parser = argparse.ArgumentParser(
        description='Add randomized stranger letter slots to prepare_train.py '
                    'output and write as JSONL.')
    parser.add_argument('--input', default='train_data.json',
                        help='Input JSON file (output of prepare_train.py).')
    parser.add_argument('--output', default='train_data.jsonl',
                        help='Output JSONL file.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for stranger-slot assignment.')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {args.input}...")
    cases = load_cases(args.input)
    print(f"  {len(cases)} cases")

    answer_counter = Counter()
    stranger_pos_counter = Counter()
    by_status = Counter()

    print(f"\nWriting {args.output}...")
    with open(args.output, 'w') as f:
        for case in cases:
            stranger_pos, answer_letter = assign_letters(case, rng)
            rec = dict(case)
            rec['stranger_letter_pos'] = stranger_pos
            rec['answer_letter'] = answer_letter
            f.write(json.dumps(rec) + '\n')

            answer_counter[answer_letter] += 1
            stranger_pos_counter[chr(ord('A') + stranger_pos)] += 1
            by_status[case.get('query_status', 'unknown_status')] += 1

    print(f"  Wrote {len(cases)} records → {args.output}")

    print("\n" + "=" * 60)
    print("Letter-assignment summary")
    print("=" * 60)

    print("\n  query_status counts:")
    for st in sorted(by_status.keys()):
        print(f"    {st}: {by_status[st]} cases")

    print("\n  Answer-letter distribution (target letter the model must output):")
    for letter in sorted(answer_counter.keys()):
        print(f"    {letter}: {answer_counter[letter]} cases")

    print("\n  Stranger-letter slot distribution (which letter is the stranger option):")
    for letter in sorted(stranger_pos_counter.keys()):
        print(f"    {letter}: {stranger_pos_counter[letter]} cases")


if __name__ == '__main__':
    main()
