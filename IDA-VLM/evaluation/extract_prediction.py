"""Post-hoc fill the `prediction` column in eval CSVs from the `response` text.

Given a folder of CSVs produced by eval_gemini.py (columns:
``idx,label,prediction,response,query``), this script loops over every CSV and:

  1. Re-extracts an integer prediction from the `response` text using
     ``eval_gemini.extract_prediction``.
  2. If the existing `prediction` cell is empty, fills it with the extracted value.
  3. If the existing `prediction` is non-empty and disagrees with the extracted
     one, raises ``ValueError`` and prints the filename + idx (so you can
     investigate the row).
  4. Overwrites the CSV in place with the filled predictions.
  5. Computes and prints per-file accuracy (label vs prediction) and moves on.

Usage:
    python extract_prediction.py <folder>

Example:
    python extract_prediction.py \\
        /home/tian.liu/IDA-VLM/evaluation/results/distractors_gemini-2.5-pro
"""

import csv
import glob
import os
import sys

from eval_gemini import extract_prediction


# Folder containing evaluation CSV files — edit this to point at a different run.
FOLDER = '/home/tian.liu/IDA-VLM/evaluation/results/distractors_gemini-2.5-pro'


def _parse_int(s):
    """Parse a CSV cell as int. Returns None if the cell is empty/unparsable."""
    if s is None:
        return None
    s = s.strip()
    if s == '':
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def process_file(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    filled = 0
    for row in rows:
        response = row.get('response') or ''
        extracted = extract_prediction(response) if response else None
        existing = _parse_int(row.get('prediction', ''))

        if existing is None:
            if extracted is None:
                print(
                    f"[WARN] file={csv_path} idx={row.get('idx')}: "
                    f"cannot parse response, defaulting prediction to -1",
                    file=sys.stderr,
                )
                extracted = -1
            row['prediction'] = str(extracted)
            filled += 1
        else:
            if extracted is not None and existing != extracted:
                print(f"[ERROR] file={csv_path} idx={row.get('idx')}: existing={existing} vs extracted={extracted}", file=sys.stderr)
                raise ValueError(
                    f"Prediction mismatch in {csv_path} at idx={row.get('idx')}: "
                    f"existing={existing} vs extracted={extracted}; "
                    f"response={response!r}"
                )

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    correct = 0
    total = 0
    for r in rows:
        label = _parse_int(r.get('label', ''))
        pred = _parse_int(r.get('prediction', ''))
        if label is None or pred is None:
            continue
        total += 1
        if label == pred:
            correct += 1
    accuracy = 100.0 * correct / total if total else 0.0

    print(
        f"{os.path.basename(csv_path)}  "
        f"(filled {filled}/{len(rows)})  "
        f"accuracy={accuracy:.1f}% ({correct}/{total})"
    )


def main():
    csv_paths = sorted(glob.glob(os.path.join(FOLDER, '*.csv')))
    if not csv_paths:
        print(f"No CSV files found in {FOLDER}", file=sys.stderr)
        sys.exit(1)

    for csv_path in csv_paths:
        process_file(csv_path)


if __name__ == '__main__':
    main()
