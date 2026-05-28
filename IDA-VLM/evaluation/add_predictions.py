"""Recursively prepend ``predictions_`` to every CSV file under a folder.

Walks the FOLDER constant (and every subfolder), finds every ``*.csv``, and
renames files that don't already start with ``predictions_``. Idempotent:
files that already have the prefix are left untouched. Renames happen in
place — no copy.

To retarget, just edit the FOLDER constant below.

Usage:
    python add_predictions.py
    python add_predictions.py --dry_run     # preview without renaming
"""

import argparse
import os
import sys


# Folder to walk recursively — edit this to point at a different results dir.
FOLDER = "/home/tian.liu/IDA-VLM/evaluation/results_20260428"

PREFIX = "predictions_"


def main():
    parser = argparse.ArgumentParser(
        description="Recursively prepend 'predictions_' to every CSV under FOLDER."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be renamed without actually renaming.",
    )
    args = parser.parse_args()

    if not os.path.isdir(FOLDER):
        print(f"ERROR: not a directory: {FOLDER}", file=sys.stderr)
        sys.exit(1)

    print(f"Walking: {FOLDER}")

    renamed = 0
    skipped_already_prefixed = 0
    skipped_collision = 0

    for dirpath, _, filenames in os.walk(FOLDER):
        for fname in filenames:
            if not fname.endswith(".csv"):
                continue
            if fname.startswith(PREFIX):
                skipped_already_prefixed += 1
                continue

            src = os.path.join(dirpath, fname)
            dst = os.path.join(dirpath, PREFIX + fname)

            # Don't clobber an existing predictions_<name>.csv if one happens
            # to already be there (e.g. from an interrupted earlier run).
            if os.path.exists(dst):
                print(f"SKIP (collision): {src}  →  {dst} already exists",
                      file=sys.stderr)
                skipped_collision += 1
                continue

            if args.dry_run:
                print(f"[dry-run] would rename: {src}  →  {dst}")
            else:
                os.rename(src, dst)
                print(f"renamed: {src}  →  {dst}")
            renamed += 1

    summary = (
        f"\nDone. {renamed} renamed"
        f"{' (dry-run)' if args.dry_run else ''}, "
        f"{skipped_already_prefixed} already prefixed, "
        f"{skipped_collision} skipped due to name collision."
    )
    print(summary)


if __name__ == "__main__":
    main()
