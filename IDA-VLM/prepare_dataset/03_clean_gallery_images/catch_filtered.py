"""
Compute the set of gallery images removed by clean_with_gemini.py.

For each pair of (original, cleaned) household info JSON files, this script
produces a *_filtered.json file containing only the removed gallery images,
in the same nested structure:

  {household_id -> {identity_id -> {mac_addr -> {'query': [], 'gallery': [removed]}}}}

Query images are never filtered, so 'query' lists are always empty in the output.
Leaves where no gallery images were removed are omitted from the output.

Usage:
    python catch_filtered.py [--pairs ORIG CLEANED [ORIG CLEANED ...]]

If --pairs is not given, the default pairs are used (cross and same clothes).
Missing cleaned files are skipped with a warning.
"""

import os
import json
import argparse


DEFAULT_PAIRS = [
    (
        'household_info_v2_cross_clothes.json',
        'household_info_v2_cross_clothes_cleaned.json',
    ),
    (
        'household_info_v2_same_clothes.json',
        'household_info_v2_same_clothes_cleaned.json',
    ),
]


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Saved → {path}")


def compute_filtered(original: dict, cleaned: dict) -> tuple[dict, int]:
    """
    Return (filtered_dict, total_removed) where filtered_dict has the same
    nested structure as original but contains only removed gallery images.
    Query images are not filtered, so 'query' lists are always empty in the output.
    """
    filtered = {}
    total_removed = 0

    for hh_id, identities in original.items():
        for id_id, macs in identities.items():
            for mac_addr, splits in macs.items():
                kept_set = set(
                    cleaned.get(hh_id, {})
                           .get(id_id, {})
                           .get(mac_addr, {})
                           .get('gallery', [])
                )
                removed = [p for p in splits['gallery'] if p not in kept_set]
                if not removed:
                    continue

                total_removed += len(removed)
                filtered.setdefault(hh_id, {}) \
                        .setdefault(id_id, {})[mac_addr] = {'query': [], 'gallery': removed}

    return filtered, total_removed


def process_pair(orig_path: str, cleaned_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"Original : {orig_path}")
    print(f"Cleaned  : {cleaned_path}")
    print(f"{'='*60}")

    if not os.path.exists(orig_path):
        print(f"  [skip] Original file not found: {orig_path}")
        return

    if not os.path.exists(cleaned_path):
        print(f"  [skip] Cleaned file not found yet: {cleaned_path}")
        return

    original = load_json(orig_path)
    cleaned  = load_json(cleaned_path)

    n_orig = sum(len(splits['gallery'])
                 for hh in original.values()
                 for ids in hh.values()
                 for splits in ids.values())
    n_cleaned = sum(len(splits['gallery'])
                    for hh in cleaned.values()
                    for ids in hh.values()
                    for splits in ids.values())

    filtered, total_removed = compute_filtered(original, cleaned)

    print(f"  Original images : {n_orig}")
    print(f"  Kept images     : {n_cleaned}")
    print(f"  Removed images  : {total_removed} ({total_removed / max(n_orig, 1) * 100:.1f}%)")
    print(f"  Affected households  : {len(filtered)}")
    affected_ids = sum(len(ids) for ids in filtered.values())
    print(f"  Affected identities  : {affected_ids}")

    base, ext = os.path.splitext(orig_path)
    out_path = base + '_filtered' + ext
    save_json(filtered, out_path)


def main():
    parser = argparse.ArgumentParser(
        description='Extract images removed by clean_with_gemini.py.'
    )
    parser.add_argument(
        '--pairs', nargs='+', metavar='PATH',
        help='Alternating list of ORIGINAL CLEANED file pairs, e.g. '
             '--pairs orig1.json cleaned1.json orig2.json cleaned2.json'
    )
    args = parser.parse_args()

    if args.pairs:
        if len(args.pairs) % 2 != 0:
            parser.error('--pairs requires an even number of arguments (orig/cleaned pairs).')
        pairs = [(args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)]
    else:
        pairs = DEFAULT_PAIRS

    for orig_path, cleaned_path in pairs:
        process_pair(orig_path, cleaned_path)

    print("\nDone.")


if __name__ == '__main__':
    main()
