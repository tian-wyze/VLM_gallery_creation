"""
Use Gemini 2.5 Pro to filter gallery images, keeping only those that show
a clear, usable view of a person for identity recognition (ReID).

Input:  household_info_v2_{cross,same}_clothes.json
          Structure: {household_id -> {identity_id -> {mac_addr -> [image_paths]}}}
Output: household_info_v2_{cross,same}_clothes_cleaned.json  (same structure)

Images are evaluated in batches; progress is checkpointed per household so
the script can be safely interrupted and resumed.
"""

import os
import json
import time
import argparse
from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
FILTER_PROMPT = """\
You are a quality-control assistant for a person re-identification dataset.
Each image below is a cropped bounding-box crop of a person captured by a home security camera.

Your job: decide whether each image is USABLE for identity recognition.

Mark an image as YES (keep) if it:
  - Shows a recognisable view of a person (full body or at least torso + head)
  - Is reasonably sharp (not heavily motion-blurred)
  - Has the person occupying a meaningful portion of the frame

Mark an image as NO (reject) if it:
  - Shows only a small body part (e.g. feet, a hand, a sliver of torso)
  - Is heavily motion-blurred to the point of being unrecognisable
  - Is very dark or overexposed so the person cannot be identified
  - The crop is essentially empty or contains no person

Respond with EXACTLY one answer per image, in order, one per line.
Use only the words YES or NO — nothing else.
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str, verbose: bool = True) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    if verbose:
        print(f"  Saved → {path}")


def evaluate_batch(client, model_name: str, image_paths: list[str]) -> list[bool]:
    """
    Send a batch of images to Gemini and return a keep/reject flag per image.
    Falls back to keeping the image (True) on any unrecoverable error so we
    never silently lose data.
    """
    images = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        try:
            images.append(Image.open(path))
            valid_indices.append(i)
        except Exception as e:
            print(f"    [warn] Cannot open {path}: {e}")

    if not images:
        return [True] * len(image_paths)  # keep by default if unreadable

    content = [FILTER_PROMPT] + images

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=content,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            raw_lines = [l.strip().upper() for l in response.text.strip().split('\n') if l.strip()]
            # Keep only lines that are YES or NO
            answers = [l for l in raw_lines if l in ('YES', 'NO')]

            # If we got fewer answers than images, pad with YES (conservative keep)
            while len(answers) < len(images):
                answers.append('YES')

            keep_flags_valid = [a == 'YES' for a in answers[:len(images)]]

            # Map back to full image_paths list (unreadable paths keep=True)
            result = [True] * len(image_paths)
            for flag, idx in zip(keep_flags_valid, valid_indices):
                result[idx] = flag
            return result

        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"    [error] API call failed (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                print(f"    Retrying in {wait}s …")
                time.sleep(wait)
            else:
                print("    Giving up — keeping all images in this batch.")
                return [True] * len(image_paths)


# --------------------------------------------------------------------------- #
# Core filtering logic
# --------------------------------------------------------------------------- #
def clean_household_dict(
    household_dict: dict,
    client,
    model_name: str,
    checkpoint_path: str,
    batch_size: int = 8,
) -> dict:
    """
    Iterate over every (identity, mac, image) triple, ask Gemini whether
    each image is usable, and return a filtered copy of household_dict.

    Checkpoints are written per household so the script can be resumed.
    Identities with no images remaining are reported at the end.
    """
    # Resume from checkpoint if available
    if os.path.exists(checkpoint_path):
        cleaned = load_json(checkpoint_path)
        print(f"  Resuming from checkpoint ({len(cleaned)} households already done).")
    else:
        cleaned = {}

    total_before = total_after = 0

    for household_id, identities in tqdm(household_dict.items(), desc="Households"):
        if household_id in cleaned:
            # Count previously processed gallery images for the summary
            for macs in cleaned[household_id].values():
                for splits in macs.values():
                    total_after += len(splits['gallery'])
            continue

        cleaned[household_id] = {}

        for identity_id, macs in identities.items():
            cleaned[household_id][identity_id] = {}

            for mac_addr, splits in macs.items():
                gallery_paths = splits['gallery']
                total_before += len(gallery_paths)

                kept = []
                for batch_start in range(0, len(gallery_paths), batch_size):
                    batch = gallery_paths[batch_start: batch_start + batch_size]
                    flags = evaluate_batch(client, model_name, batch)
                    kept.extend(p for p, ok in zip(batch, flags) if ok)

                # Query images are copied through unchanged
                cleaned[household_id][identity_id][mac_addr] = {
                    'query': splits['query'],
                    'gallery': kept,
                }
                total_after += len(kept)

        # Checkpoint after each household
        save_json(cleaned, checkpoint_path, verbose=False)

    pct = total_after / max(total_before, 1) * 100
    print(f"  Gallery images kept: {total_after} / {total_before} ({pct:.1f}%)")

    # Report identities with no gallery images remaining after filtering
    empty_identities = [
        (hh_id, id_id)
        for hh_id, identities in cleaned.items()
        for id_id, macs in identities.items()
        if all(len(splits['gallery']) == 0 for splits in macs.values())
    ]
    print(f"  Identities with 0 remaining gallery images: {len(empty_identities)}")
    for hh_id, id_id in empty_identities:
        print(f"    {id_id} (household {hh_id})")

    return cleaned


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Filter gallery images using Gemini for person ReID quality."
    )
    parser.add_argument('--project_id', type=str, default='ai-datascience-354723')
    parser.add_argument('--location',   type=str, default='us-central1')
    parser.add_argument('--model_name', type=str, default='gemini-2.5-pro')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of images per Gemini API call')
    parser.add_argument('--input_files', nargs='+',
                        default=[
                            'household_info_v2_cross_clothes.json',
                            'household_info_v2_same_clothes.json',
                        ],
                        help='Household info JSON files to process')
    args = parser.parse_args()

    client = genai.Client(
        vertexai=True,
        project=args.project_id,
        location=args.location,
    )

    for input_path in args.input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"{'='*60}")

        if not os.path.exists(input_path):
            print(f"  [skip] File not found: {input_path}")
            continue

        base, ext = os.path.splitext(input_path)
        output_path = base + '_cleaned' + ext
        checkpoint_path = base + '_checkpoint' + ext

        household_dict = load_json(input_path)

        # Summary before filtering
        n_hh  = len(household_dict)
        n_ids = sum(len(ids) for ids in household_dict.values())
        n_img = sum(len(imgs)
                    for hh in household_dict.values()
                    for ids in hh.values()
                    for imgs in ids.values())
        print(f"  Input: {n_hh} households, {n_ids} identities, {n_img} images")

        cleaned = clean_household_dict(
            household_dict,
            client=client,
            model_name=args.model_name,
            checkpoint_path=checkpoint_path,
            batch_size=args.batch_size,
        )

        save_json(cleaned, output_path)

        # Remove checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"  Checkpoint removed.")

    print("\nDone.")


if __name__ == '__main__':
    main()
