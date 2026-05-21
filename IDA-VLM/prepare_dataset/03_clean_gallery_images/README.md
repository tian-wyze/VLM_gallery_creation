# Gallery Cleaning

This folder contains scripts for building and filtering gallery images used in the person re-identification (ReID) dataset.

## Files

| File | Description |
|------|-------------|
| `clean_gallery.py` | Builds household info JSON files from the raw dataset and prepares train/test splits. |
| `clean_with_gemini.py` | Filters gallery images using Gemini, keeping only images with a clear view of the person. |
| `catch_filtered.py` | Computes the diff between original and cleaned JSONs to extract the removed images. |
| `household_info_v2_cross_clothes.json` | Household info for the cross-clothes subset (input to `clean_with_gemini.py`). |
| `household_info_v2_same_clothes.json` | Household info for the same-clothes subset (input to `clean_with_gemini.py`). |

## JSON Structure

Both household info files share the same nested structure:

```json
{
  "<household_id>": {
    "<identity_id>": {
      "<mac_addr>": ["<image_path>", ...]
    }
  }
}
```

## `clean_with_gemini.py`

### Purpose

Gallery images from home security cameras are often noisy — they may show only partial body parts, be motion-blurred, or be too dark to be useful for identity recognition. This script uses Gemini to evaluate each gallery image and filter out low-quality ones, producing a cleaned household info JSON for use in downstream ReID experiments.

Images are rejected if they:
- Show only a small body part (feet, a hand, a sliver of torso)
- Are heavily motion-blurred
- Are too dark or overexposed
- Contain no recognisable person

### Usage

```bash
cd IDA-VLM/prepare_dataset/clean_gallery

# Process both JSON files with default settings (gemini-2.5-pro, batch size 8)
python clean_with_gemini.py

# Use a different model or batch size
python clean_with_gemini.py --model_name gemini-2.5-flash --batch_size 10

# Process a single file
python clean_with_gemini.py --input_files household_info_v2_cross_clothes.json
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--project_id` | `ai-datascience-354723` | GCP project ID for Vertex AI |
| `--location` | `us-central1` | GCP region |
| `--model_name` | `gemini-2.5-pro` | Gemini model to use |
| `--batch_size` | `8` | Number of images per API call |
| `--input_files` | both JSON files | One or more household info JSON files to process |

### Output

For each input file, the script produces a `_cleaned.json` file with the same nested structure, containing only the images Gemini judged as usable:

```
household_info_v2_cross_clothes_cleaned.json
household_info_v2_same_clothes_cleaned.json
```

At the end of each file's processing, the script prints how many images were kept and lists any identities with zero remaining gallery images.

### Resuming

A checkpoint file (`_checkpoint.json`) is written after each household. If the script is interrupted, re-running it will skip already-processed households and continue from where it left off. The checkpoint is deleted automatically on successful completion.

---

## `catch_filtered.py`

### Purpose

Computes the diff between an original household info JSON and its cleaned counterpart to produce a `_filtered.json` file containing only the images that Gemini rejected. Useful for auditing the filtering results or inspecting what was removed.

Missing cleaned files are skipped with a warning, so the script can be run while `clean_with_gemini.py` is still processing the same-clothes subset.

### Usage

```bash
cd IDA-VLM/prepare_dataset/clean_gallery

# Process both default pairs (skips any missing cleaned file)
python catch_filtered.py

# Process a custom pair
python catch_filtered.py --pairs household_info_v2_cross_clothes.json \
                                  household_info_v2_cross_clothes_cleaned.json
```

### Output

For each pair where the cleaned file exists, a `_filtered.json` is written:

```
household_info_v2_cross_clothes_filtered.json
household_info_v2_same_clothes_filtered.json   # once cleaned file is available
```

The output has the same nested structure as the input JSONs. The script also prints the number of removed images, the percentage removed, and how many households and identities were affected.
