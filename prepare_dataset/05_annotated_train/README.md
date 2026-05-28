# 05 — Annotated Identity Corpus

Parse the curated, human-annotated identity image set into a household-indexed dictionary that downstream training can sample from.

## Input

A flat folder of `.jpg` files at:

```
/home/tian.liu/tian_data/wyze_person_v2/annotated_identities/
```

Each filename encodes the household, identity, camera, event, frame, and clothes ID:

```
<user_id>_<cluster_id>_<device_mac>_<event_id>_<frame_id>_<clothes_id>.jpg
```

Note that `<user_id>` itself can contain underscores — it's a multi-part field whose number of underscore-separated tokens varies (one or two in practice). The parsing logic anchors from the **right** of the filename to handle this, since the trailing four fields each occupy exactly one underscore-separated token.

### Example

```
10001004_14411366_0_D03F2722E563_D03F2722E563131684184869_000027_000.jpg
```

| Field         | Value                              | Derivation (parsing right → left)                         |
|---------------|------------------------------------|-----------------------------------------------------------|
| `clothes_id`  | `000`                              | last token (drop the `.jpg` suffix)                       |
| `frame_id`    | `000027`                           | second-to-last token                                      |
| `event_id`    | `D03F2722E563131684184869`         | third-to-last token                                       |
| `device_mac`  | `D03F2722E563`                     | fourth-to-last token                                      |
| `cluster_id`  | `0`                                | what's left of the head, last token                       |
| `user_id`     | `10001004_14411366`                | everything left of `cluster_id` (1+ tokens)               |
| `identity_id` | `10001004_14411366_0`              | `user_id` + `_` + `cluster_id` (everything before mac)    |
| `household_id`| `10001004_14411366`                | `identity_id` minus its last token (= `user_id` here)     |

The four parsing helpers in [`get_annotated_info.py`](get_annotated_info.py) wrap that logic:

- `parse_id(fname)` → `identity_id` (e.g. `10001004_14411366_0`)
- `parse_household_id(identity_id)` → `household_id` (e.g. `10001004_14411366`)
- `parse_mac_addr(fname)` → `device_mac`
- `parse_event_id(fname)` → `event_id`

## Usage

```bash
cd IDA-VLM/prepare_dataset/05_annotated_train
python get_annotated_info.py
```

The script:

1. Lists every entry in `annotated_identities/` and filters to `.jpg`. Non-jpg entries (e.g. `selection_report.txt`) are skipped with a warning.
2. Parses each filename and populates a 3-level dict:
   ```
   res[household_id][identity_id][mac_addr] = [<absolute path>, ...]
   ```
3. Serializes the unfiltered dict to `annotated_household_info.json` (~80 MB, 527k paths).
4. Prints corpus-level and family-size statistics.
5. **Filters out households that overlap with the `04_*` eval splits** (see "Filtering" below) and writes `annotated_household_info_filtered.json`.

## Output

Two JSON files, both with the same nested structure:

```
annotated_household_info.json            # unfiltered (every parsed household)
annotated_household_info_filtered.json   # household-disjoint with 04_* eval — use this for training
```

```json
{
  "<household_id>": {
    "<identity_id>": {
      "<mac_addr>": ["<abs_path>", "<abs_path>", ...]
    }
  }
}
```

Note this format **omits the `query` / `gallery` sub-keys** used by [04_varying_gallery_length_distractors/](../04_varying_gallery_length_distractors/) — the curated set is meant to be sampled fresh by downstream consumers, not pre-split into eval roles.

## Filtering: removing households that overlap with `04_*` eval

Why: the [04_varying_gallery_length_distractors/](../04_varying_gallery_length_distractors/) JSONs are our held-out evaluation set. Any `household_id` that appears in those eval splits must be removed from this training corpus, or the train/eval boundary leaks at the household level — even when the specific identity (cluster_id) within that household differs, "stranger to family X" cases would silently include people *from* family X, contaminating distractor metrics. This matches the household-disjoint rule that `split_train_test_households` already enforces *inside* `04_*`.

What the filter does:

1. Loads both [household_info_v2_same_clothes.json](../04_varying_gallery_length_distractors/household_info_v2_same_clothes.json) and [household_info_v2_cross_clothes.json](../04_varying_gallery_length_distractors/household_info_v2_cross_clothes.json).
2. Takes the union of their top-level `household_id` keys → 416 eval-side households.
3. Drops every annotated household whose ID is in that set.
4. Writes the result to `annotated_household_info_filtered.json`.

### Filtering result

| Metric | Before filter | After filter | Δ |
|---|---:|---:|---:|
| Households | 13,177 | **13,114** | −63 |
| Identities | 22,598 | **22,474** | −124 |
| Loss (% of original) | — | — | households 0.48%, identities 0.55% |

The 63 dropped households break down as:

| Source of overlap | Households |
|---|---:|
| Annotated ∩ same_clothes | 39 |
| Annotated ∩ cross_clothes | 61 |
| Annotated ∩ (same ∪ cross) | **63** |
| …of which actually share an identity (not just a household_id) | 11 (13 identities) |

Most overlapping households share only the `user_id` (the household-account ID), but contain different family members in the two corpora — same `user_id`, different `cluster_id`. We drop them anyway because the household-disjoint rule is the safer cut.

Verified post-hoc: `set(filtered_households) ∩ set(04_combined_households) == ∅`.

## Corpus statistics

Run on **2026-04-27** against `annotated_identities/` (527,820 entries, of which 1 is `selection_report.txt` and 527,819 are images).

### Top-level counts

| Metric         | Unfiltered | Filtered (recommended) |
|----------------|-----------:|-----------------------:|
| `.jpg` files   | 527,819    | (slightly less — count it from the filtered JSON if needed) |
| Households     | **13,177** | **13,114** |
| Identities     | **22,598** | **22,474** |
| Unique events  | 89,585     | (similar order)        |

Average density (unfiltered):

- ~1.7 identities per household
- ~23 images per identity
- ~5.9 images per (identity × event)

## Corpus statistics

Run on **2026-04-27** against `annotated_identities/` (527,820 entries, of which 1 is `selection_report.txt` and 527,819 are images).

### Top-level counts

| Metric         | Value     |
|----------------|-----------|
| `.jpg` files   | 527,819   |
| Households     | **13,177** |
| Identities     | **22,598** |
| Unique events  | 89,585    |

Average density:

- ~1.7 identities per household
- ~23 images per identity
- ~5.9 images per (identity × event)

### Family size distribution

Number of identities per household — i.e. the gallery length you'd get if you used every household member at evaluation time. Unfiltered vs. filtered (the filter drops households that overlap with `04_*`):

| Family size | Unfiltered | Filtered | Δ |
|------------:|-----------:|---------:|--:|
| 1 (singleton) | 7,044 | 7,020 | −24 |
| 2 | 3,816 | 3,793 | −23 |
| 3 | 1,596 | 1,585 | −11 |
| 4 | 527   | 523   | −4  |
| 5 | 148   | 147   | −1  |
| 6 | 39    | 39    | 0   |
| 7 | 5     | 5     | 0   |
| 8 | 1     | 1     | 0   |
| 9 | 1     | 1     | 0   |
| **Total** | **13,177** | **13,114** | **−63** |

The drops are spread roughly proportionally across the small-family sizes (1–5) — no concentration on any single size, so filtering doesn't bias the family-size distribution.

Cumulative coverage of the filtered corpus:

| Family size | % households | Cumulative % |
|------------:|-------------:|-------------:|
| 1 (singleton) | 53.5% | 53.5% |
| 2 | 28.9% | 82.4% |
| 3 | 12.1% | 94.5% |
| 4 | 4.0%  | 98.5% |
| 5 | 1.1%  | 99.6% |
| 6 | 0.3%  | 99.9% |
| 7+ | 0.05% | 100.0% |

Practical implications:

- **Sizes 1–5 cover 99.6% of households** — matches the gallery-size cap (≤5) used by [04_varying_gallery_length_distractors/](../04_varying_gallery_length_distractors/), so very little data is lost if you carry that cap forward.
- **Singletons are the majority (53.5%)** — singleton-only training would severely under-sample the harder family-disambiguation cases. Stratified sampling by family size is recommended.
- **Long tail (sizes 6–9, 46 households total)** — exclude or cap depending on whether your training pipeline supports galleries > 5.

## Building the training set: `prepare_train.py`

Consumes `annotated_household_info_filtered.json` and writes `train_data.json` in the same case-level schema as [04_varying_gallery_length_distractors/](../04_varying_gallery_length_distractors/) (`{query, gallery, label, similarity, household_id, identity_id, query_household_id, query_identity_id}`). Reuses 04's `HardNegativeMiner`, `build_nondistr_case`, and distractor builders verbatim via direct import — only the input adapter, sampling rates, and the dropping of the same/cross-clothes axis differ.

### Usage

```bash
cd IDA-VLM/prepare_dataset/05_annotated_train
python prepare_train.py
```

Defaults: `--max_gallery 6`, `--distractor_fraction 0.2`, `--seed 42`. DINOv2 ViT-L/14 embeddings are cached to `cached_embeddings/dinov2_embeddings.pt` (~2.1 GB) on first run; subsequent runs reuse the cache.

### Differences from 04

- **No train/test split** — the input JSON is already eval-disjoint at the household level (see filtering above), so the entire corpus is training data.
- **No clothes axis** — every annotated image has `clothes_id = 000`. Cross-clothes signal comes implicitly from cross-camera sampling (the same person under a different camera on a different day usually wears different clothes).
- **Reduced per-identity sampling rates** — corpus is ~10× larger than 04. Cases per identity:

  | family size | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | rate | 1 | 2 | 2 | 3 | 4 | 5 | 6 | 6 | 6 |

  Singletons (53.5% of households) get the lightest weight; rare large families get more cases per identity so their coverage doesn't collapse.

- **Camera randomized per case** — `random.choice(['same', 'cross'])` like 04. Identities lacking the picked scenario yield None and the loop retries.

### Output: `train_data.json`

Run on **2026-04-27** with the defaults above:

| | size 1 | size 2 | size 3 | size 4 | size 5 | size 6 | size 7 | size 8 | size 9 | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same-cam      | 5,020 | 8,352 | 4,129 | 2,320 |   913 |   400 | 102 |   0 |  45 | 21,281 |
| cross-cam     | 2,000 | 5,232 | 3,591 | 2,222 | 1,135 |   340 |  60 |  30 |   9 | 14,619 |
| distractor    |   797 |   797 |   798 |   798 |   798 |   798 | 798 | 798 | 798 |  7,180 |
| **total**     | **7,817** | **14,381** | **8,518** | **5,340** | **2,846** | **1,538** | **960** | **828** | **852** | **43,080** |

Gallery-length distribution (post hard-negative padding, `max_gallery = 6`):

| gallery size | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cases | 1,378 | 4,110 | 6,196 | 8,126 | 9,613 | 11,017 | 960 | 828 | 852 |
| %     | 3.2%  | 9.5%  | 14.4% | 18.9% | 22.3% | 25.6%  | 2.2% | 1.9% | 2.0% |

Sizes 7–9 appear only when the family already exceeds the padding cap (47 households total — see family-size table above).

### Caveats / known imbalances

1. **Camera bias** — same-cam ends up at 59% of non-distractor cases instead of 50%. Cause: 45% of identities have only one MAC, so cross-camera attempts for those identities fail and the retry loop falls back to same-camera. Defensible as-is (mirrors the natural distribution of the corpus), but if a strict 50/50 split is needed, modify the camera-selection loop in `build_training_cases` to track failed `cross` attempts and force the missing mode on retries.
2. **Distractor over-repetition for rare family sizes** — distractors are uniformly stratified across 9 size buckets (≈798 cases each), but sizes 7/8/9 collectively have only **7 households**. Each of those households appears in ~100 distractor cases. If diversity matters more than uniform per-size coverage, weight buckets by household count instead.
3. **`samecam_size8 = 0`** — the single size-8 household has no shared MAC, so all of its cases are cross-camera (consistent with the camera-fallback rule).

### Files produced

```
prepare_train.py              # this script
train_data.json               # ~60 MB, 43,080 cases (the actual training data)
cached_embeddings/
  dinov2_embeddings.pt        # ~2.1 GB, 524,811 ViT-L/14 embeddings
```

---

# Test-set preparation — `prepare_test.py`

Now that training has moved to the curated `annotated_identities/` corpus, the **entire** v2 dataset (both `same_clothes` and `cross_clothes`) is repurposed as the test set. There is no longer a need to hold any v2 households out for training, so `prepare_test.py` consumes the full v2 household JSONs and produces 10 benchmark JSONs schema-compatible with the existing `04_*` benchmarks.

## Inputs

The two v2 household JSONs (built by 04 from the cleaned v2 corpus):

```
../04_varying_gallery_length_distractors/household_info_v2_same_clothes.json   # 313 households, 413 identities
../04_varying_gallery_length_distractors/household_info_v2_cross_clothes.json  # 404 households, 626 identities
```

## Design

`prepare_test.py` is **self-contained** — every helper function it needs (`load_json`, `save_json`, `find_valid_triples`, `build_nondistr_case`, `_build_singleton_distractor`, `_build_family_distractor`, `collect_stranger_pool`, `add_household_ids`, `build_image_metadata`, `merge_query_gallery_pools`, `collect_all_image_paths`, `compute_dinov2_embeddings`, `load_or_compute_embeddings`, `HardNegativeMiner`, `ImagePathDataset`) is copied verbatim from `04_*/varying_gallery.py`. No cross-folder imports.

Three differences from 04's logic:

1. **No train/test split.** Every household in the v2 corpora is a test candidate. `find_valid_triples(...)` is called with `list(hh_dict.keys())`.
2. **No resampling.** One case per valid `(household_id, identity_id, query_mac)` triple for non-distractor scenarios; one case per qualifying household for distractor scenarios. Test size is determined by the natural distribution of the corpus, not a per-identity quota.
3. **Hard negatives off by default** (`--pad_hard_negatives False`). Galleries contain exactly the household member(s) — `gallery_size = family_size` for non-distractor scenarios, `gallery_size = family_size` for family-distractor, and `gallery_size ∈ [1, max_gallery]` (multi-photo enrollment of the lone resident) for singleton-distractor.
4. **Non-distractor splits are resampled to ~150 cases** (`--min_cases 150`). Several splits have only 14–75 valid `(household, identity, query_mac)` triples — far too few for a stable accuracy estimate. After the initial pass that builds one case per triple, `build_nondistr_subset` redraws triples uniformly at random and rebuilds; `build_nondistr_case`'s internal randomness (random query image, random per-identity gallery image, random shuffle for label position) yields a fresh case each call. Splits that already exceed 150 cases are left untouched. Distractor splits are not resampled (both already exceed 150).

The 10-scenario structure mirrors 04 exactly: 8 non-distractor (`{sameclothes, crossclothes} × {singleton, family} × {samecamera, crosscamera}`) + 2 distractor (`distractor_cropped_{singleton, family}`).

## Usage

```bash
cd IDA-VLM/prepare_dataset/05_annotated_train

# Realistic benchmarks (default — gallery = household members only)
python prepare_test.py

# Stress-test variant (gallery padded with DINOv2-mined hard negatives, like 04)
python prepare_test.py --pad_hard_negatives True --max_gallery 5
```

Output: `benchmarks/<scenario>.json` × 10. Schema is identical to 04's:

```json
{
  "scenario": "...",
  "n_identities": <int>,
  "n_cases": <int>,
  "gallery_size_distribution": {<size>: <count>, ...},
  "eval_id_count": {<identity_id>: <case_count>, ...},
  "eval_cases": [{"query": ..., "gallery": [...], "label": ..., "query_household_id": ..., ...}, ...]
}
```

The only schema difference: per-case `similarity` field is **not** computed (saves a DINOv2 cosine-product per case; metadata-only, not used by the eval loop). To use these benchmarks in `run_test.sh`:

```bash
TEST_FOLDER="../prepare_dataset/05_annotated_train/benchmarks"
```

(File names are identical to 04's, so no other changes needed.)

## Per-split statistics (default mode, `--min_cases 150`, no hard negatives, seed=42)

Built against the full v2 household JSONs. The "raw triples" column shows the count *before* resampling — splits with `raw_triples < 150` were resampled with replacement up to 150; splits with `raw_triples ≥ 150` were left as-is.

### 8 non-distractor scenarios (1,270 total cases)

| Scenario | n_cases | raw_triples | n_identities | Gallery sizes | Resampled? |
|---|---:|---:|---:|---|:-:|
| `cropped_sameclothes_singleton_samecamera`  | 152 | 152 | 152 | `{1: 152}` | — |
| `cropped_sameclothes_singleton_crosscamera` | 150 |  75 |  75 | `{1: 150}` | ✓ (×2.0) |
| `cropped_sameclothes_family_samecamera`     | 150 |  19 |  19 | `{2: 150}` | ✓ (×7.9) |
| `cropped_sameclothes_family_crosscamera`    | 150 |  61 |  61 | `{2: 119, 3: 31}` | ✓ (×2.5) |
| `cropped_crossclothes_singleton_samecamera` | 150 |  74 |  74 | `{1: 150}` | ✓ (×2.0) |
| `cropped_crossclothes_singleton_crosscamera`| 161 | 161 | 161 | `{1: 161}` | — |
| `cropped_crossclothes_family_samecamera`    | 150 |  14 |  14 | `{2: 131, 3: 19}` | ✓ (×10.7) |
| `cropped_crossclothes_family_crosscamera`   | 207 | 207 | 207 | `{2: 147, 3: 39, 4: 17, 5: 1, 6: 3}` | — |
| **Subtotal** | **1,270** | **763** | — | — | 5/8 splits |

The gallery-size distribution is preserved by resampling (e.g. `cropped_sameclothes_family_crosscamera` was `{2: 49, 3: 12}` → `{2: 119, 3: 31}` — both ≈80%/20%), since resampling redraws from the same triple set.

### 2 distractor scenarios (717 total cases, `label = -1`)

| Scenario | n_cases | n_households | Gallery sizes |
|---|---:|---:|---|
| `distractor_cropped_singleton` | 462 | 322 | `{1: 93, 2: 90, 3: 91, 4: 94, 5: 94}` |
| `distractor_cropped_family`    | 255 | 183 | `{2: 203, 3: 41, 4: 8, 5: 2, 6: 1}` |
| **Subtotal** | **717** | — | — |

Distractor splits already exceed 150 cases, so resampling is skipped.

### Grand total: **1,987 cases across 10 benchmarks**

(For comparison, 04's benchmarks had 1,600 cases under the previous train/test split + per-identity sampling. To disable resampling and recover the natural-distribution counts, run with `--min_cases 0`.)

## Notes / known properties

1. **Singleton non-distractor scenarios are degenerate** — gallery always has exactly 1 image (the lone resident), so `label` is always `1`. A trivial baseline ("always answer 1") gets 100%. Useful only when paired with `distractor_cropped_singleton` (same gallery, stranger query, label = -1) to evaluate match-vs-reject calibration.
2. **Singleton-distractor galleries vary 1–5 in size** because `_build_singleton_distractor` shows `random.randint(1, max_gallery)` photos of the *same* lone resident. Realistic for multi-photo enrollment in production. To force `gallery_size = 1` here for symmetry with the non-distractor singleton case, pass `--max_gallery 1`.
3. **Family-distractor galleries match family size** (no padding). Same household composition as the non-distractor family scenarios; only the query identity differs.
4. **A household contributes one case per clothes corpus it appears in.** That's why `distractor_cropped_singleton.n_cases (462) > distractor_cropped_singleton.n_households (322)` — 140 households appear in both `same_clothes` and `cross_clothes`. Each contributes a separate case (different gallery composition because each clothes JSON has different image splits).
5. **Resampled cases are not exact duplicates.** `build_nondistr_case` makes three independent random choices per call (which query image, which gallery image per identity, the shuffle that determines label position), so even when the underlying triple set is small, repeated calls produce different `(query, gallery, label)` tuples. For very small triple sets (e.g. the 14-triple `cropped_crossclothes_family_samecamera`, resampled 10.7×), some near-duplicates are inevitable, but the label-position randomization alone doubles or triples the effective combinatorial space, and the variance reduction from larger N still outweighs the residual correlation between resampled cases.
6. **`n_identities` reflects the raw triple set, not the resampled case count.** A resampled split shows `n_cases > n_identities` because each identity's triple is drawn multiple times.

## Stress-test variant

Run with `--pad_hard_negatives True` to get galleries padded to `max_gallery=5` with DINOv2-mined people from other households (the 04 setup, applied to the full v2 corpus). This evaluates the model's fine-grained ID discrimination but is **not** representative of production gallery composition. Reuses cached embeddings under `cached_embeddings/`; first run computes them.

## Files produced

```
prepare_test.py                         # this script (self-contained, ~500 lines)
benchmarks/
  cropped_sameclothes_singleton_samecamera.json
  cropped_sameclothes_singleton_crosscamera.json
  cropped_sameclothes_family_samecamera.json
  cropped_sameclothes_family_crosscamera.json
  cropped_crossclothes_singleton_samecamera.json
  cropped_crossclothes_singleton_crosscamera.json
  cropped_crossclothes_family_samecamera.json
  cropped_crossclothes_family_crosscamera.json
  distractor_cropped_singleton.json
  distractor_cropped_family.json
```

---

## Relationship to the rest of the pipeline

```
01_similarity-based/          ─→ initial similarity-based clustering
02_household-based/           ─→ household-level grouping
03_clean_gallery_images/      ─→ deduplication / quality filtering
04_varying_gallery_length_distractors/  ─→ variable-N gallery + hard negatives (legacy eval benchmarks; v2 household JSONs reused here as test source)
05_annotated_train/  ◀── you are here
```

This folder consumes a **manually-curated** identity corpus (richer + cleaner than the auto-clustered datasets in 01–04). `prepare_train.py` turns the parsed corpus into trainable cases (`train_data.json`); `prepare_test.py` reuses the v2 household JSONs from 04 to build 10 evaluation benchmarks (now that training has moved off the v2 data, every v2 household is available for testing).
