# `expert_qformer` — Q-Former-Style Learnable-Query Fuser

A standalone design document for the `expert_qformer` input mode. Covers motivation, architecture, the precise role of the learnable queries / K / V, what the module output means, implementation choices, and backward compatibility.

Companion to [cross_attention.md](cross_attention.md) — both fusers plug into the same `input_mode` switch and share the `model.visual.expert_fuser` attribute, so the plumbing is identical; the math differs.

The implementation lives in:
- [utils/model_utils.py](utils/model_utils.py) — `ExpertQFormer` class + conditional instantiation.
- [replace.py](replace.py) — the shared `expert_cross_attn / expert_qformer` branch in `My_Qwen2_5_VLModel_forward`.
- [utils/collate_utils.py](utils/collate_utils.py) / [test.py](test.py) / [inference.py](inference.py) — thread `images_per_sample` through (already done for cross-attn; reused here).

---

## 1. Motivation — why go beyond cross-attention + null slot

After training `expert_cross_attn` with the null slot, strangers improved marginally but far less than expected. Diagnosis:

1. **Patches are the wrong level of abstraction for identity matching.** ViT patches carry pixel-level visual content — pose, clothing, background, lighting — and most of that is irrelevant to identity. Forcing each of ~128 patches per image to independently decide "which expert do I resemble" is a clumsy fit. Most patches end up routed to null (correctly), which just wastes compute; the identity-carrying patches each have to rediscover the same matching decision from scratch.

2. **Single-layer fusion has to do matching *and* injection in one shot.** `Wq`/`Wk`/`Wv`/`Wo` together must learn (a) what matching looks like and (b) what to write back per patch. Limited capacity for the hard part.

3. **Null slot is one-dimensional.** A single learnable `[D]` abstain direction. Hard to specialize if the stranger vs match decision requires multiple aspects (face visibility, pose coverage, background dissimilarity, etc.).

Q-Former addresses all three.

---

## 2. What Q-Former does — the core idea

Introduced in **BLIP-2 (Li et al. 2023)**, Q-Former introduces **K learnable query tokens** — trained parameters, not derived from inputs — that cross-attend to external features and produce K "digest" tokens. The downstream model consumes the digests instead of (or in addition to) the raw features.

In our adaptation, the structure is a two-stage cross-attention:

```
Stage 1 (sample-global):   K learnable queries  ↔  per-sample {null, exp_0, ..., exp_N}   →  K digest tokens
Stage 2 (per-patch):       image patch tokens   ↔  K digests of their sample              →  residual add
```

Stage 1 runs **once per sample**, independent of patch count. Stage 2 routes the compact digests to each patch cheaply.

### Why this should help our benchmarks

1. **Information bottleneck.** K ≪ (N+2) slots forces the model to compress the expert set into K meaningful summaries. Under training loss, the K queries *specialize*: one query may become a best-match extractor, another a stranger detector (attending heavily to null), another a query-vs-gallery similarity probe. Cross-attn had no such specialization axis — every patch independently re-did the same matching.

2. **Matching happens once per sample.** Stage 1 runs K × (N+2) attention ops per sample — cheap and independent of ViT patch count. Stage 2 is a simple routing over K < N+2 slots per patch. Cross-attn did sumT × (N+2) matches (most of which were "patch is background, go to null"), which is wasteful.

3. **Patches don't individually detect strangers.** By the time the K digests reach Stage 2, Stage 1 has already encoded the sample-global "is this a stranger?" decision into one or more of the K slots. Each patch just consumes the digest that matches its content type — much easier gradient path than "each patch learns to independently detect stranger."

4. **More dedicated capacity for the hard part.** Stage 1 has ~3·D² params and does the heavy lifting (matching over the expert set). Stage 2 has ~4·D² and routes. Cross-attn had all ~4·D² trying to do both at once.

5. **Interpretable.** You can dump `stage1_attn[b, :, :, :]` per sample and literally see *which learnable queries attend to which experts* — useful for diagnosing why stranger accuracy is or isn't improving.

---

## 3. Architecture — Q, K, V at each stage

### Stage 1: learnable queries attend to the expert set

```python
queries        = self.queries  .unsqueeze(0).expand(B, -1, -1)   # [B, K, D]  ← learnable parameters, shared across samples

q1 = Wq1(LN(queries))                                              # [B, K, D]
k1 = Wk1(LN(kv))                                                   # [B, M, D]   kv = [null_slot, exp_0, ..., exp_n-1]
v1 = Wv1(LN(kv))                                                   # [B, M, D]

# multi-head split → [B, K, H, d] / [B, M, H, d]
scores1 = Q1·K1^T / √d          # [B, K, H, M]
attn1   = softmax(scores1, dim=-1)
digest  = Σ attn1·V1 + queries  # [B, K, D]   ← residual with queries so Stage-1 can start as identity
```

**What Q/K/V are here:**
- **Q = `self.queries`** — a trained `[K, D]` parameter tensor, **shared across every sample in the batch**. Each row is one "aspect" the fuser asks about.
- **K / V = null_slot + per-sample expert descriptors** — same construction as [cross_attention.md § 3](cross_attention.md). Scoped per-sample by the padded `[B, M, D]` K/V tensor + boolean mask.

**What the output (`digest`) is:**
- `[B, K, D]` — per sample, K summary vectors. Row `k` is "the content of slot `k` after applying the learned matching criterion `Wq1/Wk1`, weighted by how much each expert slot matches."
- The `+ queries` residual means that at init (random-small queries, zero-init `Wo` downstream), the digest is essentially the raw learnable queries — i.e., Stage 1 starts as identity and training then adds useful modulation.

### Stage 2: image patches attend to per-sample digests

```python
q2 = Wq2(LN(image_embeds))                                         # [sumT, D]
k2 = Wk2(LN(digest))                                               # [B, K, D]
v2 = Wv2(LN(digest))                                               # [B, K, D]

# multi-head split, then gather per-token K/V by sample_id → [sumT, K, H, d]
scores2 = Q2·K2^T / √d          # [sumT, H, K]
attn2   = softmax(scores2, dim=-1)
ctx     = Σ attn2·V2            # [sumT, D]

return image_embeds + Wo(ctx)   # residual;  Wo zero-init
```

**What Q/K/V are here:**
- **Q = image patch tokens (`image_embeds`).** Same role as in cross-attn: per-patch queries.
- **K / V = the K digests of that patch's sample.** Stage 1 produced these; Stage 2 indexes them by `token_sample_id`.

**What the output is:**
- `image_embeds + Wo(ctx)` — the patch tokens get a per-token residual nudge sourced from the K sample-global digests.
- With `Wo` zero-initialized, at step 0 the residual is zero → identity function → model starts at pretrained Qwen behavior. Training reshapes `Wo` to pass useful signal.

### The `Wo`-is-zero property at init, explained once more

At step 0, the learnable queries are random-small and everything propagates, but `Wo·ctx = 0`. So:

```
image_embeds + Wo(ctx)  =  image_embeds + 0  =  image_embeds
```

The model is *exactly* the pretrained Qwen baseline. As training progresses, `Wo` picks up structure from the gradient of the loss and starts delivering useful expert-derived nudges to the image tokens.

---

## 4. What does each learnable query become?

The K queries are not assigned specific meanings — they specialize emergently under the training loss. Empirically (and by design), expect patterns like:

- **Best-match queries.** One or more queries learn to attend strongly to the correct gallery expert on match samples. Their `V` outputs carry that expert's identity content, which Stage 2 then routes to every patch.
- **Stranger detectors.** One or more queries learn to attend to the null slot when no real expert matches. Their `V` outputs near zero → harmless contribution → patches pass through ~clean. This is Q-Former's answer to the stranger problem: multiple orthogonal stranger-detection axes, versus cross-attn's single `null_slot` direction.
- **Ambiguity / similarity probes.** Queries that attend to several experts simultaneously produce digest tokens carrying "the query seems ambiguously similar to experts 2 and 3" — a signal the LLM's own attention can disambiguate downstream.
- **Aspect probes.** With 8 heads × 8 queries, there's enough capacity for queries to specialize on clothing vs face vs pose vs camera-invariant features — different heads of different queries attend to different parts of the expert representation space.

To inspect after training: dump `stage1_attn[b, :, :, :]` on known match vs known stranger samples and see which (query, head, slot) cells light up differently. Queries whose `attn1[b, k, :, 0]` (mass on null) is consistently high on strangers and low on matches are your stranger detectors.

---

## 5. Scoping: per-sample (same as cross-attn)

| Scope         | What each query sees                                                            |
|---------------|---------------------------------------------------------------------------------|
| per-image     | queries separately attend to each image's expert — degenerates, no cross-match |
| **per-sample** | **queries attend to the full {null, query, gallery 1..N} expert set for one sample** |
| per-batch     | queries attend across samples — leaks information, contaminates ReID            |

Per-sample is enforced the same way as in cross-attn: the padded `[B, M, D]` K/V tensor + sample-id maps built from `images_per_sample`. Stage 2 inherits the same scoping via `token_sample_id` gather.

---

## 6. Implementation choices

- **Number of learnable queries (K).** Default `--qformer_num_queries 8`. The knob that controls expressive capacity. For datasets with few distinct "aspects" (our ReID setup), 4–16 is a reasonable range. Going higher costs params linearly (`K · D`) and some Stage-1 compute.
- **Number of heads (H).** Default `--qformer_num_heads 8`. Applies to both stages — each head learns a different matching subspace.
- **Zero-init `Wo`.** Same trick as cross-attn. Guarantees the fuser is a no-op at step 0 so the model starts at pretrained baseline.
- **Random-small queries init.** `nn.init.normal_(self.queries, std=0.02)` — same scale as typical transformer parameter init. Non-zero so Stage-1 attention has *some* preference to start from; small enough to not dominate.
- **Zero-init `null_slot`.** Same as cross-attn. Training pulls it toward whatever direction wins attention on stranger-query samples.
- **Queries + residual in Stage 1.** `digest = attn·V + queries` — without the residual, digest would be a function of experts only, with no "query identity" preserved. The residual lets the model distinguish digest-slot-0 from digest-slot-1 etc. so Stage 2 can route on query identity.
- **Per-sample padded K/V.** Variable gallery lengths (our distractor benchmarks: N=1..5) handled by the same padding-plus-mask pattern as cross-attn. M = max(counts) + 1.

### Parameter count

At Qwen-7B (`hidden_size = 3584`), K = 8, H = 8:

| Component                          | Params |
|-------------------------------------|--------|
| Learnable queries `[K, D]`         | ~29K   |
| `null_slot [D]`                    | ~3.6K  |
| LayerNorms ×4 (stage 1+2)          | ~29K   |
| Stage 1: `Wq1`, `Wk1`, `Wv1`       | ~39M   |
| Stage 2: `Wq2`, `Wk2`, `Wv2`, `Wo` | ~51M   |
| **Total**                          | **~90M (~1.2% of 7B)** |

Roughly 2× the cross-attn fuser but still ≤1.3% of the base model.

### Compute cost

Per batch, with B samples × (N+1) images × T tokens-per-image, K queries, D hidden dim:

- Stage 1: `B · K · M · D` per head (M ≤ N+2) → e.g. `4 · 8 · 6 · 3584 = 0.7M` per head
- Stage 2: `sumT · K · D` per head → e.g. `(4·6·128) · 8 · 3584 = 88M` per head

Stage 2 dominates and is still a rounding error next to the LLM forward.

---

## 7. Relation to `ExpertCrossAttention`

Both share the same plumbing (`input_mode` switch, `expert_fuser` attribute, `images_per_sample`, `training_parameters expert_fuser`). They differ only in the math inside `.forward(...)`:

| Aspect                  | `ExpertCrossAttention`                          | `ExpertQFormer`                                            |
|-------------------------|-------------------------------------------------|------------------------------------------------------------|
| Query source            | Image patch tokens (derived from ViT)           | K learnable parameters (trained)                           |
| Matching happens        | Per-patch, once per token                       | Per-sample, once per sample (Stage 1) then routed (Stage 2) |
| Output                  | Residual add on image tokens                    | Residual add on image tokens                               |
| Abstain path            | `null_slot` in K/V (one direction)              | `null_slot` + specialization across K queries              |
| Params (7B, defaults)   | ~51M                                            | ~90M                                                       |
| Interpretability        | Attention mass per slot per patch (sumT × M)   | Attention mass per query per slot (K × M) — compact        |

Switching between them is just `--input_mode expert_cross_attn` vs `--input_mode expert_qformer`. The rest of the run configuration (warm-start, training_parameters, batch size) is identical.

---

## 8. How to run

In `run_train.sh`:

```bash
INPUT_MODE="expert_qformer"
```

And the full command uses the same set of flags cross-attn used, plus two Q-Former-specific knobs:

```bash
python train.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --train_file /path/to/train_data.json \
  --object_type person \
  --feature_mode expert \
  --expert_feature wyzev0323token \
  --input_mode expert_qformer \
  --qformer_num_queries 8 \
  --qformer_num_heads 8 \
  --training_parameters merger expert_projector expert_fuser \
  --warmup_connector_path runs/<prior_run>/connector.pt \
  --learning_rate 2e-4 --batch_size 4 \
  --captions False --prefix qformer-distractor-sft-qwen7b
```

---

## 9. Backward compatibility

All existing input modes (`image_only`, `expert_and_image_attn`, `expert_and_image_add`, `expert_only`, `expert_and_image_concat`, `expert_cross_attn`) are untouched.

- `model.visual.expert_fuser` is still only instantiated for `expert_cross_attn` or `expert_qformer`; otherwise `None`.
- The `elif` branch in [replace.py](replace.py)'s `My_Qwen2_5_VLModel_forward` now gates on both `expert_cross_attn` and `expert_qformer` (they share the same upstream plumbing — the fuser module itself differs).
- `--training_parameters expert_fuser` trains whichever fuser class is installed (substring match in `setup_trainable_parameters`). You don't need a new flag.
- `run_test.sh` parser recognizes `expert_qformer` as a 2-token input_mode suffix; existing 2/3/4-token modes continue to parse identically.

### Checkpoint compatibility

- A **cross-attn connector** (trained with `input_mode=expert_cross_attn`) loaded into a **Q-Former run** via `--warmup_connector_path` will only match the `expert_projector.*` keys (those are the only keys the warm-start logic filters in). The fuser itself is freshly initialized. This is intentional — the math differs, so sharing fuser weights isn't meaningful.
- The reverse direction (Q-Former → cross-attn warm-start) also just loads `expert_projector`.

---

## 10. Sanity checks before committing to a long run

1. **Zero-init check.** With `Wo.weight = 0`, the first training step's loss on the pre-fuser model should equal pure-Qwen baseline loss. If it doesn't, something is wrong upstream (expert feature shape mismatch, dtype issue, etc.).

2. **Stage-1 attention shape.** Print `attn1.shape` after one forward — should be `[B, K, H, max(N+1)+1]`. If it isn't, `images_per_sample` or `expert_sample_id` is miswired.

3. **Per-sample scoping.** On a batch with two samples, verify that perturbing sample-1 experts doesn't change sample-0 tokens' output. (Can do this with a hook.)

4. **Stranger-batch tracking.** After ~500 steps, take a stranger-labeled batch and dump `attn1[:, :, :, 0].mean()` (mean mass on null slot). You should see it above 1/M (i.e., above uniform). If it stays at 1/M, Stage 1 isn't learning to route strangers to null yet.

5. **Match-batch tracking.** On a family batch, dump `attn1[:, :, :, 0]` — should be *below* uniform (null is losing mass to the correct expert slot).

Queries whose Stage-1 attention consistently concentrates on null for strangers and on a specific real slot for matches are functioning as "stranger detector + best-match extractor" pairs — that's the specialization you want to see emerge.

---

## 11. Cheat sheet

| Symbol                      | Shape               | What it is                                                                         |
|-----------------------------|---------------------|------------------------------------------------------------------------------------|
| `self.queries`              | `[K, D]`            | Learnable query parameters, shared across all samples in the batch.                |
| `self.null_slot`            | `[D]`               | Learnable abstain anchor, prepended to every sample's K/V set.                     |
| `image_embeds`              | `[sumT, D]`         | Qwen ViT patch tokens, stacked across the batch.                                   |
| `expert_feature`            | `[N_img, D]`        | One `expert_projector` output per image.                                           |
| `kv`                        | `[B, M, D]`         | Per-sample K/V: slot 0 = null, slots 1..n = expert vectors. M = max(N+1)+1.        |
| `queries_b = queries.expand` | `[B, K, D]`         | Broadcasted queries, same for every sample.                                        |
| **Stage 1**                 |                     |                                                                                    |
| `Q1`                        | `[B, K, H, d]`      | `Wq1(LN(queries))`                                                                 |
| `K1 / V1`                   | `[B, M, H, d]`      | `Wk1/Wv1(LN(kv))`                                                                  |
| `scores1`                   | `[B, K, H, M]`      | `Q1·K1^T / √d`, masked to valid slots.                                             |
| `attn1`                     | `[B, K, H, M]`      | Softmax over slots, per (sample, query, head).                                     |
| `digest`                    | `[B, K, D]`         | `Σ attn1·V1 + queries` (residual).                                                 |
| **Stage 2**                 |                     |                                                                                    |
| `Q2`                        | `[sumT, H, d]`      | `Wq2(LN(image_embeds))`                                                            |
| `K2 / V2`                   | `[B, K, H, d]`      | `Wk2/Wv2(LN(digest))`                                                              |
| `scores2`                   | `[sumT, H, K]`      | `Q2·K2^T / √d`, gathered per-sample.                                               |
| `attn2`                     | `[sumT, H, K]`      | Softmax over K digests, per (patch, head).                                         |
| `ctx`                       | `[sumT, D]`         | `Σ attn2·V2`, heads concatenated.                                                  |
| `Wo(ctx)`                   | `[sumT, D]`         | Residual delta. Zero at init (`Wo.weight = 0`).                                    |
| **Output**                  | `[sumT, D]`         | `image_embeds + Wo(ctx)` — what the LLM reads as image tokens.                     |
