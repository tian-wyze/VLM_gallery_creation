# `expert_cross_attn` — Per-Sample Learnable Cross-Attention

A standalone, end-to-end explanation of the `expert_cross_attn` input mode for the IDA-VLM training pipeline. This document covers motivation, architecture, the precise role of Q/K/V, what the cross-attention output means, the null-slot design, scoping, implementation choices, and backward compatibility.

The implementation lives in:
- [utils/model_utils.py](utils/model_utils.py) — `ExpertCrossAttention` module and instantiation.
- [replace.py](replace.py) — the `expert_cross_attn` branch inside `My_Qwen2_5_VLModel_forward`.
- [utils/collate_utils.py](utils/collate_utils.py) / [test.py](test.py) / [inference.py](inference.py) — thread `images_per_sample` into the `expert_inputs` dict.

---

## 1. Motivation — why the existing `expert_and_image_attn` caps the upside

In `expert_and_image_attn`, the fusion is:

```
image_embeds = (1 − attn) * image_embeds + attn * broadcasted_expert_feature
```

where `attn` is the softmax of cosine similarities between each image-patch token and *that image's own* expert vector. Two structural weaknesses:

1. **Fixed mass budget per image.** Softmax over the `Ti` patch tokens of an image sums to 1. With `Ti ~ 64–256`, the *average* per-token weight is 0.4–1.5%. Even the best-matching token picks up only ~1% of the expert signal. Most of the fusion is basically identity.
2. **No cross-image interaction.** A query patch token can only see the query image's own expert. It never compares itself against a gallery image's expert descriptor — which is exactly the comparison the ReID task is about.

`expert_cross_attn` fixes both: it makes the fusion **learnable** (`Q`/`K`/`V`/`O` projections, multi-head) and it widens the attention scope so every token sees every expert vector in the same sample.

---

## 2. High-level data flow per sample (1 query + N gallery images)

```
Per-image path (unchanged):
  raw image ──► Qwen2.5 ViT + merger ──► patch tokens  [Ti, D]
  raw image ──► Expert backbone      ──► 1 vec [E_dim] ──► expert_projector ──► [D]

Per-sample fuser (new):
                       Patch tokens of ALL images in the sample
                       concatenated:  [sumT, D]     (queries)
                                            │
                                            ▼
                    ┌────────────────────────────────────────┐
                    │   ExpertCrossAttention  (multi-head)   │
                    │   Wq │ Wk │ Wv  learnable projections  │
                    │   Wo zero-init ⇒ step 0 is a no-op     │
                    └───────────────┬────────────────────────┘
                                    │         ▲
                                    │         │  keys / values  (N+2 slots)
                                    │         │
     [null_slot, query_expert, gallery1_expert, gallery2_expert, ... galleryN_expert]
                                    │   1 learnable null + (N+1) expert vectors, each [D]
                                    ▼
                  image_embeds ← image_embeds + Wo(attn_ctx)       (residual)
```

The rest of the model forward is unchanged — the LLM sees a sequence of token embeddings that have been augmented with identity-matching information.

---

## 3. What are Q, K, V in our case?

### Short answer

- **Q (query)** = Qwen ViT patch tokens. Every patch of every image — query image and gallery images alike — is a query row.
- **K / V (keys / values)** = the per-image expert descriptors, plus one learnable **null slot**. All (N+2) slots are shared across every patch in that sample.

In code ([model_utils.py:169-174](utils/model_utils.py#L169-L174)):

```python
q = self.Wq(self.ln_q(image_embeds))        # Q  ← Qwen ViT patches   [sumT, D]
k = self.Wk(self.ln_kv(kv))                 # K  ← expert descriptors [B, M, D]
v = self.Wv(self.ln_kv(kv))                 # V  ← expert descriptors [B, M, D]
```

### What lives in each K/V slot

Per sample:

```
slot 0:  null_slot                      (learnable "no match" anchor; shared parameter across all samples)
slot 1:  Wk/Wv(expert of image 0)       — typically the query image
slot 2:  Wk/Wv(expert of image 1)       — gallery 1
slot 3:  Wk/Wv(expert of image 2)       — gallery 2
 …
slot n:  Wk/Wv(expert of image n−1)     — gallery N
```

"Expert of image i" = **one vector per image** — the output of `expert_projector(expert_backbone(image_i))`. For Wyze v03_23_token → a 1280-dim vector from the ReID encoder, projected to Qwen's `hidden_size`. Each image contributes exactly one K/V pair, regardless of how many patches it has.

### Concrete example

Take a sample with 1 query image + 4 gallery images, each producing ~128 patch tokens:

- `Q`: ~5 × 128 = **640 query rows** (one per patch across all five images).
- `K / V`: **6 rows** per sample = `{null, query_expert, g1_expert, g2_expert, g3_expert, g4_expert}`.

### Three deliberate asymmetries

1. **Q is per patch; K/V is per image.** Qwen ViT patches are rich but noisy (background, texture, pose, clothing, face pixels). Expert descriptors are compact identity codes — one pre-digested vector per image. Making Q per-patch and K/V per-image means each patch gets to say "which identity do I line up with," rather than all patches sharing one identity signal per image.

2. **`Wq`, `Wk`, `Wv` are independent learnable projections.** Even though K and V come from the same `kv` tensor, they're transformed by different linear maps. `Wk` learns the *matching criterion* (what makes a token and an expert look alike); `Wv` learns *what to read back if matched*. This is the standard transformer decoupling — important because the relevance signal and the content signal don't have to live in the same subspace.

3. **K/V is scoped per-sample, not per-image or per-batch.** The same (N+2)-slot K/V set is reused for every patch in that sample. A different sample uses a different (N+2)-slot set. Enforced by building a padded `[B, M, D]` K/V tensor indexed by `sample_id` ([model_utils.py:154-166](utils/model_utils.py#L154-L166)):

```python
kv[:, 0] = self.null_slot.to(kv.dtype)        # every sample gets the null slot at index 0
mask[:, 0] = True
for b in range(B):
    idx = (expert_sample_id == b).nonzero(as_tuple=True)[0]   # images in sample b
    n = idx.numel()
    kv[b, 1:1 + n] = expert_feature[idx]       # expert vectors for sample b
    mask[b, 1:1 + n] = True
```

A query patch in sample 0 cannot attend to gallery experts of sample 3.

---

## 4. What is the cross-attention output, and what does it mean?

After Q/K/V exist, the module does four more things ([model_utils.py:176-188](utils/model_utils.py#L176-L188)):

```python
scores = torch.einsum('thd,tmhd->thm', q, k_t) * self.scale   # [sumT, H, M]  ← raw dot-products
scores = scores.masked_fill(~mask_t.unsqueeze(1), float('-inf'))
attn   = scores.softmax(dim=-1)                                # [sumT, H, M]  ← weights (sum to 1 over M)
ctx    = torch.einsum('thm,tmhd->thd', attn, v_t).reshape(-1, self.H * self.d)  # [sumT, D]
return image_embeds + self.Wo(ctx)                             # [sumT, D]     ← residual
```

### Stage 1 — `scores [sumT, H, M]`: "how relevant is each slot to this patch, per head?"

For each (patch token × head × slot) triple, a single scalar: *"on this head's matching criterion, how well does this patch resemble this expert descriptor?"* Scaled by `1/√d` so the softmax doesn't saturate at high hidden dimensions.

### Stage 2 — `attn [sumT, H, M]`: "per-head soft routing"

Softmax normalizes each head's scores over the `M` slots. For every (patch, head), you get a distribution summing to 1 over `{null, expert_0, expert_1, …, expert_{N}}`. Read `attn[t, h, :]` as *"head h's opinion about which identity patch t belongs to."* Different heads can disagree — one head might vote for null, another for gallery-3, another for the query — and their votes get reconciled downstream.

### Stage 3 — `ctx [sumT, D]`: the actual attention output

Per-patch, per-head weighted sum of V-projected expert content, concatenated across heads into a single `[D]` vector per patch. `ctx[t]` is a vector in Qwen's hidden space whose content answers:

> "If patch `t` had to summarize the expert descriptors that matched it well — one head at a time, via `Wv` — what one vector would that summary be?"

Concrete readings:

- **Only gallery-2 matched** → `ctx[t]` ≈ `Wv(g2_expert)` repeated across heads.
- **Mostly null matched** → `ctx[t]` ≈ `Wv(null_slot)`, which training drives toward a small / zero-ish vector.
- **Blend of query + gallery-1** → `ctx[t]` is a linear combination of `Wv(q_expert)` and `Wv(g1_expert)`, weighted by the attention. Different heads may emphasize different identities, and that's preserved because we `reshape` rather than average across heads, so `Wo` sees head-specific content.

`ctx` is *not yet* in the same subspace as the image tokens. It lives in `Wv`'s output space — a learned "expert content" subspace. The next step brings it back.

### Stage 4 — `Wo(ctx) [sumT, D]`: the residual delta — "how much to nudge the patch token by"

`Wo` is a learned linear map from "expert content space" back to "image-token space." It decides:
- How strongly to inject the attention output back into the LLM input.
- How to mix the per-head contributions together (stage 3 concatenated heads without mixing).

`Wo.weight` is **zero-initialized** ([model_utils.py:98-99](utils/model_utils.py#L98-L99)), so at step 0:

```
Wo(ctx) = 0   →   return image_embeds + 0 = image_embeds
```

The cross-attention is a no-op. Training gradually shapes `Wo` to produce a useful delta. The final output `image_embeds + Wo(ctx)` is the per-patch token embedding the LLM actually reads.

### What the LLM ends up seeing

The LLM's image tokens are now ViT patches *plus* a per-patch expert-informed nudge:

- **Stranger patches** → nudge ≈ 0 (via null slot). The LLM sees the plain ViT output and can answer `-1`.
- **Patches matching one specific gallery identity** → nudge in `Wo · Wv(that_expert)` direction. The LLM learns this direction means "this patch votes for that identity."
- **Ambiguous patches (e.g., family members wearing similar clothes)** → nudge is a *blend* of multiple experts. The LLM's own attention layers get to arbitrate by looking at the whole sequence.

The output is in the image-token subspace, added *residually* — so the fuser never replaces the ViT signal, it only augments it with identity-matching information distilled from the expert descriptors.

### One-sentence version

**`Q·K^T → softmax → weighted sum of V → Wo → residual add`**: each patch token pulls an identity-matching signal from the pool of expert descriptors available in its sample, runs it through a learned projection back into image-token space, and adds it as a nudge — so the LLM reads ViT patches that have been enriched with "who do I look like in this sample?" information.

---

## 5. Scoping: why per-sample, not per-image or per-batch

| Scope          | K/V per token           | What tokens see                                                                      |
|----------------|-------------------------|--------------------------------------------------------------------------------------|
| per-image      | 1 expert vector         | same as `expert_and_image_attn`, just parameterized — degenerates to a gate          |
| **per-sample** | **(N+2) expert vectors** | the full query+gallery identity-descriptor set for that sample, plus null            |
| per-batch      | B·(N+2)                 | leaks across samples → contaminates ReID                                             |

Per-sample is the only scope that is both (a) wide enough to let query-gallery comparison happen and (b) narrow enough to respect sample boundaries. Enforced via `token_sample_id` / `expert_sample_id` maps built from `images_per_sample` (threaded through the collate function).

---

## 6. Null slot (abstain token for stranger queries)

### Motivation

Without the null slot, softmax over `(N+1)` expert slots *forces* every image patch token to spend exactly 1.0 unit of mass on the real expert descriptors — there is no way to express "none of these match." For family/match scenarios that is fine (the softmax peaks on the matching identity). For **distractor (stranger) scenarios**, where the ground-truth answer is `-1`, this is harmful:

- The softmax smears mass ~uniformly across gallery experts, and `Wv · Wo` injects the *average* gallery identity back into the query's image tokens, pulling them toward the gallery cluster.
- The LLM, which is supposed to answer `-1`, now reads image tokens contaminated with gallery-identity content — systematic bias toward guessing a gallery index.

We observed this empirically: after adding `expert_cross_attn`, family-scenario accuracy went up vs the `expert_and_image_attn` baseline, but distractor accuracy dropped significantly. The null slot fixes that without touching the family path.

### What it is

One learnable `[D]` parameter, `self.null_slot`, prepended to every sample's K/V set:

```
K/V per sample = [ null_slot ,  query_expert ,  gallery_1_expert , … , gallery_N_expert ]
                   └───────────┴──────────── softmax over (N+2) slots ────────────┘
```

Both init values are zero:
- `null_slot = 0` → at init, all slots (null + experts) enter the softmax on equal footing; no bias.
- `Wo.weight = 0` → at init, the fuser output is zero regardless of the attention pattern.

### What training does

Training shapes `null_slot` into whatever direction maximizes attention from **stranger-query image tokens** specifically — because those are the tokens for which no real expert provides a better match. The LLM's loss gradient on the `-1` label pushes the system toward *"route stranger tokens to null → `Wo(ctx) ≈ 0` → image tokens unperturbed → LLM answers -1 cleanly."*

### Backward compatibility

`null_slot` is a new named parameter. Old `connector.pt` files trained with the null-less fuser still load via `strict=False`; the null slot simply stays at its zero init and is trained fresh:

```python
ckpt = torch.load('runs/<old_cross_attn_run>/connector.pt')
missing, unexpected = model.load_state_dict(ckpt, strict=False)
# 'model.visual.expert_fuser.null_slot' will appear in `missing` → stays zero-init.
```

### Ablating the null slot

Not exposed as a CLI flag. If you want to run without it:

```python
model.visual.expert_fuser.null_slot.data.zero_()
model.visual.expert_fuser.null_slot.requires_grad = False
```

This freezes it at zero (neutral contribution to the softmax logit — the null slot is still technically in the K/V set, but its logit stays near 0 throughout training).

### What to look for during training / evaluation

- After a few hundred steps, print `attn[:, :, 0].mean()` (mean softmax mass on slot 0) separately on match-label and stranger-label batches. You should see it trend *up* on stranger batches and stay low on match batches.
- If stranger accuracy is still below `expert_and_image_attn` after enough steps, the null slot probably hasn't found its region yet — try combining with `--warmup_connector_path` so the K/V side starts aligned and the null slot only has to learn its own direction.

---

## 7. Key implementation choices

- **Zero-initialized output projection (`Wo`).** `nn.init.zeros_(self.Wo.weight)` → at step 0 the fuser adds zero. The model starts with exactly the pretrained Qwen behavior and *learns* how much expert signal to inject, rather than being disrupted at init.
- **Learnable null slot.** A zero-initialized `[D]` parameter prepended to every sample's K/V set so the softmax always has an explicit "no match" option.
- **Multi-head** (default 8 heads). Different heads can specialize (body shape, clothing cues, face, camera-invariant features, etc.).
- **Padding + mask for variable gallery length.** `N` varies across samples in the distractor benchmarks (singleton → `N=1`, family → `N≤5`). The fuser builds a `[B, M, D]` padded K/V tensor with `M = max(N+1) + 1` (the `+1` is for the null slot) and a boolean validity mask; invalid slots get `-inf` in the scores so they receive zero softmax mass.
- **Optional warm-start.** A previously-trained `expert_projector` can be loaded via `--warmup_connector_path` so the K/V side of the attention starts already aligned to Qwen's hidden space, leaving only the fuser to learn.

---

## 8. Training parameters

Recommended starting setup:

```
--training_parameters merger expert_projector expert_fuser
--warmup_connector_path <path_to_prior_run>/connector.pt   # loads only 'expert_projector'
```

The new `expert_fuser` key maps to `model.visual.expert_fuser.*` in `setup_trainable_parameters`. Existing `training_parameters` choices (`merger`, `expert_projector`, `expert`, `llm`) are unchanged.

---

## 9. Parameter / compute cost

At Qwen-7B (`hidden_size = 3584`), 8 heads, the fuser has ~4·hidden² ≈ 51M params — ≤1% of the base model. Per batch the cross-attention does `sumT · (N+2) · D` multiplies per head; on a 4-sample × (5+1)-image × 128-token batch that's under 10M multiplies. Effectively free compared to the ViT and the LLM.

---

## 10. Backward compatibility

All existing input modes (`image_only`, `expert_and_image_attn`, `expert_and_image_add`, `expert_only`, `expert_and_image_concat`) are untouched. The new path is fully gated by `input_mode == 'expert_cross_attn'`:

- `model.visual.expert_fuser` is created only when that mode is selected; otherwise it's `None`.
- The new `'expert_cross_attn'` branch in [replace.py](replace.py)'s `My_Qwen2_5_VLModel_forward` is an additional `elif`, not a rewrite.
- `images_per_sample` is now always computed and placed in `expert_inputs`, but it is only read by the new branch, so older connectors and configs behave identically.

---

## 11. Cheat sheet

| Symbol                | Shape          | What it is                                                                                 |
|-----------------------|----------------|--------------------------------------------------------------------------------------------|
| `image_embeds`        | `[sumT, D]`    | Qwen ViT patch tokens, stacked across every image in the batch.                             |
| `expert_feature`      | `[N_img, D]`   | One `expert_projector` output per image.                                                    |
| `null_slot`           | `[D]`          | Learnable abstain anchor, shared across all samples.                                        |
| `kv`                  | `[B, M, D]`    | Padded per-sample K/V set: slot 0 = null, slots 1..n = expert vectors. `M = max(N+1) + 1`. |
| `Q = Wq(LN(image_embeds))` | `[sumT, D]` | Per-patch queries.                                                                       |
| `K = Wk(LN(kv))`      | `[B, M, D]`    | Per-slot keys.                                                                              |
| `V = Wv(LN(kv))`      | `[B, M, D]`    | Per-slot values.                                                                            |
| `scores`              | `[sumT, H, M]` | `q·k^T / √d`, per-head.                                                                     |
| `attn`                | `[sumT, H, M]` | Softmax over slots, per head. Sums to 1 across M per (t, h).                                |
| `ctx`                 | `[sumT, D]`    | Per-patch V-projected content, heads concatenated.                                          |
| `Wo(ctx)`             | `[sumT, D]`    | Residual delta. Zero at init (`Wo.weight = 0`).                                             |
| **Output**            | `[sumT, D]`    | `image_embeds + Wo(ctx)` — what the LLM reads as image tokens.                              |
