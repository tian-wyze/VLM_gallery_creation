"""
Prepare train/test splits for VLM person re-identification with varying gallery sizes.

Test benchmarks (10 total):
  8 non-distractor:  {same,cross}clothes × {singleton,family} × {same,cross}camera
  2 distractor:      distractor_cropped_singleton, distractor_cropped_family

Training: all scenarios from training households, balanced by family size.
Galleries are padded with hard negatives (high DINOv2 similarity) up to --max_gallery.

Label format:
  label > 0 : 1-indexed position of the matching person in the gallery
  label = -1: the query is a stranger (distractor); no gallery person matches

Each case includes a "similarity" field: cosine similarities between the query
and each gallery image (DINOv2 L/14 embeddings).

Input:
  household_info_v2_{same,cross}_clothes.json
  Structure: {household_id → {identity_id → {mac → {query: [], gallery: []}}}}
"""

import json
import os
import random
import argparse
import torch
import torchvision.transforms as transforms
from collections import defaultdict, Counter
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def parse_id(filename):
    """Extract identity_id from an image path."""
    return '_'.join(filename.split('/')[-1].split('_')[:-4])

def parse_household_id(identity_id):
    return '_'.join(identity_id.split('_')[:-1])

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Saved → {path}")


class UsageTracker:
    """Track per-identity image usage to promote diversity across cases (#6)."""

    def __init__(self):
        self._used = defaultdict(set)

    def pick(self, identity_id, candidates):
        """Pick one image, preferring those not yet used for this identity."""
        unused = [c for c in candidates if c not in self._used[identity_id]]
        chosen = random.choice(unused) if unused else random.choice(candidates)
        self._used[identity_id].add(chosen)
        return chosen


def merge_query_gallery_pools(hh_dict, household_ids):
    """Return a copy of hh_dict where query and gallery pools are merged (#1)."""
    merged = {}
    for hh_id in household_ids:
        if hh_id not in hh_dict:
            continue
        merged[hh_id] = {}
        for id_id, macs in hh_dict[hh_id].items():
            merged[hh_id][id_id] = {}
            for mac, splits in macs.items():
                pool = list(set(splits.get('query', []) + splits.get('gallery', [])))
                merged[hh_id][id_id][mac] = {'query': pool, 'gallery': pool}
    return merged


def collect_all_image_paths(*hh_dicts):
    """Collect all unique image paths from one or more household dicts."""
    paths = set()
    for hh_dict in hh_dicts:
        for identities in hh_dict.values():
            for macs in identities.values():
                for splits in macs.values():
                    paths.update(splits.get('query', []))
                    paths.update(splits.get('gallery', []))
    return sorted(paths)


def build_image_metadata(*hh_dicts):
    """Map each image path to (household_id, identity_id)."""
    meta = {}
    for hh_dict in hh_dicts:
        for hh_id, identities in hh_dict.items():
            for id_id, macs in identities.items():
                for splits in macs.values():
                    for img in splits.get('query', []) + splits.get('gallery', []):
                        meta[img] = (hh_id, id_id)
    return meta


# ═══════════════════════════════════════════════════════════════════════
# DINOv2 embedding computation & caching
# ═══════════════════════════════════════════════════════════════════════

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


def compute_dinov2_embeddings(image_paths, batch_size=128):
    """Compute DINOv2 ViT-L/14 embeddings for a list of image paths."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImagePathDataset(image_paths, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  DINOv2 device: {device}')
    model = model.to(device)

    all_features = []
    for batch in tqdm(loader, desc='  DINOv2 embedding'):
        with torch.no_grad():
            feat = model(batch.to(device))
        feat = feat / feat.norm(dim=1, keepdim=True)
        all_features.append(feat.cpu())

    features = torch.cat(all_features, dim=0)
    embeddings = {path: features[i] for i, path in enumerate(image_paths)}
    return embeddings


def load_or_compute_embeddings(image_paths, cache_dir='cached_embeddings'):
    """Load cached DINOv2 embeddings or compute them."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'dinov2_embeddings.pt')

    cached = {}
    if os.path.exists(cache_file):
        print(f'  Loading cached embeddings from {cache_file}')
        cached = torch.load(cache_file, map_location='cpu')

    missing = [p for p in image_paths if p not in cached]
    if missing:
        print(f'  Computing embeddings for {len(missing)} new images '
              f'({len(cached)} already cached)')
        new = compute_dinov2_embeddings(missing)
        cached.update(new)
        torch.save(cached, cache_file)
        print(f'  Saved {len(cached)} embeddings → {cache_file}')
    else:
        print(f'  All {len(image_paths)} embeddings loaded from cache')

    return cached


# ═══════════════════════════════════════════════════════════════════════
# Hard negative mining
# ═══════════════════════════════════════════════════════════════════════

class HardNegativeMiner:
    """Find hard negative gallery images using pre-computed embeddings.

    On init, precomputes top-K nearest neighbors for every image in the pool
    via batched matrix multiplication.  find_hard_negatives() is then a cheap
    lookup + filtering (no per-call matmul).
    """

    def __init__(self, embeddings, pool_paths, metadata, top_k=200,
                 batch_size=256, device=None):
        """
        embeddings: {image_path: tensor}
        pool_paths: list of candidate image paths
        metadata:   {image_path: (household_id, identity_id)}
        top_k:      number of nearest neighbors to precompute per image
        device:     torch device for the top-K precompute. Defaults to CUDA
                    when the pool is large (>50k) and CUDA is available;
                    otherwise CPU. Pool matrix is freed from GPU after precompute.
        """
        self.embeddings = embeddings
        self.pool_paths = pool_paths
        self.pool_matrix = torch.stack([embeddings[p] for p in pool_paths])  # (N, D)
        self.pool_hh = [metadata[p][0] for p in pool_paths]
        self.pool_id = [metadata[p][1] for p in pool_paths]
        self.path_to_idx = {p: i for i, p in enumerate(pool_paths)}

        N = len(pool_paths)
        self._top_k = min(top_k, N)

        if device is None:
            device = ('cuda' if torch.cuda.is_available() and N > 50_000
                      else 'cpu')

        print(f"    Precomputing top-{self._top_k} neighbors for {N} "
              f"images on {device}...")
        pool_dev = self.pool_matrix.to(device) if device != 'cpu' else self.pool_matrix
        self._nbr_indices = torch.zeros(N, self._top_k, dtype=torch.long)
        for start in tqdm(range(0, N, batch_size), desc='    Neighbors'):
            end = min(start + batch_size, N)
            sims = torch.mm(pool_dev[start:end], pool_dev.T)
            _, topk = torch.topk(sims, self._top_k, dim=1)
            self._nbr_indices[start:end] = topk.cpu()
        if device != 'cpu':
            del pool_dev
            torch.cuda.empty_cache()

    def find_hard_negatives(self, query_path, exclude_hh_ids, n,
                            sample_from_top_k=10):
        """Return up to n hard-negative image paths from different identities.

        exclude_hh_ids:    set (or single str) of household IDs to exclude.
        sample_from_top_k: collect the top-K valid candidates by similarity,
                           then randomly sample n from them for diversity.
                           Set to n to disable randomization (strict top-n).
        The query image itself is also always excluded.
        """
        if n <= 0:
            return []

        if isinstance(exclude_hh_ids, str):
            exclude_hh_ids = {exclude_hh_ids}

        idx = self.path_to_idx.get(query_path)
        if idx is not None:
            neighbors = self._nbr_indices[idx]
        else:
            q_emb = self.embeddings[query_path].unsqueeze(0)
            sims = torch.mm(q_emb, self.pool_matrix.T).squeeze(0)
            _, neighbors = torch.topk(sims, self._top_k)

        # Collect up to pool_size valid candidates — each from a different
        # household (which automatically means a different identity).
        pool_size = max(n, sample_from_top_k)
        candidates = []
        seen_hhs = set()
        for ni in neighbors:
            ni = ni.item()
            path = self.pool_paths[ni]
            if path == query_path:
                continue
            if self.pool_hh[ni] in exclude_hh_ids:
                continue
            if self.pool_hh[ni] in seen_hhs:
                continue
            candidates.append(path)
            seen_hhs.add(self.pool_hh[ni])
            if len(candidates) >= pool_size:
                break

        # Randomly sample n from the top-K valid candidates
        if len(candidates) <= n:
            return candidates
        return random.sample(candidates, n)

    def get_similarities(self, query_path, gallery_paths):
        """Cosine similarities between query and each gallery image."""
        q = self.embeddings[query_path]
        return [round(torch.dot(q, self.embeddings[g]).item(), 4)
                for g in gallery_paths]


# ═══════════════════════════════════════════════════════════════════════
# Household classification & train/test split
# ═══════════════════════════════════════════════════════════════════════

def classify_households(hh_same, hh_cross):
    """
    Classify every household into three tiers:
      tier1 — family, ≥2 members share a MAC with gallery  (same_camera viable)
      tier2 — family, no shared MAC                         (cross_camera only)
      tier3 — singleton
    """
    all_hh = set(hh_same.keys()) | set(hh_cross.keys())
    tier1, tier2, tier3 = [], [], []

    for hh_id in all_hh:
        ids_union = set(hh_same.get(hh_id, {}).keys()) | set(hh_cross.get(hh_id, {}).keys())
        if len(ids_union) <= 1:
            tier3.append(hh_id)
            continue

        shared = False
        for hh_dict in [hh_same, hh_cross]:
            if hh_id not in hh_dict:
                continue
            mac_to_ids = defaultdict(set)
            for id_id, macs in hh_dict[hh_id].items():
                for mac, splits in macs.items():
                    if splits.get('gallery', []):
                        mac_to_ids[mac].add(id_id)
            if any(len(ids) >= 2 for ids in mac_to_ids.values()):
                shared = True
                break
        (tier1 if shared else tier2).append(hh_id)

    return tier1, tier2, tier3


def split_train_test(hh_same, hh_cross,
                     n_singleton=70, n_fam_samecam=30, n_fam_crosscam=15):
    """Stratified split at the household level."""
    tier1, tier2, tier3 = classify_households(hh_same, hh_cross)
    # #8: Sort before shuffle for determinism
    tier1.sort(); tier2.sort(); tier3.sort()
    random.shuffle(tier1); random.shuffle(tier2); random.shuffle(tier3)

    t1 = tier1[:min(n_fam_samecam, len(tier1))]
    t2 = tier2[:min(n_fam_crosscam, len(tier2))]
    t3 = tier3[:min(n_singleton, len(tier3))]

    test_ids = set(t1) | set(t2) | set(t3)
    all_ids = set(tier1) | set(tier2) | set(tier3)
    train_ids = all_ids - test_ids

    print(f"\n{'='*60}")
    print("Train / Test Household Split")
    print(f"{'='*60}")
    print(f"  Family shared-MAC:  {len(t1):3d} test / {len(tier1)-len(t1):3d} train / {len(tier1):3d} total")
    print(f"  Family cross-only:  {len(t2):3d} test / {len(tier2)-len(t2):3d} train / {len(tier2):3d} total")
    print(f"  Singleton:          {len(t3):3d} test / {len(tier3)-len(t3):3d} train / {len(tier3):3d} total")
    print(f"  All:                {len(test_ids):3d} test / {len(train_ids):3d} train / {len(all_ids):3d} total")
    return test_ids, train_ids


# ═══════════════════════════════════════════════════════════════════════
# Non-distractor case building
# ═══════════════════════════════════════════════════════════════════════

def find_valid_triples(hh_dict, household_ids, hh_type, camera):
    """Find all (household_id, identity_id, query_mac) triples for a scenario."""
    triples = []
    for hh_id in household_ids:
        if hh_id not in hh_dict:
            continue
        identities = hh_dict[hh_id]
        n = len(identities)
        if hh_type == 'singleton' and n != 1:
            continue
        if hh_type == 'family' and n < 2:
            continue

        for qid, macs in identities.items():
            for qmac, splits in macs.items():
                if not splits.get('query', []):
                    continue
                ok = True
                for id_id, id_macs in identities.items():
                    if camera == 'same':
                        if not id_macs.get(qmac, {}).get('gallery', []):
                            ok = False; break
                    else:
                        if not any(s.get('gallery', [])
                                   for m, s in id_macs.items() if m != qmac):
                            ok = False; break
                if ok:
                    triples.append((hh_id, qid, qmac))
    return triples


def build_nondistr_case(hh_dict, hh_id, query_id, query_mac, camera,
                        miner=None, max_gallery=5, tracker=None):
    """
    Build one non-distractor case.

    Gallery = 1 image per household member + hard negatives up to max_gallery.
    Gallery is shuffled; label = 1-indexed position of the matching member.
    """
    identities = hh_dict[hh_id]

    q_imgs = identities[query_id].get(query_mac, {}).get('query', [])
    if not q_imgs:
        return None
    q_img = tracker.pick(query_id, q_imgs) if tracker else random.choice(q_imgs)

    gallery = []  # (tag, image_path)
    for id_id, id_macs in identities.items():
        if camera == 'same':
            g_imgs = id_macs.get(query_mac, {}).get('gallery', [])
        else:
            g_imgs = [img for mac, splits in id_macs.items()
                      if mac != query_mac
                      for img in splits.get('gallery', [])]
        if not g_imgs:
            return None
        # Exclude the query image itself from gallery candidates
        g_candidates = [g for g in g_imgs if g != q_img]
        if not g_candidates:
            return None
        chosen = tracker.pick(id_id, g_candidates) if tracker else random.choice(g_candidates)
        gallery.append((id_id, chosen))

    # Pad with hard negatives up to a random target size in [len(gallery), max_gallery]
    target_size = random.randint(len(gallery), max(len(gallery), max_gallery))
    n_pad = max(0, target_size - len(gallery))
    if n_pad > 0 and miner is not None:
        hard_negs = miner.find_hard_negatives(q_img, hh_id, n_pad)
        for img in hard_negs:
            gallery.append((None, img))

    random.shuffle(gallery)
    label = next(i + 1 for i, (tag, _) in enumerate(gallery) if tag == query_id)

    return {
        'query': q_img,
        'gallery': [img for _, img in gallery],
        'label': label,
    }


def sample_test_cases(triples, hh_dict, camera, n_ids=50, n_cases=150,
                      miner=None, max_gallery=5):
    """Sample n_cases test cases, covering up to n_ids identities."""
    if not triples:
        return [], {}

    by_id = defaultdict(list)
    for t in triples:
        by_id[t[1]].append(t)

    all_ids = list(by_id.keys())
    n_ids = min(n_ids, len(all_ids))
    sampled_ids = random.sample(all_ids, n_ids)

    per_id = n_cases // n_ids
    extra = n_cases % n_ids

    cases = []
    id_ct = {}

    for i, id_id in enumerate(sampled_ids):
        target = per_id + (1 if i < extra else 0)
        built = 0
        for _ in range(target * 10):
            t = random.choice(by_id[id_id])
            c = build_nondistr_case(hh_dict, t[0], t[1], t[2], camera,
                                    miner=miner, max_gallery=max_gallery)
            if c:
                cases.append(c)
                built += 1
                if built >= target:
                    break
        id_ct[id_id] = built

    return cases, id_ct


# ═══════════════════════════════════════════════════════════════════════
# Distractor case building
# ═══════════════════════════════════════════════════════════════════════

def collect_stranger_pool(hh_same, hh_cross, household_ids):
    """Collect (household_id, identity_id, image_path) for stranger queries."""
    pool = []
    for hh_dict in [hh_same, hh_cross]:
        for hh_id in household_ids:
            if hh_id not in hh_dict:
                continue
            for id_id, macs in hh_dict[hh_id].items():
                for splits in macs.values():
                    for img in splits.get('query', []):
                        pool.append((hh_id, id_id, img))
    return pool


def _build_singleton_distractor(targets, stranger_pool, embeddings, max_gallery=5):
    """Singleton distractor: gallery = up to max_gallery images of the sole
    resident, selected by similarity to the stranger query."""
    if not targets:
        return None
    hh_id, hh_dict = random.choice(targets)
    id_id = list(hh_dict[hh_id].keys())[0]

    # Collect ALL gallery images of the resident
    all_g = [img for splits in hh_dict[hh_id][id_id].values()
             for img in splits.get('gallery', [])]
    if not all_g:
        return None

    # Pick stranger
    valid = [s for s in stranger_pool if s[0] != hh_id]
    if not valid:
        return None
    _, _, q_path = random.choice(valid)

    # Random target gallery size in [1, max_gallery]
    target_size = random.randint(1, max_gallery)

    # Select the top-target_size images of the resident most similar to the query
    q_emb = embeddings.get(q_path)
    if q_emb is not None:
        sims = [(torch.dot(q_emb, embeddings[g]).item(), g)
                for g in all_g if g in embeddings]
        sims.sort(reverse=True)
        gallery = [g for _, g in sims[:target_size]]
    else:
        gallery = all_g[:target_size]

    random.shuffle(gallery)
    return {'query': q_path, 'gallery': gallery, 'label': -1}


def _build_family_distractor(targets, stranger_pool, miner,
                              max_gallery=5, tracker=None):
    """Family distractor: gallery = 1 image per member + hard negatives
    similar to the stranger query, up to max_gallery."""
    if not targets:
        return None
    hh_id, hh_dict = random.choice(targets)

    gallery = []
    for id_id, macs in hh_dict[hh_id].items():
        all_g = [img for splits in macs.values() for img in splits.get('gallery', [])]
        if all_g:
            chosen = tracker.pick(id_id, all_g) if tracker else random.choice(all_g)
            gallery.append(chosen)
    if not gallery:
        return None

    valid = [s for s in stranger_pool if s[0] != hh_id]
    if not valid:
        return None
    stranger_hh, _, q_path = random.choice(valid)

    # Pad up to a random target size in [len(gallery), max_gallery]
    # — exclude BOTH target household AND stranger's household
    target_size = random.randint(len(gallery), max(len(gallery), max_gallery))
    n_pad = max(0, target_size - len(gallery))
    if n_pad > 0 and miner is not None:
        hard_negs = miner.find_hard_negatives(q_path, {hh_id, stranger_hh}, n_pad)
        gallery.extend(hard_negs)

    random.shuffle(gallery)
    return {'query': q_path, 'gallery': gallery, 'label': -1}


def build_distractor_subset(hh_same, hh_cross, test_ids, hh_type,
                            embeddings, miner, n_cases=200, max_gallery=5,
                            tracker=None):
    """Build distractor test cases, stratified by family size."""
    stranger_pool = collect_stranger_pool(hh_same, hh_cross, test_ids)

    by_size = defaultdict(list)
    for hh_dict in [hh_same, hh_cross]:
        for hh_id in test_ids:
            if hh_id not in hh_dict:
                continue
            n = len(hh_dict[hh_id])
            if hh_type == 'singleton' and n == 1:
                by_size[n].append((hh_id, hh_dict))
            elif hh_type == 'family' and n >= 2:
                by_size[n].append((hh_id, hh_dict))

    if not by_size:
        return []

    sizes = sorted(by_size.keys())
    per_size = n_cases // len(sizes)
    extra = n_cases % len(sizes)

    cases = []
    for i, sz in enumerate(sizes):
        target = per_size + (1 if i >= len(sizes) - extra else 0)
        built = 0
        for _ in range(target * 5):
            if hh_type == 'singleton':
                c = _build_singleton_distractor(
                    by_size[sz], stranger_pool, embeddings, max_gallery)
            else:
                c = _build_family_distractor(
                    by_size[sz], stranger_pool, miner, max_gallery, tracker)
            if c:
                cases.append(c)
                built += 1
                if built >= target:
                    break
    return cases


# ═══════════════════════════════════════════════════════════════════════
# Training data
# ═══════════════════════════════════════════════════════════════════════

def build_training_data(hh_same, hh_cross, train_ids,
                        sampling_rates, miner, embeddings,
                        distractor_fraction=0.2, max_gallery=5):
    """Build training cases with hard-negative padding and role swapping."""
    non_distr = []
    stats = defaultdict(int)

    # #1: Merge query/gallery pools
    hh_same_m = merge_query_gallery_pools(hh_same, train_ids)
    hh_cross_m = merge_query_gallery_pools(hh_cross, train_ids)

    # #6: Diversity tracker
    tracker = UsageTracker()

    for hh_dict, clothes in [(hh_same_m, 'same_clothes'), (hh_cross_m, 'cross_clothes')]:
        sorted_ids = sorted(hh_id for hh_id in train_ids if hh_id in hh_dict)
        for hh_id in tqdm(sorted_ids, desc=f'  Train non-distr ({clothes})'):
            identities = hh_dict[hh_id]
            fam_size = len(identities)
            n_per_id = sampling_rates.get(fam_size, max(sampling_rates.values()))

            for qid in identities:
                valid_macs = [m for m, s in identities[qid].items()
                              if s.get('query', []) or s.get('gallery', [])]
                if not valid_macs:
                    continue

                built = 0
                for _ in range(n_per_id * 20):
                    camera = random.choice(['same', 'cross'])
                    qmac = random.choice(valid_macs)
                    case = build_nondistr_case(
                        hh_dict, hh_id, qid, qmac, camera,
                        miner=miner, max_gallery=max_gallery, tracker=tracker)
                    if case:
                        non_distr.append(case)
                        stats[f'{clothes}_{camera}cam_size{fam_size}'] += 1
                        built += 1
                        if built >= n_per_id:
                            break

    print(f"  Non-distractor cases built: {len(non_distr)}")

    # Distractor training cases — stratified by family size
    stranger_pool = collect_stranger_pool(hh_same_m, hh_cross_m, train_ids)
    by_size = defaultdict(list)
    for hh_dict in [hh_same_m, hh_cross_m]:
        for hh_id in train_ids:
            if hh_id in hh_dict:
                by_size[len(hh_dict[hh_id])].append((hh_id, hh_dict))

    n_distr = int(len(non_distr) * distractor_fraction)
    sizes = sorted(by_size.keys())
    per_size = n_distr // len(sizes) if sizes else 0
    extra_cases = n_distr % len(sizes) if sizes else 0
    distr = []

    print(f"  Building {n_distr} distractor training cases across {len(sizes)} size buckets...")
    for i, sz in enumerate(sizes):
        target = per_size + (1 if i >= len(sizes) - extra_cases else 0)
        built = 0
        for _ in range(target * 5):
            if sz == 1:
                c = _build_singleton_distractor(
                    by_size[sz], stranger_pool, embeddings, max_gallery)
            else:
                c = _build_family_distractor(
                    by_size[sz], stranger_pool, miner, max_gallery, tracker)
            if c:
                distr.append(c)
                stats[f'distractor_size{len(c["gallery"])}'] += 1
                built += 1
                if built >= target:
                    break
        print(f"    size {sz}: {built}/{target} cases")

    all_cases = non_distr + distr
    random.shuffle(all_cases)
    return all_cases, stats


# ═══════════════════════════════════════════════════════════════════════
# Similarity scores
# ═══════════════════════════════════════════════════════════════════════

def add_similarity_scores(cases, miner):
    """Add DINOv2 cosine similarity between query and each gallery image."""
    for case in cases:
        case['similarity'] = miner.get_similarities(case['query'], case['gallery'])


def add_household_ids(cases, metadata):
    """Add household_id and identity_id for query and each gallery image.

    Padded hard-negative images show their source household/identity, so you
    can tell at a glance which gallery entries are household members vs.
    distractors from other households.
    """
    for case in cases:
        q_hh, q_id = metadata.get(case['query'], ('unknown', 'unknown'))
        case['query_household_id'] = q_hh
        case['query_identity_id'] = q_id
        case['household_id'] = [
            metadata.get(g, ('unknown', 'unknown'))[0] for g in case['gallery']
        ]
        case['identity_id'] = [
            metadata.get(g, ('unknown', 'unknown'))[1] for g in case['gallery']
        ]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Prepare train/test for person ReID with varying gallery sizes.')
    parser.add_argument('--same_clothes', default='household_info_v2_same_clothes.json')
    parser.add_argument('--cross_clothes', default='household_info_v2_cross_clothes.json')
    parser.add_argument('--output_dir', default='benchmarks')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache_dir', default='cached_embeddings')
    parser.add_argument('--max_gallery', type=int, default=5,
                        help='Max gallery size (household members + hard negatives)')
    # Split sizes
    parser.add_argument('--n_test_singleton', type=int, default=70)
    parser.add_argument('--n_test_fam_samecam', type=int, default=30)
    parser.add_argument('--n_test_fam_crosscam', type=int, default=15)
    # Test subset sizes
    parser.add_argument('--test_n_ids', type=int, default=50)
    parser.add_argument('--test_n_cases', type=int, default=150)
    parser.add_argument('--distractor_n_cases', type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load household dicts ──
    print("Loading household dicts...")
    hh_same = load_json(args.same_clothes)
    hh_cross = load_json(args.cross_clothes)

    # ── Split ──
    test_ids, train_ids = split_train_test(
        hh_same, hh_cross,
        n_singleton=args.n_test_singleton,
        n_fam_samecam=args.n_test_fam_samecam,
        n_fam_crosscam=args.n_test_fam_crosscam,
    )
    save_json({
        'seed': args.seed,
        'n_test': len(test_ids),
        'n_train': len(train_ids),
        'test_household_ids': sorted(test_ids),
        'train_household_ids': sorted(train_ids),
    }, 'train_test_split.json')

    # ── Compute / load DINOv2 embeddings ──
    print("\nPreparing DINOv2 embeddings...")
    all_paths = collect_all_image_paths(hh_same, hh_cross)
    embeddings = load_or_compute_embeddings(all_paths, args.cache_dir)
    metadata = build_image_metadata(hh_same, hh_cross)

    # ── Build miners for each split ──
    print("\nBuilding hard-negative miners...")
    test_pool = [p for p in all_paths if metadata.get(p, ('', ''))[0] in test_ids]
    train_pool = [p for p in all_paths if metadata.get(p, ('', ''))[0] in train_ids]
    test_miner = HardNegativeMiner(embeddings, test_pool, metadata)
    train_miner = HardNegativeMiner(embeddings, train_pool, metadata)
    print(f"  Test pool:  {len(test_pool)} images")
    print(f"  Train pool: {len(train_pool)} images")

    # ── 8 non-distractor test subsets ──
    print("\nBuilding 8 non-distractor test subsets...")
    os.makedirs(args.output_dir, exist_ok=True)
    test_summary = {}

    for clothes_label, hh_dict in [('sameclothes', hh_same), ('crossclothes', hh_cross)]:
        for hh_type in ['singleton', 'family']:
            for cam_label in ['samecamera', 'crosscamera']:
                scenario = f'cropped_{clothes_label}_{hh_type}_{cam_label}'
                camera = 'same' if cam_label == 'samecamera' else 'cross'

                triples = find_valid_triples(hh_dict, test_ids, hh_type, camera)
                cases, id_ct = sample_test_cases(
                    triples, hh_dict, camera,
                    n_ids=args.test_n_ids, n_cases=args.test_n_cases,
                    miner=test_miner, max_gallery=args.max_gallery,
                )
                add_similarity_scores(cases, test_miner)
                add_household_ids(cases, metadata)

                gal_sizes = Counter(len(c['gallery']) for c in cases)
                res = {
                    'scenario': scenario,
                    'n_identities': len(id_ct),
                    'n_cases': len(cases),
                    'gallery_size_distribution': dict(sorted(gal_sizes.items())),
                    'eval_id_count': id_ct,
                    'eval_cases': cases,
                }
                save_json(res, os.path.join(args.output_dir, f'{scenario}.json'))
                test_summary[scenario] = (len(id_ct), len(cases), gal_sizes)

    # ── 2 distractor test subsets ──
    print("\nBuilding 2 distractor test subsets...")
    for hh_type in ['singleton', 'family']:
        scenario = f'distractor_cropped_{hh_type}'
        cases = build_distractor_subset(
            hh_same, hh_cross, test_ids, hh_type,
            embeddings=embeddings, miner=test_miner,
            n_cases=args.distractor_n_cases, max_gallery=args.max_gallery,
        )
        add_similarity_scores(cases, test_miner)
        add_household_ids(cases, metadata)

        gal_sizes = Counter(len(c['gallery']) for c in cases)
        res = {
            'scenario': scenario,
            'n_cases': len(cases),
            'gallery_size_distribution': dict(sorted(gal_sizes.items())),
            'eval_cases': cases,
        }
        save_json(res, os.path.join(args.output_dir, f'{scenario}.json'))
        test_summary[scenario] = (0, len(cases), gal_sizes)

    # ── Training data ──
    print("\nBuilding training data...")
    sampling_rates = {1: 30, 2: 40, 3: 30, 4: 30, 5: 30, 6: 30}
    train_cases, train_stats = build_training_data(
        hh_same, hh_cross, train_ids, sampling_rates,
        miner=train_miner, embeddings=embeddings,
        distractor_fraction=0.2, max_gallery=args.max_gallery,
    )
    print("\n  Adding similarity scores and household IDs to training data...")
    add_similarity_scores(train_cases, train_miner)
    add_household_ids(train_cases, metadata)
    save_json(train_cases, 'train_data.json')

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Test Benchmarks")
    print(f"{'='*60}")
    total_test = 0
    for scenario, (n_ids, n_cases, gal_sizes) in test_summary.items():
        gal_str = ', '.join(f'gal{k}:{v}' for k, v in sorted(gal_sizes.items()))
        id_str = f'{n_ids} ids, ' if n_ids else ''
        print(f"  {scenario:50s}  {id_str}{n_cases} cases  [{gal_str}]")
        total_test += n_cases
    print(f"  {'─'*70}")
    print(f"  Total test cases: {total_test}")

    print(f"\n{'='*60}")
    print("Training Data")
    print(f"{'='*60}")
    for cat in sorted(train_stats.keys()):
        print(f"  {cat:40s}  {train_stats[cat]} cases")
    print(f"  {'─'*70}")
    print(f"  Total training cases: {len(train_cases)}")

    train_gal = Counter(len(c['gallery']) for c in train_cases)
    print(f"\n  Training gallery-length distribution:")
    for gl in sorted(train_gal.keys()):
        print(f"    gallery size {gl}: {train_gal[gl]} cases")


if __name__ == '__main__':
    main()
