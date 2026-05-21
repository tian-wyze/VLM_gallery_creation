"""prepare_test.py — Build 10 test benchmarks from the full v2 household JSONs.

Self-contained: every helper is copied directly from
``../04_varying_gallery_length_distractors/varying_gallery.py`` so this script
has no cross-folder dependency.  See section comments for which functions are
copied verbatim from 04 and which are this file's contribution.

Differs from 04 in three ways:

1. **No train/test split.** Every household in the v2 corpora is a test
   candidate. Training has moved to the curated corpus in 05_annotated_train/,
   so the v2 data is now used exclusively for evaluation.
2. **No resampling.** One case per valid (household, identity, query_mac) triple
   for non-distractor scenarios; one case per household for distractor scenarios.
3. **Hard negatives are off by default.** Galleries contain exactly the
   household member(s) — `gallery_size = family_size`. The realistic Wyze
   deployment has only registered household members in the gallery; hard
   negatives are a stress-test configuration.

Output: 10 JSONs in ``benchmarks/``, schema-compatible with 04's benchmarks so
``run_test.sh`` works after only swapping ``TEST_FOLDER``.

Usage:
    python prepare_test.py
    python prepare_test.py --pad_hard_negatives True   # stress-test variant
"""

import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
# I/O helpers — copied verbatim from 04/varying_gallery.py
# ═══════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════
# Household-info traversal helpers — copied verbatim from 04/varying_gallery.py
# ═══════════════════════════════════════════════════════════════════════

def merge_query_gallery_pools(hh_dict, household_ids):
    """Return a copy of hh_dict where query and gallery pools are merged."""
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
# DINOv2 embeddings + hard-negative mining — copied verbatim from 04/varying_gallery.py
# Only used when --pad_hard_negatives True.
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


class HardNegativeMiner:
    """Find hard negative gallery images using pre-computed embeddings."""

    def __init__(self, embeddings, pool_paths, metadata, top_k=200,
                 batch_size=256, device=None):
        self.embeddings = embeddings
        self.pool_paths = pool_paths
        self.pool_matrix = torch.stack([embeddings[p] for p in pool_paths])
        self.pool_hh = [metadata[p][0] for p in pool_paths]
        self.pool_id = [metadata[p][1] for p in pool_paths]
        self.path_to_idx = {p: i for i, p in enumerate(pool_paths)}

        N = len(pool_paths)
        self._top_k = min(top_k, N)
        if device is None:
            device = ('cuda' if torch.cuda.is_available() and N > 50_000 else 'cpu')

        print(f"    Precomputing top-{self._top_k} neighbors for {N} images on {device}...")
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

        if len(candidates) <= n:
            return candidates
        return random.sample(candidates, n)


# ═══════════════════════════════════════════════════════════════════════
# Case builders — copied verbatim from 04/varying_gallery.py
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
                            ok = False
                            break
                    else:
                        if not any(s.get('gallery', [])
                                   for m, s in id_macs.items() if m != qmac):
                            ok = False
                            break
                if ok:
                    triples.append((hh_id, qid, qmac))
    return triples


def build_nondistr_case(hh_dict, hh_id, query_id, query_mac, camera,
                        miner=None, max_gallery=5):
    """Build one non-distractor case.

    Gallery = 1 image per household member + (optional) hard negatives padded
    up to a random target size in [family_size, max_gallery]. When miner=None
    no padding is performed and gallery_size = family_size.
    """
    identities = hh_dict[hh_id]

    q_imgs = identities[query_id].get(query_mac, {}).get('query', [])
    if not q_imgs:
        return None
    q_img = random.choice(q_imgs)

    gallery = []  # list of (tag, image_path)
    for id_id, id_macs in identities.items():
        if camera == 'same':
            g_imgs = id_macs.get(query_mac, {}).get('gallery', [])
        else:
            g_imgs = [img for mac, splits in id_macs.items()
                      if mac != query_mac
                      for img in splits.get('gallery', [])]
        if not g_imgs:
            return None
        g_candidates = [g for g in g_imgs if g != q_img]
        if not g_candidates:
            return None
        chosen = random.choice(g_candidates)
        gallery.append((id_id, chosen))

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
    """Singleton distractor: gallery = up to ``max_gallery`` images of the sole
    resident, selected by similarity to the stranger query when embeddings
    are available; otherwise the first ``target_size`` resident images."""
    if not targets:
        return None
    hh_id, hh_dict = random.choice(targets)
    id_id = list(hh_dict[hh_id].keys())[0]

    all_g = [img for splits in hh_dict[hh_id][id_id].values()
             for img in splits.get('gallery', [])]
    if not all_g:
        return None

    valid = [s for s in stranger_pool if s[0] != hh_id]
    if not valid:
        return None
    _, _, q_path = random.choice(valid)

    target_size = random.randint(1, max_gallery)
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


def _build_family_distractor(targets, stranger_pool, miner, max_gallery=5):
    """Family distractor: gallery = 1 image per member + (optional) hard
    negatives similar to the stranger query, up to ``max_gallery``."""
    if not targets:
        return None
    hh_id, hh_dict = random.choice(targets)

    gallery = []
    for id_id, macs in hh_dict[hh_id].items():
        all_g = [img for splits in macs.values() for img in splits.get('gallery', [])]
        if all_g:
            chosen = random.choice(all_g)
            gallery.append(chosen)
    if not gallery:
        return None

    valid = [s for s in stranger_pool if s[0] != hh_id]
    if not valid:
        return None
    stranger_hh, _, q_path = random.choice(valid)

    target_size = random.randint(len(gallery), max(len(gallery), max_gallery))
    n_pad = max(0, target_size - len(gallery))
    if n_pad > 0 and miner is not None:
        hard_negs = miner.find_hard_negatives(q_path, {hh_id, stranger_hh}, n_pad)
        gallery.extend(hard_negs)

    random.shuffle(gallery)
    return {'query': q_path, 'gallery': gallery, 'label': -1}


def add_household_ids(cases, metadata):
    """Add household_id and identity_id for query and each gallery image."""
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
# Subset wrappers — this file's contribution.
# Difference from 04: no resampling, no per-identity quotas. We iterate every
# valid triple (non-distractor) or every qualifying household (distractor) and
# build exactly one case per element.
# ═══════════════════════════════════════════════════════════════════════

def build_nondistr_subset(hh_dict, hh_type, camera, miner, max_gallery, min_cases=150):
    """One case per (household, identity, query_mac) triple, plus resampling
    with replacement until ~min_cases is reached.

    Splits with very few valid triples (e.g. ``cropped_sameclothes_family_samecamera``
    has 19 triples) yield unstable accuracy estimates with one case per triple.
    To stabilise evaluation, after the initial pass we draw additional triples
    uniformly at random and rebuild — ``build_nondistr_case``'s internal
    randomness (random query image, random per-identity gallery image, random
    shuffle for label position) produces a fresh case each call, so resampling
    multiplies effective coverage even when the triple set is small.

    A budget cap prevents the loop from running forever when the triple set is
    so saturated that build_nondistr_case keeps returning ``None``.
    """
    triples = find_valid_triples(hh_dict, list(hh_dict.keys()), hh_type, camera)
    if not triples:
        return [], {}

    cases = []
    id_ct = defaultdict(int)

    # Pass 1: one case per triple — every valid triple is represented at least once.
    for hh_id, query_id, query_mac in triples:
        c = build_nondistr_case(
            hh_dict, hh_id, query_id, query_mac, camera,
            miner=miner, max_gallery=max_gallery,
        )
        if c is not None:
            cases.append(c)
            id_ct[query_id] += 1

    # Pass 2: resample with replacement until we hit min_cases (or budget runs out).
    if cases and len(cases) < min_cases:
        budget = (min_cases - len(cases)) * 5
        while len(cases) < min_cases and budget > 0:
            budget -= 1
            hh_id, query_id, query_mac = random.choice(triples)
            c = build_nondistr_case(
                hh_dict, hh_id, query_id, query_mac, camera,
                miner=miner, max_gallery=max_gallery,
            )
            if c is not None:
                cases.append(c)
                id_ct[query_id] += 1

    return cases, dict(id_ct)


def build_distractor_subset(hh_same, hh_cross, hh_type, miner, embeddings, max_gallery):
    """One case per qualifying household, paired with one randomly-chosen stranger."""
    all_household_ids = set(hh_same.keys()) | set(hh_cross.keys())
    stranger_pool = collect_stranger_pool(hh_same, hh_cross, all_household_ids)

    cases = []
    id_ct = defaultdict(int)
    for hh_dict in [hh_same, hh_cross]:
        for hh_id, identities in hh_dict.items():
            n = len(identities)
            if hh_type == 'singleton' and n != 1:
                continue
            if hh_type == 'family' and n < 2:
                continue

            target = (hh_id, hh_dict)
            if hh_type == 'singleton':
                c = _build_singleton_distractor(
                    [target], stranger_pool, embeddings, max_gallery=max_gallery,
                )
            else:
                c = _build_family_distractor(
                    [target], stranger_pool, miner, max_gallery=max_gallery,
                )
            if c is not None:
                cases.append(c)
                id_ct[hh_id] += 1
    return cases, dict(id_ct)


# ═══════════════════════════════════════════════════════════════════════
# Output assembly
# ═══════════════════════════════════════════════════════════════════════

def write_benchmark(scenario, cases, id_ct, metadata, out_dir):
    """Enrich cases with household/identity ids and write one benchmark JSON."""
    add_household_ids(cases, metadata)

    gallery_size_dist = defaultdict(int)
    for c in cases:
        gallery_size_dist[len(c['gallery'])] += 1

    payload = {
        'scenario': scenario,
        'n_identities': len(id_ct),
        'n_cases': len(cases),
        'gallery_size_distribution': dict(sorted(gallery_size_dist.items())),
        'eval_id_count': id_ct,
        'eval_cases': cases,
    }
    save_json(payload, os.path.join(out_dir, f'{scenario}.json'))
    print(f'  wrote {scenario}: '
          f'n_cases={payload["n_cases"]}, '
          f'n_identities={payload["n_identities"]}, '
          f'gallery_sizes={payload["gallery_size_distribution"]}')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--same_clothes',
        default='../04_varying_gallery_length_distractors/household_info_v2_same_clothes.json',
    )
    parser.add_argument(
        '--cross_clothes',
        default='../04_varying_gallery_length_distractors/household_info_v2_cross_clothes.json',
    )
    parser.add_argument('--output_dir', default='benchmarks')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--pad_hard_negatives',
        type=lambda x: x.lower() == 'true',
        default=False,
        help='If True, pad galleries up to --max_gallery with DINOv2-mined hard '
             'negatives (stress-test variant). Default False = realistic.',
    )
    parser.add_argument('--max_gallery', type=int, default=5,
                        help='Max gallery size when --pad_hard_negatives True; '
                             'also caps the singleton-distractor gallery size.')
    parser.add_argument('--cache_dir', default='cached_embeddings',
                        help='DINOv2 embedding cache (only used when '
                             '--pad_hard_negatives True).')
    parser.add_argument('--min_cases', type=int, default=150,
                        help='Resample non-distractor splits with replacement '
                             'until they reach this case count, to stabilise '
                             'accuracy estimates on splits with few valid '
                             'triples. Set to 0 to disable resampling.')
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading household info from:')
    print(f'  {args.same_clothes}')
    print(f'  {args.cross_clothes}')
    hh_same = load_json(args.same_clothes)
    hh_cross = load_json(args.cross_clothes)
    print(f'  same_clothes: {len(hh_same)} households')
    print(f'  cross_clothes: {len(hh_cross)} households')

    metadata = build_image_metadata(hh_same, hh_cross)

    if args.pad_hard_negatives:
        print('\n--pad_hard_negatives True: computing/loading DINOv2 embeddings…')
        all_paths = collect_all_image_paths(hh_same, hh_cross)
        embeddings = load_or_compute_embeddings(all_paths, cache_dir=args.cache_dir)
        all_household_ids = set(hh_same.keys()) | set(hh_cross.keys())
        pool_same = merge_query_gallery_pools(hh_same, all_household_ids)
        pool_cross = merge_query_gallery_pools(hh_cross, all_household_ids)
        search_pool = []
        for hh_dict in [pool_same, pool_cross]:
            for hh_id, identities in hh_dict.items():
                for id_id, macs in identities.items():
                    for splits in macs.values():
                        for img in splits.get('gallery', []):
                            search_pool.append(img)
        # Deduplicate while preserving order so HardNegativeMiner gets clean input.
        seen = set()
        search_pool = [p for p in search_pool if not (p in seen or seen.add(p))]
        miner = HardNegativeMiner(embeddings, search_pool, metadata)
    else:
        embeddings = {}
        miner = None
        print('\nHard negatives DISABLED (gallery = household members only).')

    print(f'\nBuilding 8 non-distractor scenarios into {args.output_dir}/')
    for clothes in ['sameclothes', 'crossclothes']:
        hh_dict = hh_same if clothes == 'sameclothes' else hh_cross
        for hh_type in ['singleton', 'family']:
            for camera in ['samecamera', 'crosscamera']:
                cam = 'same' if camera == 'samecamera' else 'cross'
                cases, id_ct = build_nondistr_subset(
                    hh_dict, hh_type, cam, miner, args.max_gallery,
                    min_cases=args.min_cases,
                )
                scenario = f'cropped_{clothes}_{hh_type}_{camera}'
                write_benchmark(scenario, cases, id_ct, metadata, args.output_dir)

    print(f'\nBuilding 2 distractor scenarios into {args.output_dir}/')
    for hh_type in ['singleton', 'family']:
        cases, id_ct = build_distractor_subset(
            hh_same, hh_cross, hh_type, miner, embeddings, args.max_gallery,
        )
        scenario = f'distractor_cropped_{hh_type}'
        write_benchmark(scenario, cases, id_ct, metadata, args.output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
