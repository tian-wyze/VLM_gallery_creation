"""
Prepare training data from the curated annotated corpus
(annotated_household_info_filtered.json) for VLM person re-identification.

Compared with 04_varying_gallery_length_distractors/varying_gallery.py:
- No train/test split — the filtered corpus is already eval-disjoint at the
  household level (see 05_annotated_train/README.md).
- No same/cross-clothes axis — the annotated corpus carries a single
  clothes_id (000) for every image. Cross-camera sampling implicitly picks
  up different-outfit pairs when the same person appears under different
  cameras on different days.
- Reduced per-identity sampling rates (this corpus is ~10x larger than 04).

Output schema matches 04: every case has
  {query, gallery, label, similarity, household_id, identity_id,
   query_household_id, query_identity_id}
where label > 0 is the 1-indexed match position and label = -1 is a stranger.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
# I/O helpers
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
# DINOv2 embedding computation & caching
# (copied from 04_varying_gallery_length_distractors/varying_gallery.py)
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
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, drop_last=False)

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
    return {path: features[i] for i, path in enumerate(image_paths)}


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
# (copied from 04_varying_gallery_length_distractors/varying_gallery.py)
# ═══════════════════════════════════════════════════════════════════════

class HardNegativeMiner:
    """Find hard-negative gallery images using pre-computed embeddings.

    On init, precomputes top-K nearest neighbors for every image in the pool
    via batched matrix multiplication. find_hard_negatives() is then a cheap
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
        self.pool_matrix = torch.stack([embeddings[p] for p in pool_paths])
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
        pool_dev = (self.pool_matrix.to(device) if device != 'cpu'
                    else self.pool_matrix)
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

    def get_similarities(self, query_path, gallery_paths):
        """Cosine similarities between query and each gallery image."""
        q = self.embeddings[query_path]
        return [round(torch.dot(q, self.embeddings[g]).item(), 4)
                for g in gallery_paths]


# ═══════════════════════════════════════════════════════════════════════
# Diversity tracker + case builders
# (copied from 04_varying_gallery_length_distractors/varying_gallery.py)
# ═══════════════════════════════════════════════════════════════════════

class UsageTracker:
    """Track per-identity image usage to promote diversity across cases."""

    def __init__(self):
        self._used = defaultdict(set)

    def pick(self, identity_id, candidates):
        """Pick one image, preferring those not yet used for this identity."""
        unused = [c for c in candidates if c not in self._used[identity_id]]
        chosen = random.choice(unused) if unused else random.choice(candidates)
        self._used[identity_id].add(chosen)
        return chosen


def build_nondistr_case(hh_dict, hh_id, query_id, query_mac, camera,
                        miner=None, max_gallery=5, tracker=None):
    """Build one non-distractor case.

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
        g_candidates = [g for g in g_imgs if g != q_img]
        if not g_candidates:
            return None
        chosen = (tracker.pick(id_id, g_candidates) if tracker
                  else random.choice(g_candidates))
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


def collect_stranger_pool(hh_dict, household_ids):
    """Collect (household_id, identity_id, image_path) for stranger queries."""
    pool = []
    for hh_id in household_ids:
        if hh_id not in hh_dict:
            continue
        for id_id, macs in hh_dict[hh_id].items():
            for splits in macs.values():
                for img in splits.get('query', []):
                    pool.append((hh_id, id_id, img))
    return pool


def build_singleton_distractor(targets, stranger_pool, embeddings, max_gallery=5):
    """Singleton distractor: gallery = up to max_gallery images of the sole
    resident, selected by similarity to the stranger query."""
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


def build_family_distractor(targets, stranger_pool, miner,
                            max_gallery=5, tracker=None):
    """Family distractor: gallery = 1 image per member + hard negatives
    similar to the stranger query, up to max_gallery."""
    if not targets:
        return None
    hh_id, hh_dict = random.choice(targets)

    gallery = []
    for id_id, macs in hh_dict[hh_id].items():
        all_g = [img for splits in macs.values()
                 for img in splits.get('gallery', [])]
        if all_g:
            chosen = (tracker.pick(id_id, all_g) if tracker
                      else random.choice(all_g))
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


# ═══════════════════════════════════════════════════════════════════════
# Similarity scores + household-id annotation
# (copied from 04_varying_gallery_length_distractors/varying_gallery.py)
# ═══════════════════════════════════════════════════════════════════════

def add_similarity_scores(cases, miner):
    """Add DINOv2 cosine similarity between query and each gallery image."""
    for case in cases:
        case['similarity'] = miner.get_similarities(case['query'], case['gallery'])


def add_household_ids(cases, metadata):
    """Annotate each case with household_id and identity_id for query and
    every gallery image. Padded hard-negative images carry their source
    household/identity, so member vs. distractor entries are distinguishable.
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
# Format adapter
# ═══════════════════════════════════════════════════════════════════════

def wrap_flat_corpus(hh_dict):
    """Adapt 05's flat-list format to 04's {query, gallery} format.

    Input :  {hh: {iid: {mac: [paths]}}}
    Output:  {hh: {iid: {mac: {'query': paths, 'gallery': paths}}}}

    Both keys point to the same list — this matches the merged-pool form 04
    uses for its training pipeline (merge_query_gallery_pools).
    """
    out = {}
    for hh, ids in hh_dict.items():
        out[hh] = {
            iid: {mac: {'query': paths, 'gallery': paths}
                  for mac, paths in macs.items()}
            for iid, macs in ids.items()
        }
    return out


def collect_all_image_paths(hh_dict):
    paths = set()
    for ids in hh_dict.values():
        for macs in ids.values():
            for splits in macs.values():
                paths.update(splits['query'])  # gallery is the same list
    return sorted(paths)


def build_image_metadata(hh_dict):
    meta = {}
    for hh, ids in hh_dict.items():
        for iid, macs in ids.items():
            for splits in macs.values():
                for img in splits['query']:
                    meta[img] = (hh, iid)
    return meta


# ═══════════════════════════════════════════════════════════════════════
# Training case construction
# ═══════════════════════════════════════════════════════════════════════

def build_training_cases(hh_dict, sampling_rates, miner, embeddings,
                         distractor_fraction=0.2, max_gallery=6):
    """Build training cases from the merged-pool corpus.

    Camera scenario is randomized per case: identities lacking the chosen
    scenario yield None and the retry loop tries another camera/MAC.
    """
    non_distr = []
    stats = defaultdict(int)
    tracker = UsageTracker()

    sorted_hids = sorted(hh_dict.keys())
    for hh_id in tqdm(sorted_hids, desc='  Non-distractor'):
        identities = hh_dict[hh_id]
        fam_size = len(identities)
        n_per_id = sampling_rates.get(fam_size, max(sampling_rates.values()))

        for qid in identities:
            valid_macs = [m for m, s in identities[qid].items()
                          if s.get('query') or s.get('gallery')]
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
                    stats[f'{camera}cam_size{fam_size}'] += 1
                    built += 1
                    if built >= n_per_id:
                        break

    print(f"  Non-distractor cases built: {len(non_distr)}")

    # ── Distractors: distractor_fraction × non-distractor count, stratified
    #    by family size. Same logic as 04's training distractor block.
    train_ids = list(hh_dict.keys())
    stranger_pool = collect_stranger_pool(hh_dict, train_ids)
    by_size = defaultdict(list)
    for hh_id in train_ids:
        by_size[len(hh_dict[hh_id])].append((hh_id, hh_dict))

    n_distr = int(len(non_distr) * distractor_fraction)
    sizes = sorted(by_size.keys())
    per_size = n_distr // len(sizes) if sizes else 0
    extra = n_distr % len(sizes) if sizes else 0
    distr = []

    print(f"  Building {n_distr} distractor cases across "
          f"{len(sizes)} size buckets...")
    for i, sz in enumerate(sizes):
        target = per_size + (1 if i >= len(sizes) - extra else 0)
        built = 0
        for _ in range(target * 5):
            if sz == 1:
                c = build_singleton_distractor(
                    by_size[sz], stranger_pool, embeddings, max_gallery)
            else:
                c = build_family_distractor(
                    by_size[sz], stranger_pool, miner, max_gallery, tracker)
            if c:
                distr.append(c)
                stats[f'distractor_size{sz}'] += 1
                built += 1
                if built >= target:
                    break
        print(f"    size {sz}: {built}/{target} cases "
              f"({len(by_size[sz])} households available)")

    cases = non_distr + distr
    random.shuffle(cases)
    return cases, stats


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data from the annotated corpus.')
    parser.add_argument('--input',
                        default='annotated_household_info_filtered.json')
    parser.add_argument('--output', default='train_data.json')
    parser.add_argument('--cache_dir', default='cached_embeddings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_gallery', type=int, default=6,
                        help='Max gallery size (members + hard negatives)')
    parser.add_argument('--distractor_fraction', type=float, default=0.4)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.input}...")
    raw = load_json(args.input)
    hh_dict = wrap_flat_corpus(raw)
    n_hh = len(hh_dict)
    n_id = sum(len(ids) for ids in hh_dict.values())
    print(f"  {n_hh} households, {n_id} identities")

    print("\nPreparing DINOv2 embeddings...")
    all_paths = collect_all_image_paths(hh_dict)
    print(f"  {len(all_paths)} unique images")
    embeddings = load_or_compute_embeddings(all_paths, args.cache_dir)
    metadata = build_image_metadata(hh_dict)

    print("\nBuilding hard-negative miner...")
    miner = HardNegativeMiner(embeddings, all_paths, metadata)
    print(f"  Pool size: {len(all_paths)} images")

    # Roughly half of the rates one would extrapolate from 04 after accounting
    # for the ~10x larger identity pool. Singletons (53.5% of households) get
    # the lightest weight; family size 2 (the most common multi-person case)
    # gets a moderate boost; rare large families get more cases per identity
    # so their (rare) coverage does not collapse.
    sampling_rates = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 6}

    print("\nBuilding training cases...")
    cases, stats = build_training_cases(
        hh_dict, sampling_rates,
        miner=miner, embeddings=embeddings,
        distractor_fraction=args.distractor_fraction,
        max_gallery=args.max_gallery,
    )

    print("\n  Adding similarity scores and household IDs...")
    add_similarity_scores(cases, miner)
    add_household_ids(cases, metadata)
    save_json(cases, args.output)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Training Data")
    print(f"{'='*60}")
    for cat in sorted(stats.keys()):
        print(f"  {cat:30s}  {stats[cat]} cases")
    print(f"  {'─'*60}")
    print(f"  Total training cases: {len(cases)}")

    gal = Counter(len(c['gallery']) for c in cases)
    print("\n  Gallery-length distribution:")
    for gl in sorted(gal.keys()):
        print(f"    gallery size {gl}: {gal[gl]} cases")


if __name__ == '__main__':
    main()
