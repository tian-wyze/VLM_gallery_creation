import json
import os
import random
from collections import defaultdict
from tqdm import tqdm


def parse_id(filename):
    return '_'.join(filename.split('/')[-1].split('_')[:-4])

def parse_household_id(identity_id):
    return '_'.join(identity_id.split('_')[:-1])

def parse_mac_addr(item):
    return item.split('_')[-4]

def parse_event_id(item):
    return item.split('_')[-3]


def save_json(data, filename, verbose=False):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    if verbose:
        print(f"Data saved to {filename}")


if __name__ == "__main__":

    random.seed(42)

    data_folder = "/home/tian.liu/tian_data/wyze_person_v2/annotated_identities"

    # 1. List every file in the folder.
    all_files = sorted(os.listdir(data_folder))
    print(f"Total entries in {data_folder}: {len(all_files)}")

    # 2. Keep only .jpg files; flag anything else so the user sees it.
    filenames = [f for f in all_files if f.endswith(".jpg")]
    non_jpg = [f for f in all_files if not f.endswith(".jpg")]
    if non_jpg:
        print(f"WARNING: skipping {len(non_jpg)} non-.jpg entries: {non_jpg}")
    print(f"Keeping {len(filenames)} .jpg files")

    # 3. Parse household / identity / mac / event from each filename and
    #    populate res[household_id][identity_id][mac_addr] = [paths...].
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    event_set = set()

    for fname in tqdm(filenames, desc="Parsing"):
        identity_id = parse_id(fname)
        household_id = parse_household_id(identity_id)
        mac_addr = parse_mac_addr(fname)
        event_set.add(parse_event_id(fname))
        res[household_id][identity_id][mac_addr].append(os.path.join(data_folder, fname))

    # Convert nested defaultdicts → plain dicts so it serialises cleanly.
    res_plain = {
        hid: {iid: dict(macs) for iid, macs in ids.items()}
        for hid, ids in res.items()
    }
    save_json(res_plain, "annotated_household_info.json", verbose=True)

    # 4. Stats.
    total_households = len(res_plain)
    total_identities = sum(len(res_plain[hid]) for hid in res_plain)
    total_events = len(event_set)
    print()
    print(f"Total households: {total_households}")
    print(f"Total identities: {total_identities}")
    print(f"Total events:     {total_events}")

    # Distribution of family size (# identities per household). This is the
    # "varying gallery length" — singleton households contribute gallery=1
    # eval cases, family households of size N contribute gallery=N cases.
    family_size_dist = defaultdict(int)
    for hid in res_plain:
        family_size_dist[len(res_plain[hid])] += 1

    print()
    print("Family size distribution (# identities per household → # households):")
    for size in sorted(family_size_dist.keys()):
        print(f"  size = {size:2d} : {family_size_dist[size]} households")

    # 5. Filter out households that overlap with the 04_* eval splits.
    # The 04_varying_gallery_length_distractors benchmarks are our held-out
    # eval set; any household_id that appears there must be removed from the
    # training corpus to keep the train/eval boundary household-disjoint
    # (matches the rule split_train_test_households enforces *inside* 04_*).
    eval_paths = [
        "/home/tian.liu/IDA-VLM/prepare_dataset/04_varying_gallery_length_distractors/household_info_v2_same_clothes.json",
        "/home/tian.liu/IDA-VLM/prepare_dataset/04_varying_gallery_length_distractors/household_info_v2_cross_clothes.json",
    ]
    eval_household_ids = set()
    for p in eval_paths:
        with open(p) as f:
            eval_household_ids |= set(json.load(f).keys())
    print()
    print(f"Loaded {len(eval_household_ids)} eval-set households from 04_* "
          f"(union of same_clothes + cross_clothes).")

    overlap = eval_household_ids & set(res_plain.keys())
    res_filtered = {hid: ids for hid, ids in res_plain.items() if hid not in overlap}
    save_json(res_filtered, "annotated_household_info_filtered.json", verbose=True)

    print(f"Dropped {len(overlap)} overlapping households "
          f"({len(res_plain)} → {len(res_filtered)}).")
    filt_total_identities = sum(len(res_filtered[hid]) for hid in res_filtered)
    print(f"Filtered totals: {len(res_filtered)} households, "
          f"{filt_total_identities} identities "
          f"(dropped {total_identities - filt_total_identities} identities).")

    # Family-size distribution after filtering, for reference.
    filt_family_size_dist = defaultdict(int)
    for hid in res_filtered:
        filt_family_size_dist[len(res_filtered[hid])] += 1
    print()
    print("Family size distribution after filtering:")
    for size in sorted(filt_family_size_dist.keys()):
        print(f"  size = {size:2d} : {filt_family_size_dist[size]} households")
