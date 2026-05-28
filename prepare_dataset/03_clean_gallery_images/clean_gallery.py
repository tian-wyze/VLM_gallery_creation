import json
from collections import defaultdict
from multiprocessing.resource_sharer import stop
import os
from tqdm import tqdm
import random


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


def get_household_info(gallery_images, query_images, data_folder):
    res = defaultdict(dict)

    def _add(item, split):
        identity_id = parse_id(item)
        household_id = parse_household_id(identity_id)
        mac_addr = parse_mac_addr(item)
        res[household_id] = res.get(household_id, dict())
        res[household_id][identity_id] = res[household_id].get(identity_id, dict())
        if mac_addr not in res[household_id][identity_id]:
            res[household_id][identity_id][mac_addr] = {'query': [], 'gallery': []}
        res[household_id][identity_id][mac_addr][split].append(os.path.join(data_folder, item))

    for item in gallery_images:
        _add(item, 'gallery')
    for item in query_images:
        _add(item, 'query')

    return res


def build_cases(q, positive_pool, negative_pool, data_folder, add_path, n=1):
    """Build n cases for the same query with independently sampled galleries.
    Each case has 1 positive and 4 negatives, shuffled randomly.
    Returns a list of case dicts."""
    cases = []
    for _ in range(n):
        positive_example = random.choice(positive_pool)
        negative_examples = random.sample(negative_pool, 4)
        g = [(positive_example, 1)] + [(neg, 0) for neg in negative_examples]
        random.shuffle(g)
        label = g.index((positive_example, 1)) + 1
        cases.append({
            'query': data_folder + q if add_path else q,
            'gallery': [data_folder + x[0] if add_path else x[0] for x in g],
            'label': label
        })
    return cases


def prepare_household_gallery(scenario, data_folder, household_type, clothes, camera,
                               household_dict, query, household_ids, split):

    print(f'\n{scenario}, split={split}')
    eval_cases = {} # key is identity_id, value is a list of test cases for this identity_id
    total_eval_ct = 0
    id_ct = defaultdict(int)
    event_ct = defaultdict(int)

    # loop through each query image
    for q in query.keys():
        identity_id = parse_id(q)
        household_id = parse_household_id(identity_id)

        # only consider the predefined households for this split (train/test)
        if household_id not in household_ids:
            continue

        mac_addr = parse_mac_addr(q)
        event_id = parse_event_id(q)

        #---------- create positive and negative pool based on the household type and camera type
        if household_type == 'singleton':
            if camera == 'same': # same camera, i.e., same mac address
                if mac_addr not in household_dict[household_id][identity_id]:
                    # raise ValueError(f"Query image {q} has mac address {mac_addr} not found in household_dict for identity {identity_id} in household {household_id}")
                    continue
                positive_pool = household_dict[household_id][identity_id][mac_addr]['gallery']
            else: # cross camera, i.e., different camera mac address
                positive_pool = []
                for mac in household_dict[household_id][identity_id]:
                    if mac != mac_addr:
                        positive_pool += household_dict[household_id][identity_id][mac]['gallery']

            # since this is single household, negative pool is randomly sampled from
            # other households among the predefined household_ids
            # [to-do]: the negative pool can be more challenging by sampling from other households
            # using the DINOv2 feature similarity, instead of random sampling. We can leave this for future improvement.
            negative_pool = []
            for household in household_ids: ## note here we use household_ids
                if household != household_id: # from other predefined households
                    for identity in household_dict[household]:
                        for mac in household_dict[household][identity]:
                            negative_pool += household_dict[household][identity][mac]['gallery']

        elif household_type == 'family':
            if camera == 'same': # same camera mac address
                if mac_addr not in household_dict[household_id][identity_id]:
                    continue
                positive_pool = household_dict[household_id][identity_id][mac_addr]['gallery']

                # negatives are from other members in the same household with the same camera mac address
                negative_pool = []
                for identity in household_dict[household_id]:
                    if identity != identity_id:
                        if mac_addr in household_dict[household_id][identity]:
                            negative_pool += household_dict[household_id][identity][mac_addr]['gallery']

            else: # cross camera, i.e., different camera mac address
                positive_pool = []
                for mac in household_dict[household_id][identity_id]:
                    if mac != mac_addr:
                        positive_pool += household_dict[household_id][identity_id][mac]['gallery']

                # negatives are from other members in the same household with different camera mac address
                negative_pool = []
                for identity in household_dict[household_id]:
                    if identity != identity_id:
                        for mac in household_dict[household_id][identity]:
                            if mac != mac_addr:
                                negative_pool += household_dict[household_id][identity][mac]['gallery']
        else:
            raise ValueError

        #---------- prepare the eval cases
        # for test: 1 case per query; for train: n_train_cases cases with different sampled galleries
        if len(positive_pool) > 1 and len(negative_pool) > 4:
            id_ct[identity_id] += 1
            event_ct[event_id] += 1

            # note for training split, we create 3 cases per query with different randomly sampled galleries to boost training data size
            n = 1 if split == 'test' else 3

            # since training data contains images from cross and same clothes folder, we need to add path
            # for evaluation, we will provide the data_folder path separately in command line.
            # add_path = True if split == 'train' else False
            add_path = True

            new_cases = build_cases(q, positive_pool, negative_pool, data_folder, add_path, n)
            if identity_id not in eval_cases:
                eval_cases[identity_id] = []
            eval_cases[identity_id] += new_cases
            total_eval_ct += n

    print(f"Total ID: {len(id_ct)}, Cases: {total_eval_ct}, Events: {len(event_ct)}")

    # sort id_ct and event_ct by value
    id_ct = dict(sorted(id_ct.items(), key=lambda item: item[1], reverse=True))
    event_ct = dict(sorted(event_ct.items(), key=lambda item: item[1], reverse=True))
    # exit()

    # ----------- Sampling for test set
    # we do some subsampling here to create test split for quick evaluation.
    # We sample max(50, len(id_ct)) identities, and for each identity
    # we unifromly sample max(3, ct) examples from it original sorted queries.
    if split == 'test':
        sampled_id_ct = {k: v for k, v in id_ct.items() if parse_household_id(k) in household_ids}
        sampled_id_ct = dict(list(sampled_id_ct.items())[:50])

        # sample eval cases
        sampled_eval_cases = []
        sampled_household_id = set()

        for identity_id in sampled_id_ct:

            household_id = parse_household_id(identity_id)
            sampled_household_id.add(household_id)

            cases = eval_cases[identity_id]
            # sort the cases by the query filename
            cases = sorted(cases, key=lambda x: x['query'])

            # let's uniformly sample max(3, ct) examples from the original sorted queries for this identity_id
            # by taking the start, 2 middle frames if there are enough examples, otherwise just take all examples
            if len(cases) > 3:
                sampled_cases = [cases[0], cases[len(cases) // 3], cases[2 * len(cases) // 3]]
            else:
                sampled_cases = cases

            sampled_id_ct[identity_id] = f"{len(sampled_cases)}/{len(cases)}"
            sampled_eval_cases += sampled_cases
        # print(f"Sampled test cases: {len(sampled_eval_cases)}")
        # print(f"Sampled test households: {len(sampled_household_id)}")

    elif split == 'train':
        # for training set, we keep all the cases from the predefined train households without sampling
        sampled_eval_cases = []
        sampled_id_ct = id_ct
        sampled_event_ct = event_ct
        for identity_id in id_ct:
            cases = eval_cases[identity_id]
            sampled_eval_cases += cases
            sampled_id_ct[identity_id] = f"{len(cases)}/{len(cases)}"
    else:
        raise ValueError("split should be either 'train' or 'test'")

    if split == 'test':
        # calculate the sampled_event_ct
        sampled_event_ct = defaultdict(int)
        for case in sampled_eval_cases:
            event_id = parse_event_id(case['query'])
            sampled_event_ct[event_id] += 1

        res = dict()
        res['eval_id_count'] = sampled_id_ct # only the sampled identities
        res['eval_event_count'] = sampled_event_ct # only the events corresponding to the sampled identities
        res['eval_query'] = [case['query'] for case in sampled_eval_cases]
        res['eval_gallery'] = [img for case in sampled_eval_cases for img in case['gallery']]
        res['eval_cases'] = sampled_eval_cases

        save_json(res, f'benchmarks/{scenario}.json')
        # print(f"{scenario} test cases saved to {scenario}.json!")
        print(f"Sampled ID {len(sampled_id_ct)}, Cases: {len(sampled_eval_cases)}, Events: {len(sampled_event_ct)}")

    return sampled_eval_cases



def split_train_test_households(household_dict, n_family_samecam=30, n_family_crosscam=15, n_singleton=5):
    """Sample test households stratified by three tiers:
      - Tier 1 (family_samecam): family households where >=2 identities share a common MAC address.
                                  Good for family_samecamera AND family_crosscamera scenarios.
      - Tier 2 (family_crosscam): family households where members only appear on different cameras.
                                  Only contributes to family_crosscamera scenario.
      - Tier 3 (singleton):       single-identity households.
    Prioritizes tier-1 to boost family_samecamera eval case counts.
    Returns (test_household_ids, train_household_ids).
    """
    tier1, tier2, tier3 = [], [], []
    for hid, identities in household_dict.items():
        if len(identities) == 1:
            tier3.append(hid)
        else:
            # Check if any MAC address is shared by >=2 identities
            mac_to_ids = defaultdict(set)
            for identity_id, macs in identities.items():
                for mac in macs:
                    mac_to_ids[mac].add(identity_id)
            if any(len(ids) >= 2 for ids in mac_to_ids.values()):
                tier1.append(hid)
            else:
                tier2.append(hid)

    sampled_t1 = random.sample(tier1, min(n_family_samecam, len(tier1)))
    sampled_t2 = random.sample(tier2, min(n_family_crosscam, len(tier2)))
    sampled_t3 = random.sample(tier3, min(n_singleton, len(tier3)))
    print(f"Sampled Test Household: family-samecam: {len(sampled_t1)} / {len(tier1)}, "
          f"family-crosscam: {len(sampled_t2)} / {len(tier2)}, "
          f"singleton: {len(sampled_t3)} / {len(tier3)}")
    test_household_ids = set(sampled_t1) | set(sampled_t2) | set(sampled_t3)
    train_household_ids = set(household_dict.keys()) - test_household_ids
    print(f"Train households: {len(train_household_ids)}, Test households: {len(test_household_ids)}")
    return test_household_ids, train_household_ids


def filter_train_data(train_data, all_test_households):
    """Remove train cases where any image (query or any gallery image) belongs to a test household.
    No overlap in households or identities between train and test is allowed.
    """
    filtered = []
    for case in train_data:
        query_household = parse_household_id(parse_id(case['query']))
        if query_household in all_test_households:
            continue
        if any(parse_household_id(parse_id(img)) in all_test_households
               for img in case['gallery']):
            continue
        filtered.append(case)
    return filtered


def check_household_info(json_file, case, data_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)
    gallery = data['gallery']
    query = data['queries']

    # Note: verified empirically that gallery and query identities are identical
    # (perfect overlap, 0 gallery-only and 0 query-only identities in both
    # same_clothes and cross_clothes datasets). Building household_dict from
    # gallery alone is therefore sufficient for household classification.
    household_dict = get_household_info(gallery, list(query.keys()), data_folder)
    save_json(household_dict, f'household_info_{case}.json')

    # print total number of households, total number of identities, and total number of gallery/query images
    total_households = len(household_dict)
    total_identities = sum(len(household_dict[hid]) for hid in household_dict)
    total_gallery_images = sum(len(household_dict[hid][iid][mac]['gallery']) for hid in household_dict for iid in household_dict[hid] for mac in household_dict[hid][iid])
    total_query_images = sum(len(household_dict[hid][iid][mac]['query']) for hid in household_dict for iid in household_dict[hid] for mac in household_dict[hid][iid])
    print(f"Total households: {total_households}, Total identities: {total_identities}, Total gallery images: {total_gallery_images}, Total query images: {total_query_images}")

    # print number of households with only a single identity
    single_identity_households = sum(1 for hid in household_dict if len(household_dict[hid]) == 1)
    print(f"Households with only a single identity: {single_identity_households}")

    check_mac_addr(household_dict)

    return household_dict, query


def check_mac_addr(household_dict):

    identity_id_mac_count = defaultdict(set)
    for household_id in household_dict:
        for identity_id in household_dict[household_id]:
            for mac_addr in household_dict[household_id][identity_id]:
                identity_id_mac_count[identity_id].add(mac_addr)

    print(f"Total unique identity IDs: {len(identity_id_mac_count)}")
    # for identity_id, mac_addrs in identity_id_mac_count.items():
    #     if len(mac_addrs) > 1:
    #         print(f"Identity IDs with > 1 unique MAC address: {identity_id} has {len(mac_addrs)} MAC addresses")



if __name__ == "__main__":

    random.seed(42)

    # create subsets of the data based on different strategies
    clothes_type = [
        'same',
        'cross'
    ]
    house_hold_type = [
        'singleton',
        'family'
    ]
    camera_type = [
        'same',
        'cross'
    ]

    train_data = []
    all_test_households = set()
    for clothes in clothes_type:
        print()
        print("--" * 10)
        # Load household info once per clothes type (same across all 4 household/camera scenarios)
        if clothes == 'cross':
            json_file = f'/home/tian.liu/tian_data/wyze_person_v2/cross_clothes/cross_clothes.json'
            case = f'v2_cross_clothes'
            data_folder = f'/home/tian.liu/tian_data/wyze_person_v2/cross_clothes/'
        elif clothes == 'same':
            json_file = f'/home/tian.liu/tian_data/wyze_person_v2/same_clothes/same_clothes.json'
            case = f'v2_same_clothes'
            data_folder = f'/home/tian.liu/tian_data/wyze_person_v2/same_clothes/'
        else:
            raise ValueError("clothes type should be either 'cross' or 'same'")

        household_dict, query = check_household_info(json_file, case, data_folder)

        # exit()


    #     # Pre-sample test households once for this clothes type, reused across all 4 scenarios
    #     test_household_ids, train_household_ids = split_train_test_households(household_dict)
    #     # print(f"[{clothes}_clothes]: split {len(test_household_ids)} test households, {len(train_household_ids)} train households.")

    #     # exit()
    #     for household in house_hold_type:
    #         for camera in camera_type:
    #             scenario = f'cropped_{clothes}clothes_{household}_{camera}camera'

    #             test_cases = prepare_household_gallery(scenario=scenario,
    #                                                 data_folder=data_folder,
    #                                                 household_type=household,
    #                                                 clothes=clothes,
    #                                                 camera=camera,
    #                                                 household_dict=household_dict,
    #                                                 query=query,
    #                                                 household_ids=test_household_ids,
    #                                                 split='test'
    #                                                 )

    #             train_cases = prepare_household_gallery(scenario=scenario,
    #                                                 data_folder=data_folder,
    #                                                 household_type=household,
    #                                                 clothes=clothes,
    #                                                 camera=camera,
    #                                                 household_dict=household_dict,
    #                                                 query=query,
    #                                                 household_ids=train_household_ids,
    #                                                 split='train'
    #                                                 )


    #             # collect train cases from all scenarios for training set
    #             train_data += train_cases
    #             all_test_households |= test_household_ids

    # # print('All scenarios prepared eval cases!')
    # print('-----'*5)
    # print(f"Total test households across all scenarios: {len(all_test_households)}")

    # # Filter train_data to remove any case whose query or gallery image belongs to a test household
    # train_data_before = len(train_data)
    # train_data = filter_train_data(train_data, all_test_households)
    # print(f"Train data filtered: {len(train_data)} / {train_data_before} cases kept.")

    # # save the train_data out to a json file
    # save_json(train_data, f'train_data.json')