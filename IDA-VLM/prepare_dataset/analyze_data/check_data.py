import json
from collections import defaultdict

# read in json files, compare the overlapping in identities betwee their gallery and query sets

def parse_id(filename):
    return '_'.join(filename.split('/')[-1].split('_')[:-4])

def get_gallery_id(gallery):
    res = set()
    for item in gallery:
        res.add(parse_id(item))
    return res

def get_query_id(query):
    res = set()
    for item, _ in query.items():
        res.add(parse_id(item))
    return res

def get_household_id(gallery):
    res = defaultdict(dict)
    for item in gallery:
        mac_addr = item.split('/')[-1].split('_')[-4]
        id = parse_id(item)
        cluster_id = id.split('_')[-1]
        household_id = '_'.join(id.split('_')[:-1])

        res[household_id] = res.get(household_id, dict())
        res[household_id]['mac_addr'] = res[household_id].get('mac_addr', set())
        res[household_id]['mac_addr'].add(mac_addr)
        res[household_id]['cluster_ids'] = res[household_id].get('cluster_ids', set())
        res[household_id]['cluster_ids'].add(cluster_id)

    # add a mac_ct and cluster_ct for each household
    for household_id in res:
        res[household_id]['mac_ct'] = len(res[household_id]['mac_addr'])
        res[household_id]['cluster_ct'] = len(res[household_id]['cluster_ids'])

    # sort the res by cluster_ct, then mac_ct
    res = dict(sorted(res.items(), key=lambda x: (x[1]['cluster_ct'], x[1]['mac_ct']), reverse=True))

    # convert the set to list for json serialization
    for key in res:
        res[key]['mac_addr'] = list(res[key]['mac_addr'])
        res[key]['cluster_ids'] = list(res[key]['cluster_ids'])

    return res

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":

    file_0 = ""
    file_1 = "/home/tian.liu/tian_data/wyze_person_v2/cross_clothes/cross_clothes.json"
    file_2 = "/home/tian.liu/tian_data/wyze_person_v2/same_clothes/same_clothes.json"

    with open(file_1, 'r') as f:
        data_1 = json.load(f)

    with open(file_2, 'r') as f:
        data_2 = json.load(f)

    # Extract identities from both datasets
    id_gallery_1 = get_gallery_id(data_1['gallery'])
    id_query_1 = get_query_id(data_1['queries'])
    id_gallery_2 = get_gallery_id(data_2['gallery'])
    id_query_2 = get_query_id(data_2['queries'])


    # Find overlapping identities between gallery and query sets of both datasets
    print(f"Dataset 1 - Gallery identities: {len(id_gallery_1)}, Query identities: {len(id_query_1)}")
    print(f"Dataset 2 - Gallery identities: {len(id_gallery_2)}, Query identities: {len(id_query_2)}")

    identities_1 = id_gallery_1.union(id_query_1)
    identities_2 = id_gallery_2.union(id_query_2)
    print(f"Total unique identities in Dataset 1: {len(identities_1)}")
    print(f"Total unique identities in Dataset 2: {len(identities_2)}")

    overlapping_identities = identities_1.intersection(identities_2)
    print(f"Overlapping identities: {len(overlapping_identities)}")

    # check how many households are there in two datasets
    household_1 = get_household_id(data_1['gallery'])
    household_2 = get_household_id(data_2['gallery'])
    print(f"Dataset 1 - Households: {len(household_1)}")
    print(f"Dataset 2 - Households: {len(household_2)}")

    household_query_1 = get_household_id(data_1['queries'].keys())
    household_query_2 = get_household_id(data_2['queries'].keys())
    print(f"Dataset 1 - Query Households: {len(household_query_1)}")
    print(f"Dataset 2 - Query Households: {len(household_query_2)}")

    # save household information to json file
    save_json(household_1, 'household_1.json')
    save_json(household_2, 'household_2.json')
    save_json(household_query_1, 'household_query_1.json')
    save_json(household_query_2, 'household_query_2.json')

    # collect households in the gallery with cluster_ct > 1
    for ct in range(1, 6):
        multi_cluster_household_1 = {k: v for k, v in household_1.items() if v['cluster_ct'] > ct}
        print(f"Dataset 1 - Households with more than {ct} clusters: {len(multi_cluster_household_1)}")

    for ct in range(1, 6):
        multi_cluster_household_2 = {k: v for k, v in household_2.items() if v['cluster_ct'] > ct}
        print(f"Dataset 2 - Households with more than {ct} clusters: {len(multi_cluster_household_2)}")
