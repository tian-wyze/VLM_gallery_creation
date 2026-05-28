import json
from collections import defaultdict


def parse_id(filename):
    return '_'.join(filename.split('/')[-1].split('_')[:-4])

def parse_household_id(identity_id):
    return '_'.join(identity_id.split('_')[:-1])


def extract_ids_from_cases(cases):
    """Return (identity_ids, household_ids) sets from query and all gallery images."""
    identity_ids = set()
    household_ids = set()
    for case in cases:
        for img in [case['query']] + case['gallery']:
            identity_id = parse_id(img)
            identity_ids.add(identity_id)
            household_ids.add(parse_household_id(identity_id))
    return identity_ids, household_ids


SCENARIOS = [
    'singleton_sameclothes_samecamera',
    'singleton_sameclothes_crosscamera',
    'singleton_crossclothes_samecamera',
    'singleton_crossclothes_crosscamera',
    'family_sameclothes_samecamera',
    'family_sameclothes_crosscamera',
    'family_crossclothes_samecamera',
    'family_crossclothes_crosscamera',
]

if __name__ == '__main__':
    # Load train data
    with open('train_data.json', 'r') as f:
        train_data = json.load(f)
    train_identity_ids, train_household_ids = extract_ids_from_cases(train_data)
    print(f"Train data: {len(train_data)} cases, "
          f"{len(train_identity_ids)} unique identities, "
          f"{len(train_household_ids)} unique households\n")

    any_leakage = False
    for scenario in SCENARIOS:
        fname = f'{scenario}.json'
        try:
            with open(fname, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[SKIP] {fname} not found.")
            continue

        eval_cases = data.get('eval_cases', [])
        eval_identity_ids, eval_household_ids = extract_ids_from_cases(eval_cases)

        overlap_identities = eval_identity_ids & train_identity_ids
        overlap_households = eval_household_ids & train_household_ids

        print(f"[{scenario}]")
        print(f"  Eval cases: {len(eval_cases)}, "
              f"identities: {len(eval_identity_ids)}, "
              f"households: {len(eval_household_ids)}")

        if overlap_identities:
            any_leakage = True
            print(f"  !! IDENTITY LEAKAGE: {len(overlap_identities)} overlapping identity IDs")
            # for id_ in sorted(overlap_identities):
            #     print(f"       {id_}")
        else:
            print(f"  Identity leakage: NONE")

        if overlap_households:
            any_leakage = True
            print(f"  !! HOUSEHOLD LEAKAGE: {len(overlap_households)} overlapping household IDs")
            # for hid in sorted(overlap_households):
            #     print(f"       {hid}")
        else:
            print(f"  Household leakage: NONE")
        print()

    if any_leakage:
        print("RESULT: Leakage detected — see above.")
    else:
        print("RESULT: No leakage found across all scenarios.")
