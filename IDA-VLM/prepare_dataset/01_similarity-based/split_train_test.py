import json
import random

def parse_identity(path):
    # parse the identity from the path
    return '_'.join(path.split('/')[-1].split('_')[:-4])

def format_split(query, gallery):
    data = {
        'gallery': gallery,
        'query': query,
        'identity': {}
    }

    for idx, g in enumerate(gallery):
        identity = parse_identity(g)
        if identity not in data['identity']:
            data['identity'][identity] = {
                'query_idx': [],
                'gallery_idx': []
            }
        data['identity'][identity]['gallery_idx'].append(idx)

    for idx, q in enumerate(query):
        identity = parse_identity(q)
        if identity not in data['identity']:
            raise ValueError(f'Identity {identity} in query not found.')
        data['identity'][identity]['query_idx'].append(idx)

    return data


if __name__ == "__main__":

    # file = '/home/tian.liu/data/wyze_person_v2/cross_clothes/cross_clothes.json'
    # dataset = 'wyze_person_v2_cross_clothes'
    # N = 400

    file = '/home/tian.liu/data/wyze_person_v2/same_clothes/same_clothes.json'
    dataset = 'wyze_person_v2_same_clothes'
    N = 200

    with open(file, 'r') as f:
        data = json.load(f)

    gallery = data['gallery']
    queries = data['queries']

    print('length of gallery:', len(gallery))
    print('length of queries:', len(queries))

    # count the unique identities in the gallery
    identity = set()
    for item in gallery:
        identity.add(parse_identity(item))
    print('number of unique identities in gallery:', len(identity))

    # count the unique identities in the queries
    identity = set()
    for item, _ in queries.items():
        identity.add(parse_identity(item))
    print('number of unique identities in queries:', len(identity))

    # split identities into train and test
    # set a random seed for reproducibility
    identity = list(identity)
    random.seed(42)
    random.shuffle(identity)

    train_id = identity[:N]
    test_id = identity[N:]
    print('number of train identities:', len(train_id))
    print('number of test identities:', len(test_id))

    # get the gallery and query for train and test identities
    gallery_train = []
    gallery_test = []
    query_train = []
    query_test = []

    for item in gallery:
        if parse_identity(item) in train_id:
            gallery_train.append(item)
        else:
            gallery_test.append(item)

    for q, idx in queries.items():
        if parse_identity(q) in train_id:
            query_train.append(q)
        else:
            query_test.append(q)

    print('number of train query:', len(query_train))
    print('number of train gallery:', len(gallery_train))
    print('number of test query:', len(query_test))
    print('number of test gallery:', len(gallery_test))

    # format the train and test splits
    train_split = format_split(query_train, gallery_train)
    test_split = format_split(query_test, gallery_test)

    # write out the train and test splits to json files
    with open(f'../dataset/{dataset}/train_split.json', 'w') as f:
        json.dump(train_split, f, indent=4)
    with open(f'../dataset/{dataset}/test_split.json', 'w') as f:
        json.dump(test_split, f, indent=4)

    print('done!')


