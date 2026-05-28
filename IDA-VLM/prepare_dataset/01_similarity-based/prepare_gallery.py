import json
import os
import random
import sys
import torch
from split_train_test import parse_identity
from tqdm import tqdm

def prepare_cases(gallery, query, identity, k, threshold, similarity):
    test_cases = []
    num_less_k = 0

    for id, info in tqdm(identity.items(), desc="Processing identities"):
        q_idx = info['query_idx'] # index of the query image for this identity
        g_idx = info['gallery_idx'] # indices of the gallery images for this identity

        for q in q_idx:
            q_path = query[q]
            g_bucket = []
            sim = similarity[q] # similarity between this query and all gallery images

            # select positive example:
            # randomly sample 1 gallery image of the same identity,
            # and put into the g_bucket, regardless of similarity.
            pos_idx = random.choice(g_idx)
            g_bucket.append((gallery[pos_idx], sim[pos_idx].item(), 1))

            # select hard negatives:
            # use a mask to filter the gallery images less than a threshold,
            # less than threshold is 0, greater than threshold is 1,
            # and the gallery images of the same identity will be marked as 0.
            mask = sim >= threshold
            mask[g_idx] = False
            neg_idx_pool = torch.where(mask)[0].tolist()

            # no enough hard negatives, skip this query
            if len(neg_idx_pool) < k - 1:
                num_less_k += 1
                continue
            neg_idx = random.sample(neg_idx_pool, k - 1)
            g_bucket.extend([(gallery[i], sim[i].item(), 0) for i in neg_idx])

            # shuffle the g_bucket to mix positive and negative examples
            random.shuffle(g_bucket)

            # get the answer, which the idx of the positive example in the g_bucket
            answer = [item[2] for item in g_bucket].index(1) + 1 # 1-based index

            test_cases.append({
                'query': q_path,
                'gallery': [item[0] for item in g_bucket],
                'similarity': [round(item[1], 4) for item in g_bucket],
                'label': [item[2] for item in g_bucket],
                'answer': answer
            })

    return test_cases, num_less_k


if __name__ == "__main__":

    # dataset = 'wyze_person_v2_cross_clothes'
    # folder = '/home/tian.liu/data/wyze_person_v2/cross_clothes'

    dataset = 'wyze_person_v2_same_clothes'
    folder = '/home/tian.liu/data/wyze_person_v2/same_clothes'

    args = sys.argv
    if len(args) > 3:
        split = args[1]
        k = int(args[2])
        threshold = float(args[3])
    else:
        print('Usage: python prepare_gallery.py <split> <k> <threshold>')
        print('Example: python prepare_gallery.py test 5 0.8')
        sys.exit(1)

        # split = 'test' # default to test split if no argument is provided

    split_file = f'../dataset/{dataset}/{split}_split.json'

    with open(split_file, 'r') as f:
        data = json.load(f)

    gallery = data['gallery']
    query = data['query']
    identity = data['identity']
    print('length of gallery:', len(gallery))
    print('length of query:', len(query))
    print('number of unique identities:', len(identity))

    # load pre-calculated similarity matrix
    similarity = torch.load(f'../dataset/{dataset}/{split}_similarity.pt')

    print()
    print(f'k: {k}, threshold: {threshold}')
    test_cases, num_less_k = prepare_cases(gallery, query, identity, k, threshold, similarity)
    print('number of prepared test cases:', len(test_cases))
    print('number of queries with less than k gallery images:', num_less_k)

    # save the test cases out to a json file
    fn = f'../dataset/{dataset}/{split}_cases_k={k}_threshold={threshold}.json'
    with open(fn, 'w') as f:
        json.dump(test_cases, f, indent=4)

    print('Test cases saved to:', fn)