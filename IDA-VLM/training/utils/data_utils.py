# Data processing utilities for training

import json
import random
from collections import defaultdict
import sys
sys.path.append('./qwen')

from data import format_data_reid
# from data import format_data_reid_yes_no

import re
from tqdm import tqdm

def id_adjacent_shuffle(data, object_type):
    """Shuffle data while keeping items with the same ID adjacent."""
    # Step 1: Group by the shared field (e.g., "group")
    grouped = defaultdict(list)
    for item in data:
        if object_type == 'person':
            grouped[item["query"]["source"] + str(item["query"]["person_id"])].append(item)
        else:
            grouped[item["query"][f"{object_type}_id"]].append(item)

    group_blocks = list(grouped.values())

    # Ensure that this dataset shuffling and therefore splitting is deterministic
    py_rng_state = random.getstate()
    random.seed(42)
    random.shuffle(group_blocks)
    random.setstate(py_rng_state)

    shuffled = [item for group in group_blocks for item in group]
    return shuffled


def person_caption_revise(caption):
    # A crude way to revise the captions to make them identifiable by the model

    # Use regular expressions to find cases like "A man" (or with adjectives, e.g. "A tall man") and replace with "[A man]" or "[A tall man]"
    # Handle "A man", "a man", "The man", "the man", "this man", "This man", "A lady", "the lady", etc.
    # Multiple adjectives and various capitalizations,
    # but we do NOT handle hyphenated adjectives (e.g., "long-haired woman")—these won't be matched.
    for article in ['A', 'The', 'a', 'the']:
        for noun in ['man', 'woman', 'child', 'boy', 'girl', 'lady', 'male', 'female', 'person']:
            # Match only space-separated adjectives before noun, not hyphenated (so "long haired woman" matches, "long-haired woman" does not)
            # The adjectives group: any number of adjectives, but not including hyphens in the adjective
            pattern = r'\b' + article + r' ((?:(?:\w+)\s+)*?)' + noun + r'\b'
            replacement = f'[{article} \\1{noun}]'
            caption = re.sub(pattern, replacement, caption)

    # Also handle "this man", "that woman" etc. with adjectives (but NOT hyphenated ones).
    for demonstrative in ['this', 'that', 'This', 'That']:
        for noun in ['man', 'woman', 'child', 'boy', 'girl', 'lady', 'male', 'female', 'person']:
            # Now allow adjectives with hyphens (for "long-haired woman") as well as space-separated adjectives
            # Allow any sequence of (\w+ or \w+-\w+) followed by whitespace (i.e., adjectives with or without hyphens)
            pattern = r'\b' + demonstrative + r' (((?:\w+(?:-\w+)?\s+)*?))' + noun + r'\b'
            replacement = f'[{demonstrative} \\1{noun}]'
            caption = re.sub(pattern, replacement, caption)

    # assert there is only one [ ] in the caption. pdb if not.
    if caption.count('[') != 1:
        caption = f'[Person]. {caption}'
    return caption

def load_person_captions():
    captions_dict = {}
    with open('../liang_data/mydata/rstp/data_captions.json', 'r') as f:
        captions = json.load(f)

    for img in tqdm(captions):
        revised_captions = person_caption_revise(img['captions'][0])
        captions_dict[img['img_path']] = revised_captions
    return captions_dict

def load_dataset(gallery_size, filter_threshold, object_type, captions=False):
    """Load the main dataset for training."""

    dataset = (f'../liang_data/mydata/reid_json/train_tuples_k={gallery_size}_filter={filter_threshold}.jsonl'
               if filter_threshold > 0
               else f'../liang_data/mydata/reid_json/train_tuples_k={gallery_size}.jsonl')
    with open(dataset, 'r') as f:
        tuples = [json.loads(line) for line in f.readlines()]

    tuples = [tuple for tuple in tuples if tuple['query']['source'] == 'rstp']

    # update the image paths in the tuples to my path
    for tuple in tuples:
        tuple['query']['img_path'] = tuple['query']['img_path'].replace('/home/liang_shi/work/', '/home/tian.liu/liang_data/')
        for i in range(gallery_size):
            tuple['gallery'][i]['img_path'] = tuple['gallery'][i]['img_path'].replace('/home/liang_shi/work/', '/home/tian.liu/liang_data/')

    if captions:
        captions_dict = load_person_captions()
    else:
        captions_dict = None
    tuples = id_adjacent_shuffle(tuples, object_type)
    tuples = [format_data_reid(tuple, object_type=object_type, captions_dict=captions_dict) for tuple in tuples]

    train_dataset = tuples[:int(len(tuples) * 0.9)]
    eval_dataset = tuples[int(len(tuples) * 0.9):]
    random.shuffle(train_dataset)

    print("Dataset ready.")
    return train_dataset, eval_dataset

def load_wyze_dataset(args, object_type, captions=False):
    """Load the main dataset for evaluation/test.

    Accepts either:
      - JSONL (preferred, output of prepare_jsonl.py): one record per line
        carrying `stranger_letter_pos` and `answer_letter`.
      - JSON benchmark file (legacy, output of prepare_test.py): dict with
        key 'eval_cases'. Cases lack letter fields and will fall back to
        the digit/-1 prompt format in format_data_reid.

    Format is detected from the file extension on `args.test_file`.
    """
    is_jsonl = args.test_file.endswith('.jsonl')
    if is_jsonl:
        test_data = []
        with open(args.test_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    test_data.append(json.loads(line))
    else:
        with open(args.test_file) as f:
            test_data = json.load(f)['eval_cases']

    ## format the test_data to match the tuples format for later processing.

    def format_data(test_data, data_path):
        tuples = []
        for item in test_data:
            query = {
                'img_path': data_path + item['query'],
            }
            gallery = []
            for gallery_item in item['gallery']:
                gallery.append({
                    'img_path': data_path + gallery_item,
                })

            tup = {
                'query': query,
                'gallery': gallery,
                'answer': item['label'],
            }
            # Carry through letter-format fields when present so
            # format_data_reid takes the lettered-options path.
            if 'answer_letter' in item:
                tup['answer_letter'] = item['answer_letter']
                tup['stranger_letter_pos'] = item['stranger_letter_pos']
            tuples.append(tup)

        return tuples

    tuples = format_data(test_data, args.data_folder)
    captions_dict = None

    # tuples = id_adjacent_shuffle(tuples, object_type)
    tuples = [format_data_reid(tuple, object_type=object_type, captions_dict=captions_dict) for tuple in tuples]

    # train_dataset = tuples[:int(len(tuples) * 0.9)]
    # eval_dataset = tuples[int(len(tuples) * 0.9):]
    # random.shuffle(train_dataset)
    eval_dataset = tuples

    print("Dataset ready.")
    print(f"Number of eval samples: {len(eval_dataset)}")
    # return train_dataset, eval_dataset
    return None, eval_dataset

def load_wyze_train_dataset(cfg, object_type, captions=False):
    """Load the main dataset for training.

    Accepts either:
      - JSONL (preferred, output of prepare_jsonl.py): one record per line
        carrying `stranger_letter_pos` and `answer_letter` so the prompt
        formatter can emit the lettered-options layout.
      - JSON: legacy array-of-cases format from prepare_train.py with
        integer `label` only (the prompt formatter falls back to the
        "Gallery 1: / Gallery 2: ..." layout with digit/-1 answers).

    Format is detected from the file extension on `cfg.train_file`.
    """
    is_jsonl = cfg.train_file.endswith('.jsonl')
    if is_jsonl:
        train_data = []
        with open(cfg.train_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    train_data.append(json.loads(line))
    else:
        with open(cfg.train_file) as f:
            train_data = json.load(f)

    ## format the train_data to match the tuples format for later processing.

    def format_data(data, data_folder):
        tuples = []
        for item in data:
            query = {
                'img_path': data_folder + item['query'],
            }
            gallery = []
            for gallery_item in item['gallery']:
                gallery.append({
                    'img_path': data_folder + gallery_item,
                })

            tup = {
                'query': query,
                'gallery': gallery,
                'answer': item['label'],
            }
            # Carry through letter-format fields when present so
            # format_data_reid can take the letter path.
            if 'answer_letter' in item:
                tup['answer_letter'] = item['answer_letter']
                tup['stranger_letter_pos'] = item['stranger_letter_pos']
            tuples.append(tup)

        return tuples

    tuples = format_data(train_data, cfg.data_folder)
    captions_dict = None

    # tuples = id_adjacent_shuffle(tuples, object_type)
    tuples = [format_data_reid(tuple, object_type=object_type, captions_dict=captions_dict) for tuple in tuples]
    # Debug print to check the first tuple after formatting
    print(f"First formatted tuple:\n{tuples[0]}")

    random.shuffle(tuples)
    # train_dataset = tuples
    train_dataset = tuples[:int(len(tuples) * 0.95)]
    eval_dataset = tuples[int(len(tuples) * 0.95):]

    print("Dataset ready.")
    print(f"Number of train samples: {len(train_dataset)}")
    print(f"Number of eval samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset



def load_pet_or_face_captions(object_type):
    if object_type == 'pet':
        caption_path = './prompts/pet.txt'
    elif object_type == 'face':
        caption_path = './prompts/vggface.txt'
    with open(caption_path, 'r') as f:
        captions = f.readlines()
        captions_dict = {line.split(': ')[0]: line.split(': ')[1] for line in captions}
        return captions_dict

def load_pet_dataset(gallery_size, filter, object_type, captions):
    tuples = []
    captions_dict = load_pet_or_face_captions(object_type) if captions else None
    with open(f'./mydata/reid_json/train_pet_sep_k={gallery_size}_filter={filter}.jsonl', 'r') as f:
        tuples += [json.loads(line) for line in f.readlines()]
        if captions:
            tuples = [tuple for tuple in tuples if '/'.join(tuple['query']['img_path'].split('/')[-4:]) in captions_dict]


    tuples = id_adjacent_shuffle(tuples, object_type)
    tuples = [format_data_reid(tuple, object_type=object_type, captions_dict=captions_dict) for tuple in tuples]

    train_dataset = tuples[:int(len(tuples) * 0.9)]
    eval_dataset = tuples[int(len(tuples) * 0.9):]

    random.seed(42)
    random.shuffle(train_dataset)
    random.shuffle(eval_dataset)
    eval_dataset = eval_dataset[:1000]

    return train_dataset, eval_dataset


def load_face_dataset(gallery_size, filter, object_type, captions):
    tuples = []
    captions_dict = load_pet_or_face_captions(object_type) if captions else None
    with open(f'./mydata/vggface/train_tuples_filter={filter}.jsonl', 'r') as f:
        tuples += [json.loads(line) for line in f.readlines()]
        if captions:
            tuples = [tuple for tuple in tuples if '/'.join(tuple['query'].split('/')[-3:]) in captions_dict] if captions else tuples

    train_dataset = [format_data_reid(tuple, object_type=object_type, captions_dict=captions_dict) for tuple in tuples]
    # tuples = []
    with open(f'./mydata/vggface/val_tuples_filter={filter}.jsonl', 'r') as f:
        tuples += [json.loads(line) for line in f.readlines()]
        if captions:
            tuples = [tuple for tuple in tuples if '/'.join(tuple['query'].split('/')[-3:]) in captions_dict]
    eval_dataset = [format_data_reid(tuple, object_type=object_type, captions_dict=captions_dict) for tuple in tuples]
    random.seed(42)
    random.shuffle(train_dataset)
    random.shuffle(eval_dataset)
    eval_dataset = eval_dataset[:1000]
    return train_dataset, eval_dataset

def load_buildings_dataset(gallery_size, filter, object_type, captions):
    tuples = []
    with open(f'./mydata/oxbuildings/paris_tuples.jsonl', 'r') as f:
        tuples += [json.loads(line) for line in f.readlines()]
    train_dataset = [format_data_reid(tuple, object_type=object_type, captions_dict=None) for tuple in tuples]
    random.seed(42)
    random.shuffle(train_dataset)
    tuples = []
    with open(f'./mydata/oxbuildings/ox_tuples.jsonl', 'r') as f:
        tuples += [json.loads(line) for line in f.readlines()]
    eval_dataset = [format_data_reid(tuple, object_type=object_type, captions_dict=None) for tuple in tuples]
    random.seed(42)
    random.shuffle(eval_dataset)
    return train_dataset, eval_dataset

def load_vehicle_dataset(gallery_size, filter, object_type, captions):
    tuples = []
    with open(f'./mydata/vehicle/VeRi/train_tuples_similarity_threshold_0.8.jsonl', 'r') as f:
        tuples += [json.loads(line) for line in f.readlines()]
    train_dataset = [format_data_reid(tuple, object_type=object_type, captions_dict=None) for tuple in tuples]
    random.seed(42)
    random.shuffle(train_dataset)
    tuples = []
    with open(f'./mydata/vehicle/VeRi/test_tuples_similarity_threshold_0.8.jsonl', 'r') as f:
        tuples += [json.loads(line) for line in f.readlines()]
    eval_dataset = [format_data_reid(tuple, object_type=object_type, captions_dict=None) for tuple in tuples]
    random.seed(42)
    random.shuffle(eval_dataset)
    eval_dataset = eval_dataset[:1000]
    return train_dataset, eval_dataset


def load_sop_captions():
    SOP_OBJECT_TYPES = ['table', 'bicycle', 'cabinet', 'chair', 'coffee_maker', 'fan', 'kettle', 'lamp', 'mug', 'sofa', 'stapler', 'toaster']
    base_path = './prompts'
    all_captions = []
    for object_type in SOP_OBJECT_TYPES:
        with open(f'{base_path}/{object_type}_final.txt', 'r') as f:
            captions = f.readlines()
            all_captions.extend(captions)
    files = [line.split(': ')[0] for line in all_captions]
    captions = [line.split(': ')[1] for line in all_captions]
    captions_dict = {files[i]: captions[i] for i in range(len(files))}
    return captions_dict

def load_sop_dataset(gallery_size, filter, object_type, captions):
    """Load Stanford Online Products dataset."""
    SOP_OBJECT_TYPES = ['table', 'bicycle', 'cabinet', 'chair', 'coffee_maker', 'fan', 'kettle', 'lamp', 'mug', 'sofa', 'stapler', 'toaster']
    captions_dict = load_sop_captions() if captions else None
    # Load training data
    tuples = []
    for obj_type in SOP_OBJECT_TYPES:
        with open(f'./mydata/Stanford_Online_Products/tuples/train/{obj_type}_k={gallery_size}_f={filter}.jsonl', 'r') as f:
            tuples += [json.loads(line) for line in f.readlines()]
    train_dataset = [format_data_reid(tuple, object_type=object_type, captions_dict=captions_dict) for tuple in tuples]
    random.seed(42)
    random.shuffle(train_dataset)

    # Load evaluation data
    tuples = []
    for obj_type in SOP_OBJECT_TYPES:
        with open(f'./mydata/Stanford_Online_Products/tuples/test/{obj_type}_k={gallery_size}_f={filter}.jsonl', 'r') as f:
            tuples += [json.loads(line) for line in f.readlines()]
    eval_dataset = [format_data_reid(tuple, object_type=object_type, captions_dict=captions_dict) for tuple in tuples]
    random.seed(42)
    random.shuffle(eval_dataset)
    eval_dataset = eval_dataset[:1000]

    print("Dataset ready.")
    return train_dataset, eval_dataset

def load_sop_yes_no_dataset(gallery_size, filter, object_type, captions):
    # Load training data
    tuples = []

    with open(f'./mydata/Stanford_Online_Products/tuples/train/new_tuples_k=1_f=0.5_hard_positives.jsonl', 'r') as f:
        tuples += [json.loads(line) for line in f.readlines()]
    train_dataset = [format_data_reid_yes_no(tuple) for tuple in tuples]
    random.seed(42)
    random.shuffle(train_dataset)

    # Load evaluation data
    tuples = []
    with open(f'./mydata/Stanford_Online_Products/tuples/test/new_tuples_k=1_f=0.5_hard_positives.jsonl', 'r') as f:
        tuples += [json.loads(line) for line in f.readlines()]
    eval_dataset = [format_data_reid_yes_no(tuple) for tuple in tuples]
    random.seed(42)
    random.shuffle(eval_dataset)
    eval_dataset = eval_dataset[:1000]

    print("Dataset ready.")
    return train_dataset, eval_dataset


def is_person(tuple):
    return tuple['query'].split('/')[-2] in ['ciin', 'denisdang', 'khanhvy', 'oong', 'phuc-map', 'thao', 'thuytien', 'viruss', 'willinvietnam', 'yuheng']

def load_ym_dataset(dataset='myvlm', train_or_test='test'):
    if train_or_test == 'train':
        with open('./mydata/ym/tuples_yollava_train_10.jsonl', 'r') as f:
            train_dataset = [json.loads(line) for line in f.readlines()]
        train_dataset = [format_data_reid_yes_no(tuple) for tuple in train_dataset]
        with open('./mydata/ym/tuples_yollava.jsonl', 'r') as f:
            eval_dataset = [json.loads(line) for line in f.readlines()]
            eval_dataset = eval_dataset[:100]
        eval_dataset = [format_data_reid_yes_no(tuple) for tuple in eval_dataset]
        return train_dataset, eval_dataset
    else:
        if dataset == 'myvlm':
            with open('./mydata/ym/tuples_myvlm.jsonl', 'r') as f:
                tuples = [json.loads(line) for line in f.readlines()]
        elif dataset == 'yollava':
            with open('./mydata/ym/tuples_yollava.jsonl', 'r') as f:
                tuples = [json.loads(line) for line in f.readlines()]
                # tuples = [tuple for tuple in tuples if not is_person(tuple)]

        random.shuffle(tuples)
    print("Dataset ready.")
    return tuples


def load_unified_dataset():

    unified_training_dataset = []
    unified_eval_dataset = []

    train_dataset, eval_dataset = load_dataset(gallery_size=5, filter_threshold=0.5, object_type='person', captions=False)
    unified_training_dataset.extend(train_dataset[:20000])
    unified_eval_dataset.extend(eval_dataset[:100])
    print("Person dataset ready.")
    train_dataset, eval_dataset = load_pet_dataset(gallery_size=5, filter=0.5, object_type='pet', captions=False)
    unified_training_dataset.extend(train_dataset[:20000])
    unified_eval_dataset.extend(eval_dataset[:100])
    print("Pet dataset ready.")
    train_dataset, eval_dataset = load_face_dataset(gallery_size=5, filter=0.1, object_type='face', captions=False)
    unified_training_dataset.extend(train_dataset[:20000])
    unified_eval_dataset.extend(eval_dataset[:100])
    print("Face dataset ready.")
    train_dataset, eval_dataset = load_buildings_dataset(gallery_size=5, filter=0.5, object_type='building', captions=False)
    unified_training_dataset.extend(train_dataset[:20000])
    unified_eval_dataset.extend(eval_dataset[:100])
    print("Building dataset ready.")
    train_dataset, eval_dataset = load_vehicle_dataset(gallery_size=5, filter=0.5, object_type='vehicle', captions=False)
    unified_training_dataset.extend(train_dataset[:20000])
    unified_eval_dataset.extend(eval_dataset[:100])
    print("Vehicle dataset ready.")
    train_dataset, eval_dataset = load_sop_dataset(gallery_size=5, filter=0.5, object_type='sop', captions=False)
    unified_training_dataset.extend(train_dataset[:20000])
    unified_eval_dataset.extend(eval_dataset[:100])
    print("SOP dataset ready.")
    random.shuffle(unified_training_dataset)
    random.shuffle(unified_eval_dataset)

    return unified_training_dataset, unified_eval_dataset
