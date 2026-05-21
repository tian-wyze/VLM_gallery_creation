import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import sys
import json
import os
import logging
# from model import load_model_and_tokenizer
import torchvision.transforms as transforms
import random
from wyze_embedding import load_person_model
from wyze_embedding import WyzeEmbeddingExtractor

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, folder, transform=None):
        self.image_paths = image_paths
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(os.path.join(self.folder, img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def cal_dinov2_embedding(img_paths, folder):

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = ImagePathDataset(img_paths, folder, preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False)

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = model.to(device)

    features = []
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            features.append(model(batch))
    features = torch.cat(features, 0)
    # normalize features
    features = features / features.norm(dim=1, keepdim=True)

    return features


def extract_PLIP_embedding(img_paths, folder, plip_model, batch_size=128):
    """Extract PLIP embeddings for a list of image paths."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.357, 0.323, 0.328), (0.252, 0.242, 0.239))
    ])

    dataset = ImagePathDataset(img_paths, folder, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    plip_model.eval()
    plip_model = plip_model.to(device).float()

    features = []
    for batch in tqdm(dataloader):
        batch = batch.to(device).float()
        with torch.no_grad():
            feat, *_ = plip_model(batch)
            features.append(feat)

    features = torch.cat(features, dim=0)
    features = features / features.norm(dim=1, keepdim=True)
    return features


def extract_embedding(img_paths, folder, extractor, batch_size=128):
    """Extract person embeddings using WyzeEmbeddingExtractor in batches."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_paths = [os.path.join(folder, p) for p in img_paths]

    all_embeddings = []
    for i in tqdm(range(0, len(full_paths), batch_size)):
        batch_paths = full_paths[i:i + batch_size]
        batch_embeddings = extractor.extract_embeddings_batch(batch_paths)  # (B, 512), numpy
        all_embeddings.append(torch.from_numpy(batch_embeddings))

    embeddings = torch.cat(all_embeddings, dim=0).to(device)  # (N, 512)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings


def load_cases(path):
    """Load cases from a JSONL file or a legacy benchmark JSON.

    Accepted shapes:
      - JSONL (preferred, output of prepare_jsonl.py): one record per line
        carrying `stranger_letter_pos` and `answer_letter`. The embedding
        evaluation produces letter predictions for these cases.
      - JSON dict with key 'eval_cases' (legacy, output of prepare_test.py):
        cases lack letter fields and are evaluated in the digit/-1 format.
    """
    if path.endswith('.jsonl'):
        cases = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases
    with open(path) as f:
        return json.load(f)['eval_cases']


def encode_paths(img_paths, model, folder, wyze_variant=None):
    """Encode a list of image paths through the chosen embedding model.

    Returns a single [N, D] tensor in the same order as `img_paths`.
    The per-case similarity logic in main looks up embeddings by path, so we
    only need to encode each unique image once per run — callers should pass
    a deduplicated path list.
    """
    if model.startswith('DINOv2'):
        return cal_dinov2_embedding(img_paths, folder)

    elif model.startswith('WYZE_embedding'):
        extractor = load_person_model(size=wyze_variant)
        print(f'Loaded Wyze Embedding Model: {wyze_variant}')
        return extract_embedding(img_paths, folder, extractor)

    elif model == "PLIP":
        sys.path.insert(0, '/home/tian.liu/VLMID')
        from experts.PLIP.visual_model import Image_encoder_ModifiedResNet
        model_path = "model_ckpts/PLIP/PLIP_RN50.pth.tar"

        plip_model = Image_encoder_ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, input_resolution=[256, 128], width=64)
        plip_model.load_state_dict(
            torch.load(model_path, map_location='cuda')['ImgEncoder_state_dict'],
            strict=True
        )
        print(f'loaded PLIP pretrained Model: {model_path}')
        return extract_PLIP_embedding(img_paths, folder, plip_model)

    else:
        raise NotImplementedError(f'Model {model} not supported yet!')



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate embedding model on person ReID benchmark')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to a benchmark file: .jsonl (preferred, '
                             'lettered-options format) or .json (legacy '
                             '{eval_cases: [...]} digit/-1 format).')
    # parser.add_argument('--data_folder', type=str, required=True, help='Path to the image data folder')
    parser.add_argument('--model', type=str, default='WYZE_embedding', choices=['DINOv2', 'WYZE_embedding', 'PLIP'], help='Embedding model to use')
    parser.add_argument('--wyze_variant', type=str, default='v03_23_token',
                       choices=['50k', 'v02_02_reid', 'v03_23_token', 'v04_15_token'],
                       help='Wyze embedding model variant (only used when --model=WYZE_embedding)')
    parser.add_argument('--stranger_threshold', type=float, default=0.5,
                       help='If the max cosine similarity between the query and the gallery '
                            'embeddings is below this threshold, predict -1 (stranger) instead '
                            'of the argmax gallery index.')
    args = parser.parse_args()

    test_file = args.test_file
    # data_folder = args.data_folder
    data_folder = ''

    model = args.model
    wyze_variant = args.wyze_variant
    print(f'test_file: {test_file}')
    print(f'data_folder: {data_folder}')
    print(f'model: {model}')
    if model == 'WYZE_embedding':
        print(f'wyze_variant: {wyze_variant}')

    # Load the test data. Auto-detects:
    #   - JSONL (preferred): per-line records carrying `stranger_letter_pos`
    #     and `answer_letter` for the lettered-options format.
    #   - JSON: legacy `{eval_cases: [...]}` envelope with integer labels.
    test_data = load_cases(test_file)

    model_label = f'{model}_{wyze_variant}' if model == 'WYZE_embedding' else model
    test_name = os.path.splitext(os.path.basename(test_file))[0]
    save_filename = f'results/predictions_{model_label}/predictions_{model_label}_{test_name}.csv'
    print(f'save_filename: {save_filename}')
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)

    # Collect every unique image path referenced by any case. The same image
    # frequently appears in multiple cases (the same identity recurs across
    # queries/galleries), so deduplicating avoids re-encoding it.
    unique_paths = []
    seen = set()
    for case in test_data:
        for p in [case['query']] + list(case['gallery']):
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
    print(f'Number of eval_cases: {len(test_data)}')
    print(f'Number of unique images to encode: {len(unique_paths)}')

    all_embeddings = encode_paths(unique_paths, model, data_folder, wyze_variant=wyze_variant)
    print(f'embedding shape: {all_embeddings.shape}')

    # path → row-index map so per-case similarity is a pure dict lookup.
    path_to_idx = {p: i for i, p in enumerate(unique_paths)}

    stranger_threshold = args.stranger_threshold
    print(f'stranger_threshold: {stranger_threshold} '
          f'(predictions with max cosine < {stranger_threshold} become -1)')

    with open(save_filename, 'w') as f:
        f.write('idx,label,prediction,response,query\n')

    similarities_data = []
    correct_ct = 0
    for idx, data in tqdm(enumerate(test_data)):
        query = data['query']
        gallery = data['gallery']
        is_letter = 'answer_letter' in data

        query_ebd = all_embeddings[path_to_idx[query]]
        gallery_ebd = all_embeddings[[path_to_idx[g] for g in gallery]]

        # cosine similarity between the query and each gallery embedding → pick best.
        similarity = torch.nn.functional.cosine_similarity(query_ebd, gallery_ebd)
        max_sim, most_similar_idx = torch.max(similarity, dim=0)
        max_sim = max_sim.item()
        most_similar_idx = most_similar_idx.item()

        if is_letter:
            # Lettered-options format. The option list is gallery images
            # interleaved with a stranger placeholder at `stranger_letter_pos`.
            # Below-threshold max similarity → predict the stranger letter;
            # otherwise → letter at the argmax gallery position, accounting
            # for the stranger-slot offset.
            stranger_pos = data['stranger_letter_pos']
            label = data['answer_letter']
            if max_sim < stranger_threshold:
                prediction = chr(ord('A') + stranger_pos)
            else:
                opt_idx = (most_similar_idx
                           if most_similar_idx < stranger_pos
                           else most_similar_idx + 1)
                prediction = chr(ord('A') + opt_idx)
        else:
            # Legacy digit/-1 format.
            label = data['label']
            if max_sim < stranger_threshold:
                prediction = -1
            else:
                prediction = most_similar_idx + 1

        # String-compare so it works for both formats.
        if str(prediction) == str(label):
            correct_ct += 1

        # Per-case similarity record persisted to a JSON sidecar after the
        # loop (see below). Useful for eyeballing whether one backbone is
        # over-confident vs. another — e.g. v04_15_token tends to peak
        # >0.9 on true matches while DINOv2 is closer to 0.7. Six-digit
        # rounding so very-similar gallery entries can still be compared.
        sim_list = [round(s, 6) for s in similarity.tolist()]
        case_record = {
            'idx': idx,
            'query': query,
            'gallery': list(gallery),
            'similarity': sim_list,
            'max_sim': round(max_sim, 6),
            'argmax_gallery': most_similar_idx + 1,
            'label': label,
            'prediction': prediction,
            'correct': str(prediction) == str(label),
        }
        if is_letter:
            case_record['stranger_letter_pos'] = data['stranger_letter_pos']
            case_record['answer_letter'] = data['answer_letter']
        similarities_data.append(case_record)

        # CSV-safe response field with the scores array (kept for the
        # model_inspector_options app + downstream CSV consumers).
        scores_str = ';'.join(f'{s:.6f}' for s in sim_list)
        response = (f'max={max_sim:.6f} argmax={most_similar_idx + 1} '
                    f'pred={prediction} scores=[{scores_str}]')

        with open(save_filename, 'a') as f:
            f.write(f'{idx},{label},{prediction},{response},{query}\n')

    print('Evaluation done!')
    accuracy = correct_ct / len(test_data) * 100
    print(f'Acc: {round(accuracy, 1)}')

    # Per-split similarity sidecar — sits next to the prediction CSV in the
    # same results/predictions_<model_label>/ folder so they're easy to
    # diff/grep alongside each other.
    sim_filename = os.path.join(
        os.path.dirname(save_filename),
        f'similarities_{model_label}_{test_name}.json',
    )
    sim_payload = {
        'test_file':          test_file,
        'model':              model_label,
        'stranger_threshold': stranger_threshold,
        'n_cases':            len(test_data),
        'accuracy':           round(accuracy, 2),
        'cases':              similarities_data,
    }
    with open(sim_filename, 'w') as f:
        json.dump(sim_payload, f, indent=2)
    print(f'Similarities saved to: {sim_filename}')