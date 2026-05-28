# SOP
# MyVLM
# YoLLaVA
import os
from glob import glob
import torch
import torchvision.transforms as transforms
import random
import json
from typing import List, Tuple, Dict
from PIL import Image
from tqdm import tqdm


SOP_OBJECT_TYPES = ['table', 'bicycle', 'cabinet', 'chair', 'coffee_maker', 'fan', 'kettle', 'lamp', 'mug', 'sofa', 'stapler', 'toaster']

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def save_tuples_jsonl(
    tuples: List[Tuple[Dict, List[Dict]]],
    out_path: str
):
    with open(out_path, 'w') as f:
        for query, gallery, answer in tuples:
            json.dump({
                "query": query,
                "gallery": gallery,
                "answer": answer,
            }, f)
            f.write('\n')


def extract_dino_features(img_paths):
    # only use the first image of each product
    img_paths = [path for path in img_paths if path.endswith('_0.JPG')]
    features = []
    
    dataset = ImagePathDataset(img_paths, preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            features.append(model(batch))
    features = torch.cat(features, 0)
    # normalize features
    features = features / features.norm(dim=1, keepdim=True)
    similarity_matrix = features @ features.t()
    return similarity_matrix

def extract_intra_instance_features(img_paths, product_ids, images_dict):
    # only use the first image of each product
    features = []
    
    dataset = ImagePathDataset(img_paths, preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            features.append(model(batch))
    features = torch.cat(features, 0)
    # calculate similarity matrix for each product
    similarity_matrix = []
    counter = 0
    for product_id in product_ids:
        product_features = features[counter:counter+len(images_dict[product_id])]
        product_features = product_features / product_features.norm(dim=1, keepdim=True)
        similarity_matrix.append(product_features @ product_features.t())
        counter += len(images_dict[product_id])
    return similarity_matrix


def filter_train_test(img_paths, train_or_test):
    new_paths = []
    with open(f'Ebay_{train_or_test}.txt', 'r') as f:
        train_img_paths = f.readlines()
        train_img_paths = train_img_paths[1:]
    train_img_paths = [os.path.basename(path.split(' ')[-1].strip()) for path in train_img_paths]
    for path in img_paths:
        if os.path.basename(path) in train_img_paths:
            new_paths.append(path)
    return new_paths

    
def create_tuples(object_type, train_or_test):

    base_paths = f'/home/liang_shi/work/mydata/Stanford_Online_Products/{object_type}_final'
    img_paths = glob(os.path.join(base_paths, '*.JPG'))
    img_paths.sort()
    img_paths = filter_train_test(img_paths, train_or_test)
    similarity_matrix = extract_dino_features(img_paths)

    product_ids = list(dict.fromkeys(os.path.basename(image).split('_')[0] for image in img_paths))
    images_dict = {product_id: [image for image in img_paths if product_id in image] for product_id in product_ids}

    tuples = []
    for j, (product_id, images) in tqdm(enumerate(images_dict.items())):
        similarities_to_product = similarity_matrix[j]
        # find products with similarity > filter
        mask = similarities_to_product > filter
        mask[j] = False
        neg_pool = [images_dict[product_ids[k]] for k in range(len(product_ids)) if mask[k]]
        neg_pool = [image for sublist in neg_pool for image in sublist]
        if len(neg_pool) < gallery_size - 1:
            continue
        for i in range(len(images)):
            query = images[i]
            pos_pool = images[:i] + images[i+1:]
            # Selecting positives
            pos_sample = random.choice(pos_pool)
            # Selecting negatives
            neg_samples = random.sample(neg_pool, gallery_size - 1)
            gallery = [pos_sample] + neg_samples
            random.shuffle(gallery)
            answer = gallery.index(pos_sample) + 1
            tuples.append((query, gallery, answer))
    return tuples



if __name__ == "__main__":
    
    gallery_size = 5
    filter = 0.2
    train_or_test = 'train'
    # Hard negatives with unfiltered positives
    
    for object_type in SOP_OBJECT_TYPES:
        tuples = create_tuples(object_type, train_or_test)
        save_tuples_jsonl(tuples, f'/home/liang_shi/work/mydata/Stanford_Online_Products/tuples/{train_or_test}/{object_type}_k={gallery_size}_f={filter}.jsonl')
    