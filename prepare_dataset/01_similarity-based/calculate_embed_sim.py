import json
import os
import sys
import torch
import torchvision.transforms as transforms
import random
from PIL import Image
from tqdm import tqdm


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


def load_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model

def cal_embedding(img_paths, folder):

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


if __name__ == "__main__":

    # dataset = 'wyze_person_v2_cross_clothes'
    # folder = '/home/tian.liu/data/wyze_person_v2/cross_clothes'

    dataset = 'wyze_person_v2_same_clothes'
    folder = '/home/tian.liu/data/wyze_person_v2/same_clothes'

    args = sys.argv
    if len(args) > 1:
        split = args[1]
    else:
        # give instructions on how to run the script, and exit
        # print('Please provide the split to process (train or test).')
        # print('Example: python calculate_embed_sim.py train')
        # sys.exit(1)

        split = 'test' # default to test split if no argument is provided

    split_file = f'../dataset/{dataset}/{split}_split.json'

    with open(split_file, 'r') as f:
        data = json.load(f)

    gallery = data['gallery']
    query = data['query']
    identity = data['identity']

    gallery_embedding = cal_embedding(gallery, folder)
    print('gallery embedding shape:', gallery_embedding.shape)
    query_embedding = cal_embedding(query, folder)
    print('query embedding shape:', query_embedding.shape)

    # calculate the similarity matrix for the gallery and query
    similarity = query_embedding @ gallery_embedding.t() # (num_query, num_gallery)
    print('similarity shape:', similarity.shape)

    # save the similarity matrix
    torch.save(similarity, f'../dataset/{dataset}/{split}_similarity.pt')