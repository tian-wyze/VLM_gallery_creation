import json
import random
import os
import matplotlib.pyplot as plt
from PIL import Image

FONTSIZE = 18

def visualize_case():

    sampled_idx = random.randint(0, len(test_cases) - 1)
    test_case = test_cases[sampled_idx]

    query_img = Image.open(os.path.join(folder, test_case['query']))
    gallery_imgs = [Image.open(os.path.join(folder, img)) for img in test_case['gallery']]
    similarity = test_case['similarity']
    label = test_case['label']

    # plot a figure with 1 row and 6 columns, the first column is the query image, and the rest are gallery images
    # put the similarity and label (positive or negative) under each gallery image
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    axes[0].imshow(query_img)
    axes[0].set_title('Query', fontsize=FONTSIZE)
    axes[0].axis('off')
    for i in range(5):
        axes[i+1].imshow(gallery_imgs[i])
        color = 'green' if label[i] == 1 else 'red'
        axes[i+1].set_title(f'Similarity: {similarity[i]:.4f}\n{"Positive" if label[i] == 1 else "Negative"}', fontsize=FONTSIZE, color=color)
        axes[i+1].axis('off')
    plt.tight_layout()

    save_folder = f'../dataset/{dataset}/examples_k={k}_thd={threshold}'
    os.makedirs(save_folder, exist_ok=True)
    save_path = f'{save_folder}/example_{sampled_idx}.png'

    plt.savefig(save_path)
    print(f'Visualization saved to: {save_path}')

if __name__ == "__main__":

    dataset = 'wyze_person_v2_cross_clothes'
    folder = '/home/tian.liu/data/wyze_person_v2/cross_clothes'

    # dataset = 'wyze_person_v2_same_clothes'
    # folder = '/home/tian.liu/data/wyze_person_v2/same_clothes'

    k = 5
    threshold = 0.5
    num_examples = 10

    # load the test cases
    file = f'../dataset/{dataset}/test_cases_k={k}_threshold={threshold}.json'
    with open(file, 'r') as f:
        test_cases = json.load(f)
    print('number of test cases:', len(test_cases))

    # choose a random test case to visualize
    for i in range(num_examples):
        visualize_case()