import json
import random
import os
import matplotlib.pyplot as plt
from PIL import Image

FONTSIZE = 18

def plot_case(save_folder='examples', sampled_idx=-1, verbose=False):

    if sampled_idx == -1:
        return

    test_case = test_cases[sampled_idx]

    query_img = Image.open(os.path.join(folder, test_case['query']))
    gallery_imgs = [Image.open(os.path.join(folder, img)) for img in test_case['gallery']]
    label = test_case['label']

    if verbose:
        print('Query image:', test_case['query'])

    # plot a figure with 1 row and 6 columns, the first column is the query image, and the rest are gallery images
    # put the similarity and label (positive or negative) under each gallery image
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    axes[0].imshow(query_img)
    axes[0].set_title('Query', fontsize=FONTSIZE)
    axes[0].axis('off')
    for i in range(5):
        axes[i+1].imshow(gallery_imgs[i])
        color = 'green' if label-1 == i else 'red'
        axes[i+1].set_title(f'{"Positive" if label-1 == i else "Negative"}', fontsize=FONTSIZE, color=color)
        axes[i+1].axis('off')
    plt.tight_layout()

    os.makedirs(save_folder, exist_ok=True)
    save_path = f'{save_folder}/example_{sampled_idx}.png'

    plt.savefig(save_path)
    print(f'Visualization saved to: {save_path}')


def visualize_case(save_folder='examples'):

    sampled_idx = random.randint(0, len(test_cases) - 1)
    plot_case(save_folder=save_folder, sampled_idx=sampled_idx)


if __name__ == "__main__":

    ## Full frame
    # folder = '/home/tian.liu/tian_data/wyze_person_v2_cross_clothes_full_frame'

    ## Cropped person
    folder = '/home/tian.liu/tian_data/wyze_person_v2/cross_clothes'
    # folder = '/home/tian.liu/tian_data/wyze_person_v2/same_clothes'

    num_examples = 10

    # load the evaluation cases
    files = [
        'singleton_crossclothes_samecamera.json',
        'singleton_crossclothes_crosscamera.json',
        'family_crossclothes_samecamera.json',
        'family_crossclothes_crosscamera.json'
        # 'singleton_sameclothes_samecamera.json',
        # 'singleton_sameclothes_crosscamera.json',
        # 'family_sameclothes_samecamera.json',
        # 'family_sameclothes_crosscamera.json'
    ]


    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        test_cases = data['eval_cases']
        print('number of test cases:', len(test_cases))

        # visualize a single case
        case_num = None
        # case_num = [0, 1, 2]
        # case_num = [3, 4, 5]
        # case_num = [0]

        if case_num is not None:
            for idx in case_num:
                plot_case(save_folder=f'visualization/examples_{file.split(".")[0]}', sampled_idx=idx, verbose=True)
            exit()

        # choose a random test case to visualize
        for i in range(num_examples):
            visualize_case(save_folder=f'visualization/examples_{file.split(".")[0]}')