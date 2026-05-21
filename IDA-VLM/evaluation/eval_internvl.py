import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import sys
import json
import os
import logging
from model import load_model_and_tokenizer

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Background color for pad2square: ImageNet mean scaled to [0, 255]
_PAD2SQUARE_BG = tuple(int(x * 255) for x in IMAGENET_MEAN)

def expand2square(pil_img: Image.Image, background_color=_PAD2SQUARE_BG) -> Image.Image:
    """
    Pad *pil_img* to a square canvas using *background_color*.

    This is the official InternVL approach for preserving aspect ratio
    before resizing to the model's input size.  The background is filled
    with the ImageNet mean colour so the padded area is "neutral" after
    normalisation.
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def build_transform_DA(
    input_size: int = 448,
    pad2square: bool = False,
):
    """
    ImageNet-normalized transform pipeline for square tiles.

    Parameters
    ----------
    input_size : int
        Target side length in pixels (default 448).
    pad2square : bool
        When *True*, pad non-square inputs to a square canvas filled with
        the ImageNet mean colour **before** resizing.  This preserves the
        original aspect ratio (official InternVL SFT flag).
    """
    steps = [
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    ]
    if pad2square:
        steps.append(T.Lambda(lambda img: expand2square(img, _PAD2SQUARE_BG)))
    steps += [
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(steps)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



def load_model(path):

    if path == "OpenGVLab/InternVL3_5-8B": # official model
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    else: # load custom checkpoint
        model, tokenizer, model_cfg = load_model_and_tokenizer(path)

    return model, tokenizer

def load_single_tile_image(image_file, data_folder):

    """Load and dynamically tile a single image for InternVL3."""
    input_size = 448
    pad2square = True
    use_thumbnail = False
    max_num = 1

    image = Image.open(os.path.join(data_folder, image_file)).convert("RGB")
    transform = build_transform_DA(input_size=input_size, pad2square=pad2square)

    tiles = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num
    )
    tile_tensors = torch.stack([transform(t) for t in tiles])

    metadata = {
        "image_size": image.size,
        "max_tiles": max_num,
        "use_thumbnail": use_thumbnail,
        "tiles": tile_tensors.shape[0],
        # "load_time": load_time,
        "cache_hit": False,
    }

    # return tile_tensors, [tile_tensors.shape[0]], metadata
    return tile_tensors



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate InternVL model on person ReID benchmark')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test JSON file')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the image data folder')
    parser.add_argument('--model_name', type=str, default='InternVL3_5-8B',
                        choices=['InternVL3-8B', 'InternVL3_5-8B', 'InternVL3_8B_DAfinetuned'],
                        help='Model to evaluate')
    parser.add_argument('--prompt_file', type=str, default='prompt.txt',
                        help='Path to the prompt file')
    args = parser.parse_args()

    test_file = args.test_file
    data_folder = args.data_folder
    print(f'test_file: {test_file}')
    print(f'data_folder: {data_folder}')
    print(f'model_name: {args.model_name}')
    print(f'prompt_file: {args.prompt_file}')

    # load the test data
    with open(test_file) as f:
        test_data = json.load(f)['eval_cases']

    if args.model_name == 'InternVL3_8B_DAfinetuned':
        # case = 'InternVL3_8B_DAfinetuned'
        case = 'InternVL3_8B_DAfinetuned_single_tile_no_thumbnail'

        model_path = '/home/tian.liu/IDA-VLM/evaluation/model_ckpts/inhouse_DA_model/internvl3_8b_llm_lora_mlp_full_vqa_annotated_qa_v2_260318_ckpt800'

    elif args.model_name == 'InternVL3-8B':
        case = 'InternVL3-8B'
        model_path = 'OpenGVLab/InternVL3-8B'

    elif args.model_name == 'InternVL3_5-8B':
        case = 'InternVL3_5-8B'
        model_path = 'OpenGVLab/InternVL3_5-8B'

    else:
        raise NotImplementedError(f'Model {args.model_name} not supported!')


    save_filename = (f'results/{case}_result_' + test_file.split('/')[-1]).replace('json', 'csv')
    print(f'save_filename: {save_filename}')

    # load prompt
    with open(args.prompt_file) as f:
        prompt_template = f.read()

    model, tokenizer = load_model(model_path)
    print('Model loaded successfully!')

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # question = 'Hi how are you?'
    # response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    # print(f'User: {question}\nAssistant: {response}')

    with open(save_filename, 'w') as f:
        f.write(f'idx,label,prediction,response,query\n')

    correct_ct = 0
    for idx, data in tqdm(enumerate(test_data)):
        query = data['query']
        gallery = data['gallery']
        label = data['label']
        # print(f'query: {query}')
        # print(f'label: {label}')

        # multi-image multi-round conversation, separate images
        if case == 'InternVL3_8B_DAfinetuned_single_tile_no_thumbnail':
            # use single tile, resized images to 448x448, and no thumbnail
            query_value = load_single_tile_image(query, data_folder).to(torch.bfloat16).cuda()
            gallery_values = []
            for i in range(5):
                gallery_values.append(load_single_tile_image(gallery[i], data_folder).to(torch.bfloat16).cuda())

        else: # use original multi-tile approach
            query_value = load_image(os.path.join(data_folder, query), max_num=12).to(torch.bfloat16).cuda()
            gallery_values = []
            for i in range(5):
                gallery_values.append(load_image(os.path.join(data_folder, gallery[i]), max_num=12).to(torch.bfloat16).cuda())

        pixel_values = torch.cat([query_value] + gallery_values, dim=0)
        num_patches_list = [query_value.size(0)] + [v.size(0) for v in gallery_values]

        question = prompt_template
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        # prediction = response.strip()
        # take the prediction from the X in the response. The person in the query image matches the person in gallery image X.
        prediction = response.split('in gallery image ')[-1].strip().split('.')[0]

        if prediction == str(label):
            correct_ct += 1
        # print(f'User: {question}\nAssistant: {response}')
        # exit()

        # write out the label and prediction
        with open(save_filename, 'a') as f:
            f.write(f'{idx},{label},{prediction},{response},{query}\n')

    print('Evaluation done!')

    accuracy = correct_ct / len(test_data) * 100
    print(f'Case: {case}, Acc: {round(accuracy, 1)}')