import torch
import numpy as np
import random
import argparse
from PIL import Image, ImageDraw, ImageFont
from data import format_data_reid
from utils import process_vision_info
import json
from tqdm import tqdm
import re
from utils.model_utils import get_expert_inputs, load_clip_model_and_processor
from utils.data_utils import load_sop_dataset, id_adjacent_shuffle
from utils import load_dataset, load_sop_dataset, load_model_and_processor, load_face_dataset, load_pet_dataset, load_wyze_dataset, setup_logging
import replace
import os
import re

class InferenceConfig:
    """Configuration class for training parameters."""

    def __init__(self, args):
        self.seed = args.seed
        self.gallery_size = args.gallery_size
        # self.model_size = args.model_size
        # self.filter = args.filter
        self.batch_size = args.batch_size
        self.object_type = args.object_type
        self.input_mode = args.input_mode
        self.feature_mode = args.feature_mode
        self.expert_feature = args.expert_feature
        self.captions = args.captions
        self.model_id = args.model_id


def fetch_image(vision_info):
    """Simple image fetching function"""
    if "image" in vision_info:
        if isinstance(vision_info["image"], Image.Image):
            return vision_info["image"]
        elif isinstance(vision_info["image"], str):
            return Image.open(vision_info["image"]).convert('RGB')
        else:
            return vision_info["image"]
    elif "image_url" in vision_info:
        if isinstance(vision_info["image_url"], str):
            return Image.open(vision_info["image_url"]).convert('RGB')
        else:
            return vision_info["image_url"]
    else:
        raise ValueError("No image found in vision_info")


def extract_vision_info(conversations):
    """Extract vision information from conversations"""
    vision_infos = []
    for conversation in conversations:
        if isinstance(conversation, list):
            for message in conversation:
                if "content" in message:
                    for content in message["content"]:
                        if isinstance(content, dict) and ("image" in content or "image_url" in content):
                            vision_infos.append(content)
        elif isinstance(conversation, dict) and "content" in conversation:
            for content in conversation["content"]:
                if isinstance(content, dict) and ("image" in content or "image_url" in content):
                    vision_infos.append(content)
    return vision_infos


def add_label_to_image(img, label):
    """Add a label to an image"""
    draw = ImageDraw.Draw(img)
    base_font_size = int(img.height * 0.08)
    try:
        font = ImageFont.truetype("arial.ttf", base_font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding = 4
    x_pos = padding
    y_pos = img.height - text_height - padding - 10

    draw.rectangle([x_pos, y_pos, x_pos + text_width + 2 * padding, y_pos + text_height + 2 * padding], fill=(255, 255, 255))
    draw.text((x_pos + padding, y_pos + padding), label, fill=(0, 0, 0), font=font)
    return img


def visualize_tuple(vision_infos, model_answer=None, correct_label=None, object_type=None):
    """
    Create visualization of query and gallery images, preserving all content.
    For person datasets, crop without padding to a 1:2 ratio (224x448), keeping the
    upper region intact; for other datasets, crop to square and resize to 224x224.
    Frames indicate predictions.
    """
    from PIL import Image, ImageDraw

    def crop_to_square(im):
        """Center-crop image to square by trimming the longer side."""
        w, h = im.size
        if w == h:
            return im
        dim = min(w, h)
        if w > h:
            left = (w - dim) // 2
            top = 0
            right = left + dim
            bottom = h
        else:
            left = 0
            top = (h - dim) // 2
            right = w
            bottom = top + dim
        return im.crop((left, top, right, bottom))

    def crop_person_ratio(im, target_width, target_height):
        """Crop to 1:2 (person) ratio prioritizing upper content, then resize."""
        w, h = im.size
        if w == 0 or h == 0:
            return Image.new('RGB', (target_width, target_height), (255, 255, 255))

        target_ratio = target_width / target_height
        current_ratio = w / h

        if current_ratio > target_ratio:
            new_w = max(int(round(h * target_ratio)), 1)
            left = max((w - new_w) // 2, 0)
            right = left + new_w
            top = 0
            bottom = h
        else:
            new_h = max(int(round(w / target_ratio)), 1)
            if new_h > h:
                new_h = h
            top = 0  # keep the top portion
            bottom = top + new_h
            left = 0
            right = w

        cropped = im.crop((left, top, right, bottom))
        return cropped.resize((target_width, target_height), Image.BICUBIC)

    # Load images and apply dataset-specific preprocessing
    images = [Image.open(info['image']).convert('RGB') for info in vision_infos]
    if object_type and object_type.lower() == "person":
        target_width, target_height = 224, 448
        processed_images = [
            crop_person_ratio(img, target_width, target_height)
            for img in images
        ]
    else:
        target_width = target_height = 224
        processed_images = [
            crop_to_square(img).resize((target_width, target_height), Image.BICUBIC)
            for img in images
        ]

    output_images = []
    # Query image: add a black frame (thick)
    query_img = processed_images[0].copy()
    draw = ImageDraw.Draw(query_img)
    frame_width = 8
    frame_color = (0, 0, 0)
    W, H = query_img.size
    for i in range(frame_width):
        draw.rectangle([i, i, W-1-i, H-1-i], outline=frame_color)
    output_images.append(query_img)

    # Gallery images: no label, add red frame for incorrect predictions and green frame for correct
    for idx, img in enumerate(processed_images[1:]):
        gallery_img = img.copy()
        gallery_idx = idx + 1  # images[1:] means gallery indices start at 1

        # Draw thick red frame for incorrect predictions and green frame for correct predictions
        if model_answer is not None and correct_label is not None:
            if gallery_idx == model_answer:
                frame_width = 8
                if model_answer == correct_label:
                    # Green frame for correct prediction
                    frame_color = (0, 255, 0)
                else:
                    # Red frame for wrong prediction
                    frame_color = (255, 0, 0)
                draw = ImageDraw.Draw(gallery_img)
                W, H = gallery_img.size
                for i in range(frame_width):  # Thicker border by drawing multiple rectangles
                    draw.rectangle([i, i, W-1-i, H-1-i], outline=frame_color)

        output_images.append(gallery_img)

    # Compose output: query image, then all gallery images in a row with 224 pixels spacing
    spacing = 0
    total_width = sum(img.width for img in output_images) + spacing * (len(output_images) - 1)
    out_height = target_height

    new_img = Image.new('RGB', (total_width, out_height), (255, 255, 255))
    x_offset = 0
    for i, img in enumerate(output_images):
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
        # Add spacing after each image except the last one
        if i < len(output_images) - 1:
            x_offset += spacing

    new_img.save('temp.png')

def load_connector_weights(model, connector_path):
    """Load connector weights into the model"""
    print(f"Loading connector from: {connector_path}")
    connector_state_dict = torch.load(connector_path, map_location='cuda')
    missing, unexpected = model.load_state_dict(connector_state_dict, strict=False)
    # if missing:
        # print(f"  Missing keys (not loaded): {missing}")
    if unexpected:
        print(f"Unexpected keys (ignored): {unexpected}")
    # print out loaded keys for verification
    loaded_keys = set(connector_state_dict.keys())
    print(f"Loaded keys ({len(loaded_keys)}): {loaded_keys}")

    print("Connector weights loaded successfully!")
    model.eval()


def extract_answer_from_caption(output_text, captions):
    """Extract the answer from a model output or ground-truth caption.

    Returns:
      - single uppercase letter (str, e.g. "A", "C") for the lettered-options
        format produced by prepare_jsonl.py / format_data_reid letter path
      - positive int (1..N) for legacy digit format
      - -1 for legacy stranger
      - None if parsing fails

    Both ground-truth captions and model outputs follow the same shape, so
    this single parser is used on both sides; the comparison `pred == gt`
    works whether both are letters or both are ints.
    """
    if not captions:
        # 1. Letter format (preferred): "Answer: X." where X is A-Z.
        match = re.search(r'Answer:\s*([A-Z])\b', output_text)
        if match:
            return match.group(1)
        # 2. Legacy digit format: "Answer: -?\d+".
        match = re.search(r'Answer:\s*(-?\d+)', output_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # 3. Legacy stranger phrasing.
        if re.search(r'does not match any', output_text, re.IGNORECASE):
            return -1
        # 4. Legacy "gallery image X" pattern.
        match = re.search(r'gallery image\s+(-?\d+)', output_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # 5. Last resort: first standalone uppercase letter, then first int.
        match = re.search(r'\b([A-Z])\b', output_text)
        if match:
            return match.group(1)
        match = re.search(r'-?\d+', output_text)
        return int(match.group(0)) if match else None
    else:
        match = re.search(r'\[(?:[^\[\]\d]* )?(\d+)\]', output_text)
        return int(match.group(1)) if match else None

def swap_in_product_name(output_texts, gt_captions, image_paths):
    """
    Replace [Product X] placeholders in texts with actual product names from image paths.

    Args:
        output_texts: List of model output texts
        gt_captions: List of ground truth captions
        image_paths: List of image file paths containing product category names

    Returns:
        Tuple of (modified_output_texts, modified_gt_captions)
    """


    # Extract product names from image paths
    product_names = []
    for path in image_paths:
        # Extract the product category from path like '/path/to/stapler_final/image.jpg'
        # Look for pattern: category_final
        match = re.search(r'/([^/]+)_final/', path)
        if match:
            product_name = match.group(1).replace('_', ' ').title()
            product_names.append(product_name)
        else:
            # Fallback: extract directory name before the filename
            dir_name = path.split('/')[-2] if '/' in path else path.split('/')[-1].split('.')[0]
            product_name = dir_name.replace('_', ' ').title()
            product_names.append(product_name)

    # Function to replace [Product X] with actual product names
    def replace_product_placeholders(text, product_name):
        # Replace [Product X] with the actual product name
        return re.sub(r'\[Product \d+\]', product_name, text)

    # Apply replacements to both output_texts and gt_captions
    modified_output_texts = []
    modified_gt_captions = []

    for i, (output_text, gt_caption) in enumerate(zip(output_texts, gt_captions)):
        if i < len(product_names):
            product_name = product_names[i]
            modified_output_texts.append(replace_product_placeholders(output_text, product_name))
            modified_gt_captions.append(replace_product_placeholders(gt_caption, product_name))
        else:
            # Fallback if we don't have enough product names
            modified_output_texts.append(output_text)
            modified_gt_captions.append(gt_caption)

    return modified_output_texts, modified_gt_captions

def batch_caption_eval(clip_model, processor, output_texts, gt_captions, all_vision_infos):
    image_paths = [all_vision_infos[i][0]['image'] for i in range(len(all_vision_infos))]

    output_texts, gt_captions = swap_in_product_name(output_texts, gt_captions, image_paths)
    image_inputs = [Image.open(image_path).convert('RGB') for image_path in image_paths]
    inputs = processor(text=output_texts, images=image_inputs, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    text_embeds_gen = outputs.text_embeds
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the probability
    # take the diagonal
    clip_score_image = logits_per_image.diag().tolist()

    inputs = processor(text=gt_captions,images=None,return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_embeds_gt = clip_model.get_text_features(**inputs)
    # similarity between text_embeds_gen and text_embeds_gt
    similarity = torch.nn.functional.cosine_similarity(text_embeds_gen, text_embeds_gt, dim=1)

    clip_score_text = similarity.tolist()

    return clip_score_image, clip_score_text


def process_batch(samples, processor, model, config):
    """Process a batch of samples through the model"""
    examples = []
    query_firsts = []
    labels = []
    gt_captions = []

    for sample in samples:
        # sample, query_first = format_data_reid(sample, reply_order=True, object_type='person')
        query_first = True # Always true if we fix the instruction prompt.
        query_firsts.append(query_first)
        for message in sample:
            if message["role"] == "assistant":
                gt_captions.append(message["content"][0]["text"])
                labels.append(extract_answer_from_caption(message["content"][0]["text"], captions=False))
                sample.remove(message)
        examples.append(sample)

    texts = [processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in examples]

    process_vision_results = [process_vision_info(example) for example in examples]
    image_inputs = [result[1] for result in process_vision_results]
    all_vision_infos = [result[0] for result in process_vision_results]

    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        input_mode='image_only'
    )
    inputs['input_mode'] = config.input_mode
    if 'expert' in inputs['input_mode']:
        expert_inputs = get_expert_inputs(model, all_vision_infos, config.feature_mode)
        # turn into fp16
        expert_inputs = expert_inputs.to(torch.float16).cuda()
        images_per_sample = [
            sum(1 for v in vis if 'image' in v or 'image_url' in v)
            for vis in all_vision_infos
        ]
        inputs['expert_inputs'] = {'inputs': expert_inputs,
            'feature_mode': config.feature_mode,
            'gallery_size': config.gallery_size,
            'batch_size': config.batch_size,
            'input_mode': config.input_mode,
            'object_type': config.object_type,
            'images_per_sample': images_per_sample,}
    inputs = inputs.to("cuda")

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # Force greedy decoding. The ReID answer is deterministic (a
            # single option letter A..; legacy: 1..N or -1), so we don't
            # want sampling. Greedy also sidesteps the bf16 multinomial NaN
            # issue that surfaces when the fine-tuned LLM produces very
            # peaky logits (SFT collapses mass onto a few answer tokens).
            generated_ids = model.generate(
                **inputs, max_new_tokens=500, do_sample=False,
                temperature=1.0, top_p=1.0, top_k=0,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
    return output_texts, labels, query_firsts, all_vision_infos, gt_captions


def main():
    """Main execution function"""
    # Parse arguments
    # default_path = "./qwen/runs/20250809065150-qwen2-5-3b-['merger', 'expert_projector']-ReID-wordindex-k=2-f=0.5-fully_random-0.0002_image_only_batch_size_4_broadcast_fix_prompt/connector.pt"

    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--connector_path", type=str, default=None, help="Path to the connector.pt file")
    parser.add_argument("--model_id", type=str, default=None, help="Model ID")
    # parser.add_argument("--model_size", type=int, default=3, help="Model size")
    parser.add_argument("--object_type", type=str, default="person", help="Object type")
    # parser.add_argument("--test_size", type=int, default=500, help="Number of test samples")
    parser.add_argument("--gallery_size", type=int, default=5, help="Gallery size (k)")
    # parser.add_argument("--filter", type=float, default=0.5, help="Filter out samples with no correct match")
    # parser.add_argument("--random_gallery", action="store_true", help="Use random gallery selection")
    parser.add_argument("--feature_mode", type=str, default="vanilla",
                       choices=["vanilla", "expert", "random", "fully_random"],
                       help="Feature mode: vanilla (no expert), expert (PLIP), random (random images), fully_random (random features)")
    # parser.add_argument("--explain", action="store_true", help="Modify prompt to ask for explanation after the answer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--input_mode", type=str, default="image_only", help="Input mode for inference")
    parser.add_argument("--captions", type=lambda x: x.lower() == 'true', default=False, help="Use captions")
    parser.add_argument("--data_folder", type=str, default="", help="Folder containing the images for evaluation")
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test JSON file')
    parser.add_argument("--prefix", type=str, default='', help="Prefix appended to the run name")
    parser.add_argument("--expert_feature", type=str, default="wyzev0202reid",
                       choices=["PLIP", "wyzev0323token", "wyzev0202reid", "wyzev0415token", "DINOv2", "None"],
                       help="Expert feature model to use when feature_mode='expert'")

    args = parser.parse_args()

    # Set random seeds
    args.seed = 42
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = InferenceConfig(args)

    # Tee stdout/stderr to <connector_dir>/test.log
    setup_logging(os.path.join(os.path.dirname(args.connector_path), 'test.log'))

    # Load dataset
    if config.object_type == "sop":
        _, eval_dataset = load_sop_dataset(args.gallery_size, args.filter, config.object_type, config.captions)
    elif config.object_type == 'face':
        _, eval_dataset = load_face_dataset(args.gallery_size, args.filter, config.object_type, config.captions)
    elif config.object_type == 'pet':
        _, eval_dataset = load_pet_dataset(args.gallery_size, args.filter, config.object_type, config.captions)
    else:
        # _, eval_dataset = load_dataset(config.gallery_size, config.filter, config.object_type, config.captions)
        _, eval_dataset = load_wyze_dataset(args, config.object_type, config.captions)



    # path = './mydata/reid_json/peiscat_k=5.jsonl'
    # with open(path, 'r') as f:
    #     samples = json.load(f)
    # eval_dataset = [format_data_reid(tuple, object_type='pet', captions_dict=None) for tuple in samples]


    # args.connector_path = './qwen/runs/20251025035347-qwen2-5-3b-sop-ReID-k=5-f=0.5-expert-0.0002_expert_and_image_attn_batch_size_4_captions_True_DINO/connector.pt'
    # args.connector_path = './qwen/runs/20251105212637-qwen2-5-3b-pet-ReID-k=5-f=0.5-vanilla-0.0002_image_only_batch_size_4_captions_True_DINO/connector.pt'
    # Load model and processor
    model, processor = load_model_and_processor(config)

    processor.tokenizer.padding_side = 'left'

    load_connector_weights(model, args.connector_path)
    # clip_model, clip_processor = load_clip_model_and_processor()

    # exit()

    # Prepare evaluation
    print(f"Starting evaluation with {len(eval_dataset)} samples")

    # random.shuffle(eval_dataset)  # Already shuffled in data_utils.py
    # subset = eval_dataset[:args.test_size]
    subset = eval_dataset


    # identities = set([sample['query'][f'{object_type}_id'] for sample in subset])
    # print(f"Identities: {len(identities)}")
    # print(f"Images: {len(subset)}")

    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    # Run evaluation
    correct = 0
    total = 0

    clip_scores_text = []
    clip_scores_image = []
    records = []

    for samples in tqdm(dataloader):
        output_texts, labels, _, all_vision_infos, gt_captions = process_batch(samples, processor, model, config)

        # if config.captions:
        #     clip_score_image, clip_score_text = batch_caption_eval(clip_model, clip_processor, output_texts, gt_captions, all_vision_infos)
        #     clip_scores_text.extend(clip_score_text)
        #     clip_scores_image.extend(clip_score_image)

        for i, output_text in enumerate(output_texts):
            print(f"Model output: {output_text}")

            # Use the same captions flag used for label extraction to keep
            # parsing consistent on both sides.
            answer = extract_answer_from_caption(output_text, captions=False)
            print(f"Extracted answer: {answer}, True label: {labels[i]}")

            total += 1
            if answer == labels[i]:
                correct += 1
                print("✓ Correct")
            else:
                print("✗ Wrong")
                pass

            records.append({
                "idx": total - 1,
                "label": labels[i],
                "prediction": answer,
                "model_output": output_text,
            })

            # visualize_tuple(all_vision_infos[i], answer, labels[i], config.object_type)
            # import pdb; pdb.set_trace()
            print("-" * 50)

    # Print results
    print(f"\nFinal Results:")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    acc = correct / total * 100
    print(f"Accuracy: {acc:.1f}% ({correct}/{total})")

    # Save per-sample records to CSV
    test_name = os.path.splitext(os.path.basename(args.test_file))[0]
    out_dir = os.path.dirname(args.connector_path)
    records_path = os.path.join(out_dir, f"predictions_{args.prefix}_{test_name}.csv")
    with open(records_path, "w") as f:
        f.write("idx,label,prediction,model_output\n")
        for r in records:
            model_output_escaped = r['model_output'].replace('"', '""')
            f.write(f"{r['idx']},{r['label']},{r['prediction']},\"{model_output_escaped}\"\n")
    # print(f"Records saved to {records_path}")

    # if config.captions:
    #     print(f"Clip score text: {np.mean(clip_scores_text)}")
    # print(f"Clip score image: {np.mean(clip_scores_image)}")

    # # Write results to os.path.dirname(args.connector_path) + "/results.txt"
    # with open(os.path.dirname(args.connector_path) + "/results.txt", "w") as f:
    #     f.write(f"Accuracy: {acc:.1f}% ({correct}/{total})\n")
    #     if config.captions:
    #         f.write(f"Clip score text: {np.mean(clip_scores_text)}\n")
    #         f.write(f"Clip score image: {np.mean(clip_scores_image)}\n")


if __name__ == "__main__":
    main()