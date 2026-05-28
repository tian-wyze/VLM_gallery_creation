import torch
import re
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import sys
import json
import os
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


def extract_prediction(text):
    """Extract the prediction from a Qwen2.5-VL model response.

    Returns:
      - single uppercase letter (str, e.g. "A") for the lettered-options
        format produced by prepare_jsonl.py — model is prompted to output
        "Answer: X." where X is a letter
      - positive int 1..N for the legacy gallery-index format
      - -1 for the legacy stranger sentinel
      - None if no answer can be parsed
    """
    # Letter format (preferred): "Answer: X." where X is A-Z.
    m = re.search(r'Answer:\s*([A-Z])\b', text)
    if m:
        return m.group(1)
    # Legacy digit format.
    m = re.search(r'Answer:\s*(-?\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.search(r'does not match any', text, re.IGNORECASE):
        return -1
    # Last resort: first standalone uppercase letter, then first signed int.
    m = re.search(r'\b([A-Z])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'-?\d+', text)
    return int(m.group(0)) if m else None


def load_cases(path):
    """Load cases from a JSONL file or a legacy benchmark JSON.

    Accepted shapes:
      - JSONL (preferred, output of prepare_jsonl.py): one record per line
        carrying `stranger_letter_pos` and `answer_letter`.
      - JSON dict with key 'eval_cases' (legacy, output of prepare_test.py):
        cases lack letter fields and will be evaluated in the digit format.
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


def load_model(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def build_letter_message(case, prompt_template, data_folder):
    """Build a Qwen2.5-VL chat message for the lettered-options format.

    The user message interleaves "<letter>:" labels with either the
    corresponding gallery image (for image options) or a text-only
    "(stranger / not in the gallery)" block for the stranger slot at
    `case['stranger_letter_pos']`. The system prompt (prompt_template)
    explains the format and the expected "Answer: X." response.

    Returns (messages, gt_letter).
    """
    n_gallery = len(case['gallery'])
    stranger_pos = case['stranger_letter_pos']

    query_path = os.path.abspath(os.path.join(data_folder, case['query']))
    content = [
        {"type": "text", "text": "Query:"},
        {"type": "image", "image": f"file://{query_path}"},
    ]

    g_idx = 0
    for opt_idx in range(n_gallery + 1):
        letter = chr(ord('A') + opt_idx)
        if opt_idx == stranger_pos:
            content.append({"type": "text",
                            "text": f"\n{letter}: (stranger / not in the gallery)"})
        else:
            g_path = os.path.abspath(os.path.join(data_folder, case['gallery'][g_idx]))
            content.append({"type": "text", "text": f"\n{letter}:"})
            content.append({"type": "image", "image": f"file://{g_path}"})
            g_idx += 1

    content.append({"type": "text", "text": "\n" + prompt_template.strip()})

    return [{"role": "user", "content": content}], case['answer_letter']


def build_legacy_message(case, prompt_template, data_folder):
    """Build a Qwen2.5-VL chat message for the legacy (digit/-1) format.

    Returns (messages, gt_int).
    """
    query_path = os.path.abspath(os.path.join(data_folder, case['query']))
    gallery_paths = [os.path.abspath(os.path.join(data_folder, g))
                     for g in case['gallery']]

    content = [
        {"type": "text", "text": "Query Image:\nImage-0:"},
        {"type": "image", "image": f"file://{query_path}"},
        {"type": "text", "text": "\nGallery Images:"},
    ]
    for i, path in enumerate(gallery_paths, start=1):
        content.append({"type": "text", "text": f"\nImage-{i}:"})
        content.append({"type": "image", "image": f"file://{path}"})

    # Strip placeholder lines that older prompts used to mark image positions.
    instruction_lines = [
        line for line in prompt_template.splitlines()
        if not line.startswith("Image-")
        and not line.startswith("Query Image")
        and not line.startswith("Gallery Image")
    ]
    content.append({"type": "text", "text": "\n" + "\n".join(instruction_lines).strip()})

    return [{"role": "user", "content": content}], case['label']


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Qwen2.5-VL model on person ReID benchmark')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to a benchmark file: .jsonl (preferred, '
                             'lettered-options format) or .json (legacy '
                             '{eval_cases: [...]} digit/-1 format).')
    parser.add_argument('--data_folder', type=str, required=False, default='',
                        help='Path to the image data folder (default empty — '
                             'benchmark files with absolute image paths need no prefix)')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-VL-7B',
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

    test_data = load_cases(test_file)

    case = args.model_name
    model_path = f'Qwen/{args.model_name}'

    # Output filename — use splitext so .jsonl maps cleanly to .csv (the
    # naive "replace('json','csv')" would corrupt .jsonl into .csvl).
    test_name = os.path.splitext(os.path.basename(test_file))[0]
    save_filename = f'results/{case}_result_{test_name}.csv'
    print(f'save_filename: {save_filename}')
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)

    # load prompt
    with open(args.prompt_file) as f:
        prompt_template = f.read()

    model, processor = load_model(model_path)
    print('Model loaded successfully!')

    with open(save_filename, 'w') as f:
        f.write('idx,label,prediction,response,query\n')

    correct_ct = 0
    for idx, data in tqdm(enumerate(test_data), total=len(test_data)):
        is_letter = 'answer_letter' in data

        if is_letter:
            messages, gt = build_letter_message(data, prompt_template, data_folder)
        else:
            messages, gt = build_legacy_message(data, prompt_template, data_folder)

        # prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        prediction = extract_prediction(response)

        # String-compare so it works for both formats:
        #   letter format: pred="A", gt="A"
        #   legacy format: pred=2, gt=2 → "2"=="2"
        if prediction is not None and str(prediction) == str(gt):
            correct_ct += 1

        with open(save_filename, 'a') as f:
            # Clean response text of commas / newlines so CSV stays 5-column.
            clean_res = response.replace(',', ';').replace('\n', ' ')
            f.write(f'{idx},{gt},{prediction},{clean_res},{data["query"]}\n')

    print('Evaluation done!')
    accuracy = correct_ct / len(test_data) * 100
    print(f'Case: {case}, Acc: {round(accuracy, 1)}')
