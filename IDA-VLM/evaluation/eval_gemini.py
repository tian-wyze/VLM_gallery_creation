import os
import re
import json
import argparse
from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types


def extract_prediction(text):
    """Extract the prediction from a Gemini model response.

    Returns:
      - single uppercase letter (str, e.g. "A") for the lettered-options
        format produced by prepare_jsonl.py — model is trained / prompted to
        output "Answer: X." where X is a letter
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


def build_letter_content(case, prompt_template, data_folder):
    """Build a Gemini multi-modal content list for the lettered-options format.

    The user message interleaves "<letter>:" labels with either the
    corresponding gallery image (for image options) or a text-only
    "(stranger / not in the gallery)" block for the stranger slot at
    `case['stranger_letter_pos']`.

    Returns (content_list, gt_letter).
    """
    n_gallery = len(case['gallery'])
    stranger_pos = case['stranger_letter_pos']

    query_img = Image.open(os.path.join(data_folder, case['query']))
    content = [prompt_template, "Query:", query_img]

    g_idx = 0
    for opt_idx in range(n_gallery + 1):
        letter = chr(ord('A') + opt_idx)
        if opt_idx == stranger_pos:
            content.append(f"{letter}: (stranger / not in the gallery)")
        else:
            img = Image.open(os.path.join(data_folder, case['gallery'][g_idx]))
            content.append(f"{letter}:")
            content.append(img)
            g_idx += 1

    return content, case['answer_letter']


def build_legacy_content(case, prompt_template, data_folder):
    """Build a Gemini multi-modal content list for the legacy (digit/-1) format.

    Returns (content_list, gt_int).
    """
    images = [Image.open(os.path.join(data_folder, case['query']))]
    for g in case['gallery']:
        images.append(Image.open(os.path.join(data_folder, g)))
    return [prompt_template] + images, case['label']


def run_evaluation():
    parser = argparse.ArgumentParser(description='Evaluate Gemini model on person ReID benchmark')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to a benchmark file: .jsonl (preferred, '
                             'lettered-options format) or .json (legacy '
                             '{eval_cases: [...]} digit/-1 format).')
    parser.add_argument('--data_folder', type=str, required=False, default='', help='Path to the image data folder')
    parser.add_argument('--project_id', type=str,
                        default=os.environ.get('GOOGLE_CLOUD_PROJECT', 'fluted-bit-436622-f3'),
                        help='GCP Project ID (defaults to $GOOGLE_CLOUD_PROJECT)')
    parser.add_argument('--location', type=str, default='us-central1', help='GCP Region')
    parser.add_argument('--model_name', type=str, default='gemini-2.5-flash-lite')
    parser.add_argument('--prompt_file', type=str, default='prompt.txt', help='Path to the prompt file')
    args = parser.parse_args()

    # Initialize Gemini Client for Vertex AI
    client = genai.Client(
        vertexai=True,
        project=args.project_id,
        location=args.location
    )

    # Load test data and prompt
    test_data = load_cases(args.test_file)
    with open(args.prompt_file) as f:
        prompt_template = f.read()

    # Output filename — use splitext so .jsonl maps cleanly to .csv (the
    # naive "replace('json', 'csv')" would corrupt .jsonl into .csvl).
    test_name = os.path.splitext(os.path.basename(args.test_file))[0]
    save_filename = f"results/predictions_gemini_{args.model_name}_result_{test_name}.csv"
    os.makedirs('results', exist_ok=True)

    with open(save_filename, 'w') as f:
        f.write('idx,label,prediction,response,query\n')

    correct_ct = 0

    for idx, data in tqdm(enumerate(test_data), total=len(test_data)):
        is_letter = 'answer_letter' in data

        try:
            if is_letter:
                content_list, gt = build_letter_content(
                    data, prompt_template, args.data_folder)
            else:
                content_list, gt = build_legacy_content(
                    data, prompt_template, args.data_folder)

            response = client.models.generate_content(
                model=args.model_name,
                contents=content_list,
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Keep it deterministic for eval
                    max_output_tokens=1024,
                )
            )

            full_text_response = response.text
            prediction = extract_prediction(full_text_response)

            # String-compare so it works for both formats:
            #   letter format: pred="A", gt="A"
            #   legacy format: pred=2, gt=2 → "2"=="2"
            if prediction is not None and str(prediction) == str(gt):
                correct_ct += 1

            with open(save_filename, 'a') as f:
                clean_res = full_text_response.replace(',', ';').replace('\n', ' ')
                f.write(f'{idx},{gt},{prediction},{clean_res},{data["query"]}\n')

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue

    accuracy = (correct_ct / len(test_data)) * 100
    print(f'\nEvaluation Done! Model: {args.model_name}')
    print(f'Accuracy: {round(accuracy, 1)}%')


if __name__ == '__main__':
    run_evaluation()
