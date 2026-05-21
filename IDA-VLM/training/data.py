import random
import inflect
p = inflect.engine()
import re
import os

# Optimized storage of prompts: use a nested dict structure and avoid duplicates.
# Each prompt is accessible by category and variant; variants (such as caption, rationale, etc.) are keys.
SYSTEM_MESSAGES = {
    "chatqa": {
        "default": (
            "You are a Vision Language Model specialized in interpreting visual data from chart images.\n"
            "Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.\n"
            "The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.\n"
            "Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."
        )
    },
    "reid": {
        # "default": (
        #     "You are a Vision-Language Model specialized in person re-identification (Re-ID).\n"
        #     "Reply succinctly with the position of the matching gallery image.\n"
        #     "Focus on accurate visual comparison of clothing, body shape, and other appearance cues, give no further commentary unless explicitly prompted."
        # ),
        "default": (
            "You are a Vision-Language Model specialized in person re-identification (Re-ID).\n"
            "Given a query image and gallery images, identify the matching person and write a short, context-focused caption with the correct tag (e.g., “[Person 1]”) as the subject, focusing on its status or environment.\n"
            "Example: “[Person 1] is wearing a blue shirt and black pants.”\n"
            "Focus on accurate visual comparison of clothing, body shape, and other appearance cues, give no further commentary unless explicitly prompted."
        ),
        "caption_short": (
            "You are a Vision-Language Model specialized in person re-identification (Re-ID).\n"
            "Given a query image and gallery images, identify the matching person and write a short, context-focused caption with the correct tag (e.g., “[Person 1]”) as the subject, focusing on its status or environment.\n"
            "Example: “[Person 1] is wearing a blue shirt and black pants.”\n"
            "Focus on accurate visual comparison of clothing, body shape, and other appearance cues, give no further commentary unless explicitly prompted."
        ),
        "rationale_answer": (
            "You are a Vision-Language Model specialized in person re-identification (Re-ID).\n"
            "Given a query image and gallery images, identify the matching person.\n"
            "Provide explanations of how each gallery person is similar or different from the query person, and then come to a conclusion.\n"
            'After the conclusion, provide the final answer in brackets, strictly following the format of "Answer: [Person 1].".'
        )
    },
    "reid_pet": {
        "default": (
            "You are a Vision-Language Model specialized in pet re-identification (Re-ID).\n"
            "Reply succinctly with the position of the matching gallery image.\n"
            "Focus on accurate visual comparison of fur patterns, size, and other appearance cues, give no further commentary unless explicitly prompted."
        ),
        "caption_short": (
            "You are a Vision-Language Model specialized in pet re-identification (Re-ID).\n"
            "Given a query image and gallery images, identify the matching pet and write a short, context-focused caption with the correct tag (e.g., “[Pet 1]”) as the subject, focusing on its status or environment.\n"
            "Example: “[Pet 1] is running across the room.”"
        ),
    },
    "reid_sop": {
        "default": (
            "You are a Vision-Language Model specialized in Stanford Online Products (SOP) re-identification (Re-ID).\n"
            "Reply succinctly with the position of the matching gallery image.\n"
            "Focus on accurate visual comparison of product type, color, and other appearance cues, give no further commentary unless explicitly prompted."
        ),
        "caption_short": (
            "You are a Vision-Language Model specialized in object re-identification (Re-ID).\n"
            "Given a query image and gallery images, identify the matching product and write a short, context-focused caption with the correct tag (e.g., “[Product 1]”) as the subject, focusing on its status or environment.\n"
            "Example: “[Product 1] is on a white kitchen countertop.”"
        ),
        "caption_rationale": (
            "You are a Vision-Language Model specialized in object re-identification (Re-ID).\n"
            "Given a query image and gallery images, identify the matching product.\n"
            "Provide explanations of how each gallery product is similar or different from the query product, and then come to a conclusion.\n"
            'After the conclusion, write a short, context-focused caption with the correct tag (e.g., “[Product 1]”) as the subject, focusing on its status or environment.\n'
            "Example: “[Product 1] is on a white kitchen countertop.”"
        ),
        "rationale_answer": (
            "You are a Vision-Language Model specialized in object re-identification (Re-ID).\n"
            "Given a query image and gallery images, identify the matching product.\n"
            "Provide explanations of how each gallery product is similar or different from the query product, and then come to a conclusion.\n"
            'After the conclusion, provide the final answer in brackets, strictly following the format of "Answer: [Product 1].".'
        )
    },
    "reid_face": {
        "default": (
            "You are a Vision-Language Model specialized in face re-identification (Re-ID).\n"
            "Reply succinctly with the position of the matching gallery image.\n"
            "Focus on accurate visual comparison of face features, such as eyes, nose, mouth, and other appearance cues, give no further commentary unless explicitly prompted."
        ),
        "caption_short": (
            "You are a Vision-Language Model specialized in face re-identification (Re-ID).\n"
            "Given a query image and gallery images, identify the matching face and write a short, context-focused caption with the correct tag (e.g., “[Face 1]”) as the subject.\n"
            "Example: “[Person 1] is smiling and wearing a red shirt.”"
        )
    },
    "reid_buildings": {
        "default": (
            "You are a Vision-Language Model specialized in buildings re-identification (Re-ID).\n"
            "Reply succinctly with the position of the matching gallery image.\n"
            "Focus on accurate visual comparison of building features, such as architecture, color, and other appearance cues, give no further commentary unless explicitly prompted."
        )
    },
    "reid_vehicles": {
        "default": (
            "You are a Vision-Language Model specialized in vehicles re-identification (Re-ID).\n"
            "Reply succinctly with the position of the matching gallery image.\n"
            "Focus on accurate visual comparison of vehicle features, such as make, model, color, and other appearance cues, give no further commentary unless explicitly prompted."
        )
    },
    "reid_unified": {
        "default": (
            "You are a Vision-Language Model specialized in unified re-identification (Re-ID) for objects of any category, including persons, pets, products, faces, buildings, vehicles, and other entities.\n"
            "Reply succinctly with the position of the matching gallery image.\n"
            "Focus on accurate visual comparison of distinctive features, appearance cues, and identifying characteristics relevant to the object category, give no further commentary unless explicitly prompted."
        )
    }
}


def get_system_message(category: str, variant: str = "default"):
    """Access a system prompt for a given category and variant (default fallback is category/default)."""
    category_dict = SYSTEM_MESSAGES.get(category, {})
    if not category_dict:
        raise KeyError(f"Category '{category}' not found in SYSTEM_MESSAGES")
    return category_dict.get(variant, category_dict.get("default"))

# System message variables that reference SYSTEM_MESSAGES dictionary
system_message_chatqa = get_system_message("chatqa", "default")
system_message_reid = get_system_message("reid", "default")
system_message_reid_caption = get_system_message("reid", "caption_short")
system_message_reid_pet = get_system_message("reid_pet", "default")
system_message_reid_buildings = get_system_message("reid_buildings", "default")
system_message_reid_vehicles = get_system_message("reid_vehicles", "default")
system_message_reid_unified = get_system_message("reid_unified", "default")
system_message_reid_face = get_system_message("reid_face", "default")
system_message_reid_face_caption = get_system_message("reid_face", "caption_short")
system_message_reid_sop = get_system_message("reid_sop", "default")
system_message_reid_sop_caption = get_system_message("reid_sop", "caption_short")

import re
from typing import Tuple

def split_by_placeholders(question: str) -> Tuple[str, str, str]:
    """
    Split a question that contains exactly two <>-style placeholders into
    """

    placeholders = list(re.finditer(r"<([^>]+)>", question))

    # Determine order
    first_name = placeholders[0].group(1).lower()
    second_name = placeholders[1].group(1).lower()

    query_first = True if first_name == "query" else False

    # Split into segments
    before, between, after = re.split(r"<[^>]+>", question, maxsplit=2)

    return before, between, after, query_first


def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message_chatqa}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]


def get_path(file_dict, object_type):
    if object_type == 'product' or object_type == 'face' or object_type == 'building' or object_type == 'vehicle':
        return file_dict
    else:
        # person, pet, sop
        return file_dict['img_path']


def _format_data_reid_letter(sample, reply_order=False, object_type='person'):
    """Build the chat message for a letter-format ReID case.

    The sample dict (produced by prepare_jsonl.py) carries:
      - query, gallery (image paths)
      - stranger_letter_pos: int in [0, len(gallery)] — where the stranger
        text option sits among the lettered slots
      - answer_letter: the single letter the assistant must output

    The user message interleaves "<letter>:" labels with either the
    corresponding gallery image (for image options) or a text-only
    "(stranger / not in the gallery)" block (for the stranger slot).
    """
    with open('/home/tian.liu/IDA-VLM/evaluation/prompt.txt') as f:
        system_message = f.read()

    n_gallery = len(sample['gallery'])
    stranger_pos = sample['stranger_letter_pos']
    user_content = [
        {"type": "text", "text": "Query: "},
        {"type": "image", "image": get_path(sample["query"], object_type)},
    ]
    g_idx = 0
    for opt_idx in range(n_gallery + 1):
        letter = chr(ord('A') + opt_idx)
        if opt_idx == stranger_pos:
            user_content.append({
                "type": "text",
                "text": f"{letter}: (stranger / not in the gallery)",
            })
        else:
            user_content.append({"type": "text", "text": f"{letter}: "})
            user_content.append({
                "type": "image",
                "image": get_path(sample["gallery"][g_idx], object_type),
            })
            g_idx += 1

    caption = f"Answer: {sample['answer_letter']}."

    message = [
        {"role": "system",
         "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": user_content},
        {"role": "assistant",
         "content": [{"type": "text", "text": caption}]},
    ]
    if reply_order:
        return message, True  # query_first
    return message


def format_data_reid(sample, reply_order=False, object_type='person', captions_dict=None):

    # Letter-format path: cases produced by prepare_jsonl.py carry an
    # `answer_letter` (e.g. "C") and a `stranger_letter_pos` saying where
    # to insert the text-only "stranger" option among the lettered slots.
    # All non-person datasets (face/pet/sop/etc.) and any legacy person
    # JSON without these fields fall through to the original code path.
    if 'answer_letter' in sample:
        return _format_data_reid_letter(sample, reply_order=reply_order,
                                        object_type=object_type)

    object_type = 'product' if object_type == 'sop' else object_type

    system_message = system_message_reid if object_type == 'person' and captions_dict is None \
        else system_message_reid_caption if object_type == 'person' and captions_dict is not None \
        else system_message_reid_pet if object_type == 'pet' \
        else system_message_reid_buildings if object_type == 'building' \
        else system_message_reid_vehicles if object_type == 'vehicle' \
        else system_message_reid_unified if object_type == 'unified'\
        else system_message_reid_face_caption if object_type == 'face' and captions_dict is not None \
        else system_message_reid_face if object_type == 'face' \
        else system_message_reid_sop if object_type == 'product' and captions_dict is None \
        else system_message_reid_sop_caption

    if captions_dict is not None:
        word = 'Person' if object_type == 'face' else object_type.capitalize()
        if object_type == 'face':
            caption = captions_dict[os.path.join(*get_path(sample["query"], object_type).split('/')[-3:])]
        elif object_type == 'pet':
            caption = captions_dict[os.path.join(*get_path(sample["query"], object_type).split('/')[-4:])]
        else:
            caption = captions_dict[os.path.basename(get_path(sample["query"], object_type))]
        caption, n = re.subn(r'\[[^\[\]]*?\]', f'[{word} {sample["answer"]}]', caption)
        caption = caption if n > 0 else f'[{word} {sample["answer"]}]'
    else: # not specifying a caption
        if sample['answer'] == -1:
            caption = "The person in the query image does not match any person in the gallery. Answer: -1."
        else:
            caption = f"The person in the query image matches the person in gallery image {sample['answer']}. Answer: {sample['answer']}"

    '''
    sample['gallery'] = sample['gallery'][:4]
    if caption == '5':
        caption = '0'
        system_message = system_message + 'Answer 0 if the query image is not in the gallery.'
    '''

    # load customized prompt +++++
    with open('/home/tian.liu/IDA-VLM/evaluation/prompt.txt') as f:
        prompt_template = f.read()
    system_message = prompt_template
    # print(f"Using customized system message from prompt.txt:\n{system_message}")


    query_first = True
    message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": (
                    [
                        {"type": "text", "text": "Query: "},
                        {"type": "image", "image": get_path(sample["query"], object_type)}
                    ]
                    + sum([
                        [
                            {"type": "text", "text": f"Gallery {i+1}:"},
                            {"type": "image", "image": get_path(sample["gallery"][i], object_type)}
                        ]
                        for i in range(len(sample["gallery"]))
                    ], [])
                ),
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ]
    if reply_order:
        return message, query_first
    else:
        return message