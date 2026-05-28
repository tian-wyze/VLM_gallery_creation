# Data collation utilities for training

import torch
from transformers import Qwen2_5_VLProcessor
from qwen_vl_utils import fetch_image, fetch_video, extract_vision_info
from PIL import Image
from .model_utils import get_expert_inputs


def process_texts(examples, processor):
    """Process examples to extract and format texts using chat template."""
    return [
        processor.apply_chat_template(example, tokenize=False, add_vision_id=False)
        for example in examples
    ]


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[dict], list[Image.Image]]:
    """Process vision information from conversations."""
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    # ------ My modification ------
    return vision_infos, image_inputs
    # -------------------------------


def process_vision_data(examples):
    """Process examples to extract vision information and image inputs."""
    process_vision_results = [process_vision_info(example) for example in examples]
    image_inputs = [result[1] for result in process_vision_results]
    all_vision_infos = [result[0] for result in process_vision_results]
    return image_inputs, all_vision_infos


def create_base_batch(texts, image_inputs, processor, config):
    """Create the base batch with processed texts and images."""
    return processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
        input_mode=config.input_mode
    )


def add_expert_inputs(batch, model, all_vision_infos, config):
    """Add expert inputs to the batch."""
    expert_inputs = get_expert_inputs(model, all_vision_infos, config.feature_mode)
    # Per-sample image count (1 query + N gallery per sample). Needed by the
    # expert_cross_attn branch in replace.py to build sample-scoped attention.
    images_per_sample = [
        sum(1 for v in vis if 'image' in v or 'image_url' in v)
        for vis in all_vision_infos
    ]
    batch['expert_inputs'] = {
        'inputs': expert_inputs,
        'feature_mode': config.feature_mode,
        'gallery_size': config.gallery_size,
        'batch_size': config.batch_size,
        'input_mode': config.input_mode,
        'object_type': config.object_type,
        'images_per_sample': images_per_sample,
    }
    return batch


def get_image_token_ids(processor):
    """Get image token IDs based on processor type."""
    if isinstance(processor, Qwen2_5_VLProcessor):
        return [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        return [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]


def create_label_masks(labels, processor):
    """Create masks for different types of tokens in labels."""
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens
    image_tokens = get_image_token_ids(processor)
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    return labels


def _build_answer_token_id_set(processor):
    """Return the set of token IDs that may carry answer content.

    The answer is a single capital letter (A, B, C, ...) naming the chosen
    option in the lettered-options prompt format produced by
    prepare_jsonl.py / format_data_reid. The assistant response is
    "Answer: <letter>.", so we keep:
      - every tokenization of A-Z (with and without a leading space) so
        the answer letter contributes to the loss whether the tokenizer
        emits it as a standalone or space-prefixed token
      - the EOS / im_end token, so the model learns when to stop
    All other response tokens (the "Answer:" boilerplate, the trailing
    period, etc.) are masked out of the loss.

    Note: A-Z is intentionally broad — the active option range varies per
    case (up to ~J for size-9 galleries today). Boilerplate words like
    "Answer" tokenize as a single multi-character token, not as the bare
    letter "A", so they do not collide with the keep set.
    """
    tokenizer = processor.tokenizer
    keep = set()
    for s in [chr(ord('A') + i) for i in range(26)]:
        for prefix in ("", " "):
            for tid in tokenizer.encode(prefix + s, add_special_tokens=False):
                keep.add(tid)
    if tokenizer.eos_token_id is not None:
        keep.add(tokenizer.eos_token_id)
    keep.add(151645)  # Qwen's <|im_end|>
    return keep


def restrict_to_answer_tokens(labels, processor):
    """Within the assistant-response range (already unmasked), keep only
    answer-bearing tokens. Everything else in the response is set to -100
    so no gradient is spent on boilerplate template text.
    """
    keep_ids = _build_answer_token_id_set(processor)
    keep_tensor = torch.tensor(list(keep_ids), device=labels.device)
    is_active = labels != -100
    is_keep = torch.isin(labels, keep_tensor)
    labels[is_active & ~is_keep] = -100
    return labels


def mask_between_tokens(config, lst):
    lst = lst.tolist()
    # Corresponds to im_start, assistant, and return ('\n')
    start = next(i + 3 for i in range(len(lst) - 2) if lst[i:i+3] == [151644, 77091, 198])
    # Corresponds to im_end
    end = lst.index(151645, start)
    # Always include the EOS (im_end) token so the model learns when to stop generating
    return torch.tensor([(i < start or i > end) for i in range(len(lst))])


def process_labels(config, batch, processor=None):
    """Mask the loss labels so only answer tokens (+ EOS) are scored.

    Step 1 — mask everything outside the assistant response (system prompt,
             user prompt, image tokens, padding).
    Step 2 — within the response, keep only answer-bearing tokens (the
             single answer letter A-Z and EOS). This prevents gradient
             waste on the "Answer:" prefix and trailing period, and makes
             the loss directly optimise for producing the correct option
             letter.
    """
    labels = batch["input_ids"].clone()
    batch_size = labels.size(0)
    for example_index in range(batch_size):
        mask = mask_between_tokens(config, labels[example_index])
        labels[example_index][mask] = -100
    if processor is not None:
        labels = restrict_to_answer_tokens(labels, processor)
    return labels


_printed_example = False

def print_training_example(examples, texts, batch, processor):
    """Print one full training example for debugging. Runs only once per process."""
    global _printed_example
    if _printed_example:
        return
    _printed_example = True

    print("\n" + "=" * 100)
    print("TRAINING EXAMPLE  (first batch, first sample) — verifying prompt + loss mask")
    print("=" * 100)

    # Conversation structure (role → text / image path)
    print("\n--- Conversation structure ---")
    for msg in examples[0]:
        role = msg.get('role', '?')
        for content in msg.get('content', []):
            if content.get('type') == 'text':
                text = content.get('text', '')
                print(f"  [{role}] TEXT: {text!r}")
            elif content.get('type') == 'image':
                print(f"  [{role}] IMAGE: {content.get('image', '?')}")

    # Fully templated text (as the tokenizer sees it)
    print("\n--- Full templated text ---")
    print(texts[0])

    # Supervised tokens (label != -100) — what actually contributes to the loss
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    supervised_positions = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    supervised_ids = [input_ids[i].item() for i in supervised_positions]
    supervised_tokens = [processor.tokenizer.decode([tid]) for tid in supervised_ids]

    print("\n--- Supervised tokens (label != -100) ---")
    print(f"  Count:   {len(supervised_ids)}")
    print(f"  Token IDs: {supervised_ids}")
    print(f"  Decoded:   {[repr(t) for t in supervised_tokens]}")
    print(f"  Joined:    {processor.tokenizer.decode(supervised_ids)!r}")
    print("=" * 100 + "\n")


def collate_fn(examples, config, processor, model):
    """
    Collate function for training data.

    Args:
        examples: List of training examples
        config: Training configuration object
        processor: Text processor for tokenization
        model: Model for expert feature extraction

    Returns:
        batch: Processed batch with texts, images, expert inputs, and labels
    """
    # Process texts using chat template
    texts = process_texts(examples, processor)

    # Process vision data (images and vision info)
    image_inputs, all_vision_infos = process_vision_data(examples)

    # Create base batch with texts and images
    batch = create_base_batch(texts, image_inputs, processor, config)

    # Add expert inputs to the batch
    batch = add_expert_inputs(batch, model, all_vision_infos, config)

    labels = process_labels(config, batch, processor=processor)
    batch["labels"] = labels

    # Print one example for manual inspection (once per process)
    print_training_example(examples, texts, batch, processor)

    return batch
