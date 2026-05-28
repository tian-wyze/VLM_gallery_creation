"""
test_capabilities_fintuned_vlm.py
--------------------------------
Compare image-description capabilities between the fine-tuned IDA-VLM
ReID model and the base Qwen2.5-VL-7B on a single image.

Runs a fixed list of probe questions on one test image and prints the
outputs side by side. Optionally also runs the fine-tuned model with
expert injection bypassed (via `--include_no_expert`) so you can
isolate merger drift from expert-injection effects.

The two models are loaded sequentially (not simultaneously) to avoid
OOM on a single 7B-capable GPU. Each phase loads, runs all questions,
then frees memory before the next phase.

Run from IDA-VLM/training/:
    python test_capabilities_fintuned_vlm.py
    python test_capabilities_fintuned_vlm.py --include_no_expert
    python test_capabilities_fintuned_vlm.py --output_file capability_report.txt
"""

import os
import warnings

# Silence noisy library warnings before importing transformers / replace.
# - pkg_resources deprecation (from experts/wyze_embedding/model_loader.py)
# - transformers preprocessor-config / slow-processor / unknown-kwarg notices
# - HF generation flag warnings (e.g. top_k under greedy decoding)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import gc
from types import SimpleNamespace

import torch
from PIL import Image
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

import replace  # noqa: F401 (monkey-patches Qwen2.5-VL; fusion is gated by expert_inputs)
from utils import load_model_and_processor, process_vision_info
from utils.model_utils import get_expert_inputs


# ---- Defaults (override on the CLI) ----------------------------------

# TEST_IMAGE = "/home/tian.liu/IDA-VLM/training/examples/9991418_9996314_1_D03F278DB00F_D03F278DB00F131749415111_000000_005_fullframe.jpg"
TEST_IMAGE = "/home/tian.liu/IDA-VLM/training/examples/9991418_9996314_1_D03F278DB00F_D03F278DB00F131749415111_000000_005.jpg"


CONNECTOR_PATH = "/home/tian.liu/IDA-VLM/training/runs/20260502_014222_annotated-distractor-sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/ckpts/connector_best_step_38000.pt"

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Probes designed for the reference image (a person in a white t-shirt
# smoking a cigarette and wearing a black watch, walking next to a car).
# Each question targets a different aspect of visual understanding so
# you can see where the fine-tune does/doesn't hurt.
QUESTIONS = [
    # ---- Open-ended / scene-level ----
    "Describe this image in detail.",
    "What is the person in the image doing?",
    "What is the person wearing?",
    "Is the person holding or smoking anything? Describe it.",
    "Is the person wearing any accessories such as a watch or jewelry?",
    "Is there a vehicle in the image? If so, describe it.",
    "What objects can you see in this image besides the person?",
    "How many people are visible in this image?",
    # ---- Fine-grained attribute probes (binary / short-answer) ----
    # Mix of true positives ("yes" expected) and true negatives ("no" expected)
    # so you can spot hallucination in either direction.
    "Is the man wearing glasses? Answer yes or no.",
    "Is the man smoking? Answer yes or no.",
    "Is the man wearing a hat? Answer yes or no.",
    "Is the man wearing a watch? Answer yes or no.",
    "Is the man indoors or outdoors?",
    "What color is the man's t-shirt?",
    "What color are the man's pants?",
    "What color is the man's watch?",
    "What is the man holding in his hand?",
    "What color is the car in the image?",
]


# ---- Inference helpers ------------------------------------------------

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_chat_inputs(processor, image, question):
    """Build a one-turn user message (image + text) and run the processor."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    vision_infos, image_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=[image_inputs],
        padding=True,
        return_tensors="pt",
        input_mode="image_only",
    )
    return inputs, vision_infos


def add_expert_inputs(model, inputs, vision_infos, config):
    """Run the expert backbone and attach expert_inputs to the batch."""
    expert_inputs = get_expert_inputs(
        model, [vision_infos], config.feature_mode
    )
    expert_inputs = expert_inputs.to(torch.float16).cuda()
    images_per_sample = [
        sum(1 for v in vision_infos if "image" in v or "image_url" in v)
    ]
    inputs["expert_inputs"] = {
        "inputs": expert_inputs,
        "feature_mode": config.feature_mode,
        "gallery_size": 1,
        "batch_size": 1,
        "input_mode": config.input_mode,
        "object_type": config.object_type,
        "images_per_sample": images_per_sample,
    }
    return inputs


@torch.no_grad()
def ask(model, processor, image, question, *, expert_config=None,
        max_new_tokens=300):
    """Run one question through `model`.

    If `expert_config` is given, runs the expert backbone and passes
    `expert_inputs` (i.e. invokes the fine-tuned fusion path). Otherwise
    the patched forward sees `expert_inputs=None` and skips fusion, so
    the model behaves like a vanilla Qwen2.5-VL (with whatever merger
    weights happen to be loaded).
    """
    inputs, vision_infos = build_chat_inputs(processor, image, question)
    inputs["input_mode"] = (
        expert_config.input_mode if expert_config is not None else "image_only"
    )
    if expert_config is not None:
        inputs = add_expert_inputs(model, inputs, vision_infos, expert_config)
    inputs = inputs.to("cuda")
    with torch.amp.autocast("cuda"):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, generated_ids)]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return out.strip()


# ---- Model loading ----------------------------------------------------

def load_base_qwen(model_id):
    """Load base Qwen2.5-VL-7B with no expert wiring and no fine-tune weights."""
    config = SimpleNamespace(
        model_id=model_id,
        feature_mode="vanilla",
        expert_feature="None",
        input_mode="image_only",
        object_type="person",
        captions=False,
        gallery_size=5,
        batch_size=1,
        seed=42,
    )
    model, processor = load_model_and_processor(config)
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor, config


def load_finetuned(model_id, connector_path):
    """Load Qwen2.5-VL-7B with the wyzev0415token expert wired, then
    overlay the connector.pt weights (merger + expert_projector)."""
    config = SimpleNamespace(
        model_id=model_id,
        feature_mode="expert",
        expert_feature="wyzev0415token",
        input_mode="expert_and_image_attn",
        object_type="person",
        captions=False,
        gallery_size=5,
        batch_size=1,
        seed=42,
    )
    model, processor = load_model_and_processor(config)
    processor.tokenizer.padding_side = "left"
    print(f"Loading connector weights from: {connector_path}")
    state_dict = torch.load(connector_path, map_location="cuda")
    _, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  Unexpected keys ignored: {unexpected}")
    model.eval()
    return model, processor, config


# ---- Main -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare fine-tuned IDA-VLM ReID model vs. base Qwen2.5-VL "
            "on image-description questions for a single image."
        ),
    )
    parser.add_argument("--image", default=TEST_IMAGE,
                        help="Path to the test image.")
    parser.add_argument("--connector_path", default=CONNECTOR_PATH,
                        help="Path to the fine-tuned connector.pt checkpoint.")
    parser.add_argument("--model_id", default=MODEL_ID,
                        help="HF id of the base Qwen2.5-VL model.")
    parser.add_argument("--include_no_expert", action="store_true",
                        help="Also run the fine-tuned model with expert "
                             "injection bypassed (isolates merger drift "
                             "from expert-injection effects).")
    parser.add_argument("--questions", nargs="+", default=None,
                        help="Override the default question list "
                             "(one question per arg).")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--output_file", default=None,
                        help="If set, also write the side-by-side report "
                             "to this text file.")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    questions = args.questions if args.questions else QUESTIONS

    # ---- Phase 1: base Qwen ----
    print("\n" + "=" * 70)
    print("Phase 1/2: Loading base Qwen2.5-VL-7B ...")
    print("=" * 70)
    base_model, base_processor, _ = load_base_qwen(args.model_id)
    base_answers = []
    for i, q in enumerate(questions, 1):
        print(f"\n[base]  Q{i}: {q}")
        out = ask(base_model, base_processor, image, q,
                  expert_config=None, max_new_tokens=args.max_new_tokens)
        base_answers.append(out)
        print(f"[base]  A: {out}")
    del base_model, base_processor
    free_memory()

    # ---- Phase 2: fine-tuned model (with expert; deployed config) ----
    print("\n" + "=" * 70)
    print("Phase 2/2: Loading fine-tuned model "
          "(expert_and_image_attn + wyzev0415token) ...")
    print("=" * 70)
    ft_model, ft_processor, ft_config = load_finetuned(
        args.model_id, args.connector_path
    )

    ft_expert_answers = []
    for i, q in enumerate(questions, 1):
        print(f"\n[ft+expert]  Q{i}: {q}")
        out = ask(ft_model, ft_processor, image, q,
                  expert_config=ft_config,
                  max_new_tokens=args.max_new_tokens)
        ft_expert_answers.append(out)
        print(f"[ft+expert]  A: {out}")

    ft_noexpert_answers = None
    if args.include_no_expert:
        # Same fine-tuned weights, but skip expert injection. The replace.py
        # forward only fuses when expert_inputs is not None — so this shows
        # the effect of merger drift alone, without the person-identity
        # vector being mixed into every image token.
        ft_noexpert_answers = []
        for i, q in enumerate(questions, 1):
            print(f"\n[ft-no_expert]  Q{i}: {q}")
            out = ask(ft_model, ft_processor, image, q,
                      expert_config=None,
                      max_new_tokens=args.max_new_tokens)
            ft_noexpert_answers.append(out)
            print(f"[ft-no_expert]  A: {out}")
    del ft_model, ft_processor
    free_memory()

    # ---- Side-by-side report ----
    lines = []
    lines.append("=" * 80)
    lines.append(f"IMAGE:     {args.image}")
    lines.append(f"CONNECTOR: {args.connector_path}")
    lines.append(f"BASE:      {args.model_id}")
    lines.append("=" * 80)
    for i, q in enumerate(questions, 1):
        lines.append(f"\n--- Q{i}: {q} ---\n")
        lines.append(f"[base Qwen]\n{base_answers[i-1]}\n")
        lines.append(f"[fine-tuned + expert (deployed)]\n{ft_expert_answers[i-1]}\n")
        if ft_noexpert_answers is not None:
            lines.append(
                f"[fine-tuned, expert injection bypassed]\n"
                f"{ft_noexpert_answers[i-1]}\n"
            )
    report = "\n".join(lines)
    print("\n" + report)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(report)
        print(f"\nResults written to: {args.output_file}")


if __name__ == "__main__":
    main()
