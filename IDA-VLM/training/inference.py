"""Single-query inference API for the fine-tuned IDA-VLM Person Re-ID model.

Loads a fine-tuned Qwen2.5-VL connector and exposes a ``predict(query_image,
gallery_images)`` method that returns the matching option letter (A..) — or
the letter of the "stranger / not in the gallery" slot when the query
person is not in any gallery image.

The prompt format matches training exactly (``_format_data_reid_letter``
in ``data.py``): N+1 lettered options where N = len(gallery_images), one
of which is the text-only stranger placeholder. By default the stranger
slot is placed at the **last** letter; pass ``stranger_letter_pos`` to
``predict()`` to put it elsewhere.

Usable both as a library (``from inference import IDAVLM``) and as a CLI
(``python inference.py --connector_path ... --query ... --gallery img1.jpg
img2.jpg ...``).
"""

import argparse
import os
import re
from types import SimpleNamespace

import torch

import replace  # noqa: F401  (monkey-patches Qwen2.5-VL on import)
from utils import load_model_and_processor, process_vision_info
from utils.model_utils import get_expert_inputs


DEFAULT_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "prompt.txt"
)


def _load_default_prompt(path=DEFAULT_PROMPT_PATH):
    with open(path) as f:
        return f.read()


def _build_conversation(query_image, gallery_images, system_prompt,
                        stranger_letter_pos=None):
    """Build the chat conversation (no assistant turn) in the same lettered-
    options format used at training time.

    Layout: query first, then N+1 lettered options where N = len(gallery_images).
    Each option is either a gallery image (with letter label) or the text-only
    "(stranger / not in the gallery)" placeholder. ``stranger_letter_pos``
    (0..N inclusive) controls which slot the stranger occupies; defaults to
    N — the last letter — so e.g. with N=3 gallery images the layout is
    A=gallery[0], B=gallery[1], C=gallery[2], D=stranger.

    The text shapes match ``_format_data_reid_letter`` in data.py exactly,
    so the model sees the same prompt distribution it was fine-tuned on.
    """
    n = len(gallery_images)
    if stranger_letter_pos is None:
        stranger_letter_pos = n  # last slot
    if not 0 <= stranger_letter_pos <= n:
        raise ValueError(
            f"stranger_letter_pos must be in [0, {n}]; got {stranger_letter_pos}"
        )

    user_content = [
        {"type": "text", "text": "Query: "},
        {"type": "image", "image": query_image},
    ]
    g_idx = 0
    for opt_idx in range(n + 1):
        letter = chr(ord("A") + opt_idx)
        if opt_idx == stranger_letter_pos:
            user_content.append({
                "type": "text",
                "text": f"{letter}: (stranger / not in the gallery)",
            })
        else:
            user_content.append({"type": "text", "text": f"{letter}: "})
            user_content.append({"type": "image", "image": gallery_images[g_idx]})
            g_idx += 1

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def _extract_answer(text):
    """Parse the answer from model output, or None on failure.

    Returns:
      - single uppercase letter (str, e.g. "A") for the lettered-options
        format used in current training (one option per case is the
        "stranger / not in gallery" placeholder; the model picks a letter)
      - int 1..N for legacy gallery-index format
      - int -1 for legacy stranger sentinel
      - None if no answer can be parsed
    """
    # Letter format (preferred): "Answer: X." where X is A-Z.
    m = re.search(r"Answer:\s*([A-Z])\b", text)
    if m:
        return m.group(1)
    # Legacy digit format.
    m = re.search(r"Answer:\s*(-?\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.search(r"does not match any", text, re.IGNORECASE):
        return -1
    m = re.search(r"gallery image\s+(-?\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Last resort: first standalone uppercase letter, then first integer.
    m = re.search(r"\b([A-Z])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"-?\d+", text)
    return int(m.group(0)) if m else None


class IDAVLM:
    """Fine-tuned Qwen2.5-VL wrapper for one-shot person re-identification queries.

    The model is prompted with N+1 lettered options (A..(A+N)) where N is the
    gallery size; one option is a text-only stranger placeholder. By default
    the stranger sits at the **last** letter, i.e. with N=3 gallery images
    the option layout is A, B, C (gallery), D (stranger).

    Example:
        >>> model = IDAVLM(
        ...     connector_path="runs/<run>/ckpts/connector_best_step_38000.pt",
        ...     model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        ...     feature_mode="expert",
        ...     expert_feature="wyzev0415token",
        ...     input_mode="expert_and_image_attn",
        ... )
        >>> out = model.predict("query.jpg", ["g1.jpg", "g2.jpg", "g3.jpg"])
        >>> out["answer"]               # "A".. or "D" if stranger; None if unparsable
        >>> out["stranger_letter"]      # "D"  (the letter occupied by the stranger slot)
        >>> out["raw_text"]             # full model response, e.g. "Answer: B."
    """

    def __init__(
        self,
        connector_path,
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        feature_mode="expert",
        # expert_feature="wyzev0323token",
        expert_feature="wyzev0415token",
        input_mode="expert_and_image_attn",
        object_type="person",
        system_prompt=None,
        max_new_tokens=500,
    ):
        config = SimpleNamespace(
            model_id=model_id,
            feature_mode=feature_mode,
            expert_feature=expert_feature,
            input_mode=input_mode,
            object_type=object_type,
            captions=False,
            gallery_size=5,
            batch_size=1,
            seed=42,
        )
        self.config = config
        self.max_new_tokens = max_new_tokens

        self.model, self.processor = load_model_and_processor(config)
        self.processor.tokenizer.padding_side = "left"

        print(f"Loading connector weights from: {connector_path}")
        state_dict = torch.load(connector_path, map_location="cuda")
        _, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"Unexpected keys ignored: {unexpected}")
        self.model.eval()

        self.system_prompt = (
            system_prompt if system_prompt is not None else _load_default_prompt()
        )

    @torch.no_grad()
    def predict(self, query_image, gallery_images, stranger_letter_pos=None):
        """Run one re-ID query.

        Args:
            query_image: path (str) or PIL.Image of the query person.
            gallery_images: list of paths (or PIL.Images); length >= 1.
            stranger_letter_pos: optional int in [0, len(gallery_images)] —
                which letter slot the stranger placeholder occupies. Defaults
                to ``len(gallery_images)`` (the last letter).

        Returns:
            dict with keys:
              - ``answer``: str — single uppercase letter A.. (matches the
                gallery slot that was picked, or the stranger letter if the
                query person is not in the gallery). ``None`` if the model
                output could not be parsed.
              - ``stranger_letter``: str — the letter currently occupied by
                the stranger slot (helps the caller tell whether ``answer``
                is a gallery hit or a stranger verdict).
              - ``raw_text``: str — the model's full decoded response.
        """
        if len(gallery_images) == 0:
            raise ValueError("gallery_images must contain at least one image")

        n = len(gallery_images)
        slot = n if stranger_letter_pos is None else stranger_letter_pos
        stranger_letter = chr(ord("A") + slot)

        conversation = _build_conversation(
            query_image, gallery_images, self.system_prompt,
            stranger_letter_pos=slot,
        )

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        vision_infos, image_inputs = process_vision_info(conversation)

        inputs = self.processor(
            text=[text],
            images=[image_inputs],
            padding=True,
            return_tensors="pt",
            input_mode="image_only",
        )
        inputs["input_mode"] = self.config.input_mode
        if "expert" in self.config.input_mode:
            expert_inputs = get_expert_inputs(
                self.model, [vision_infos], self.config.feature_mode
            )
            expert_inputs = expert_inputs.to(torch.float16).cuda()
            images_per_sample = [
                sum(1 for v in vision_infos if "image" in v or "image_url" in v)
            ]
            inputs["expert_inputs"] = {
                "inputs": expert_inputs,
                "feature_mode": self.config.feature_mode,
                "gallery_size": self.config.gallery_size,
                "batch_size": 1,
                "input_mode": self.config.input_mode,
                "object_type": self.config.object_type,
                "images_per_sample": images_per_sample,
            }
        inputs = inputs.to("cuda")

        with torch.amp.autocast("cuda"):
            # Greedy decoding — the ReID answer is deterministic; sampling is
            # both unnecessary and numerically risky on fine-tuned peaky logits.
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
            )
        trimmed = [
            g[len(i) :] for i, g in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return {
            "answer": _extract_answer(raw_text),
            "stranger_letter": stranger_letter,
            "raw_text": raw_text,
        }


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run a single re-ID inference query (query image + N gallery images)."
    )
    p.add_argument("--connector_path", required=True, help="Path to trained connector.pt")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HF model id matching the model used during training")
    p.add_argument("--feature_mode", default="expert", choices=["vanilla", "expert"])
    p.add_argument(
        "--expert_feature",
        default="wyzev0415token",
        choices=["PLIP", "wyzev0323token", "wyzev0202reid", "wyzev0415token", "DINOv2", "None"],
    )
    p.add_argument("--input_mode", default="expert_and_image_attn",
                   help="Use 'image_only' for vanilla runs, 'expert_and_image_attn' for expert runs")
    p.add_argument("--object_type", default="person")
    p.add_argument("--query", required=True, help="Path to the query image")
    p.add_argument("--gallery", required=True, nargs="+",
                   help="Paths to gallery images (one or more)")
    p.add_argument("--stranger_letter_pos", type=int, default=None,
                   help="Letter slot (0..len(gallery)) for the stranger "
                        "option. Defaults to the last slot.")
    return p.parse_args()


def main():
    args = _parse_args()
    model = IDAVLM(
        connector_path=args.connector_path,
        model_id=args.model_id,
        feature_mode=args.feature_mode,
        expert_feature=args.expert_feature,
        input_mode=args.input_mode,
        object_type=args.object_type,
    )
    result = model.predict(
        args.query, args.gallery,
        stranger_letter_pos=args.stranger_letter_pos,
    )
    print("\n=== Model output ===")
    print(result["raw_text"])
    print(f"\n=== Parsed answer ===  (stranger slot = {result['stranger_letter']})")
    print(result["answer"])
    if result["answer"] == result["stranger_letter"]:
        print("→ verdict: stranger (query person not in any gallery image)")
    elif result["answer"] is not None:
        print(f"→ verdict: matches gallery image at letter {result['answer']}")


if __name__ == "__main__":
    main()
