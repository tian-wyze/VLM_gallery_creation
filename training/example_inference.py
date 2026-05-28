"""Minimal end-to-end example for IDAVLM.predict.

The fine-tuned model is prompted with N+1 lettered options (A..(A+N)) where
N = len(gallery_images); one option is a text-only "stranger / not in the
gallery" placeholder. By default the stranger slot is the **last** letter,
so for N=5 the layout is A, B, C, D, E (gallery), F (stranger).

The predict() return dict has three keys:
  - answer:           the letter the model picked (e.g. "B"); None if unparsable.
  - stranger_letter:  the letter currently occupied by the stranger slot —
                      compare against `answer` to tell a gallery hit from
                      a stranger verdict without re-deriving.
  - raw_text:         the full decoded response, e.g. "Answer: B."
"""

from inference import IDAVLM

model = IDAVLM(
    # Older checkpoints (legacy digit/-1 format):
    # connector_path="/home/tian.liu/IDA-VLM/training/runs/20260417_214931_distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/connector.pt",
    # connector_path="/home/tian.liu/IDA-VLM/training/runs/20260430_040726_annotated-distractor-sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/ckpts/connector_epoch_2.pt",

    # Lettered-options ABCD checkpoint (current training format):
    connector_path="/home/tian.liu/IDA-VLM/training/runs/20260502_014222_annotated-distractor-sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False/ckpts/connector_best_step_38000.pt",

    model_id="Qwen/Qwen2.5-VL-7B-Instruct",       # must match the base model used during training
    feature_mode="expert",                         # "vanilla" if trained with no expert
    expert_feature="wyzev0415token",              # ignored when feature_mode="vanilla"
    input_mode="expert_and_image_attn",            # "image_only" for vanilla runs
)

result = model.predict(
    query_image="examples/9991418_9996314_1_D03F278DB00F_D03F278DB00F131749415111_000000_005.jpg",
    gallery_images=[
        "examples/6175403_16032028_3_2CAA8ED7D267_2CAA8ED7D267131736200804_000067_000.jpg",
        "examples/6175403_16032028_4_2CAA8ED7D267_2CAA8ED7D267131736193566_000090_000.jpg",
        "examples/10868404_0_D03F275D4BB3_D03F275D4BB3131711462612_000072_001.jpg",
        "examples/1960853_1_2CAA8E379D04_2CAA8E379D04131754357507_000094_004.jpg",
        "examples/7205943_7216348_7216370_1_7C78B23AAC86_7C78B23AAC86131726288550_000220_011.jpg",
    ],
    # stranger_letter_pos=0,  # uncomment to put the stranger slot at letter A instead of the last letter
)

print("answer:         ", result["answer"])           # e.g. "B"; or the stranger letter; or None
print("stranger_letter:", result["stranger_letter"])  # e.g. "F" (default = last letter)
print("raw_text:       ", result["raw_text"])         # e.g. "Answer: B."

if result["answer"] == result["stranger_letter"]:
    print("verdict: stranger (query person not in any gallery image)")
elif result["answer"] is not None:
    # Map the answer letter back to a gallery position (1-indexed for readability).
    n = 5  # len(gallery_images)
    s_idx = ord(result["stranger_letter"]) - ord("A")
    a_idx = ord(result["answer"]) - ord("A")
    gallery_pos = a_idx + 1 if a_idx < s_idx else a_idx  # account for stranger slot offset
    print(f"verdict: matches gallery image #{gallery_pos} (option {result['answer']})")
else:
    print("verdict: unparsable model output")
