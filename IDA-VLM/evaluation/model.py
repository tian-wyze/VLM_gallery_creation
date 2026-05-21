"""
Model loading utilities for InternVL3 evaluation.
"""

import json
import logging
import os

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str):
    """
    Load an InternVL3 model and its tokenizer from *model_path*.

    The model is loaded in ``bfloat16``, placed on CUDA, and set to eval mode.
    Also reads the ``pad2square`` flag from the model's ``config.json``
    (defaults to ``False`` if absent).

    Parameters
    ----------
    model_path : str
        Local directory or Hugging Face Hub identifier
        (e.g. ``"OpenGVLab/InternVL3-8B"``).

    Returns
    -------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    model_cfg : dict
        Extra configuration extracted from config.json.
        Currently contains ``{"pad2square": bool}``.
    """
    logger.info("Loading model from %s …", model_path)

    # Check flash-attn availability
    try:
        import flash_attn  # noqa: F401
        use_flash_attn = True
    except ImportError:
        logger.warning("flash_attn not installed — falling back to eager attention (slower).")
        use_flash_attn = False

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
    ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    # Read pad2square from model config (if available)
    pad2square = False
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                raw_cfg = json.load(f)
            pad2square = raw_cfg.get("pad2square", False)
        except (json.JSONDecodeError, OSError):
            pass
    if pad2square:
        logger.info("Model config has pad2square=True — will pad frames to square with ImageNet mean.")

    logger.info("Model loaded successfully.")
    model_cfg = {"pad2square": pad2square}
    return model, tokenizer, model_cfg
