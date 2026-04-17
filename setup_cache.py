"""
Redirect all model / dataset caches to a shared scratch directory so that
the home-directory disk quota is not exceeded.

Import this module **before** any library that downloads models (torch,
transformers, diffusers, datasets, clip, wandb, …).

    import setup_cache          # sets env vars once
    import torch, transformers  # libraries now use the redirected paths

The module is idempotent – re-importing is a no-op.

Override the default root by setting ``CACHE_ROOT`` in the environment
*before* importing this module.
"""

import os

_CACHE_ROOT = os.environ.get(
    "CACHE_ROOT",
    "/shared/results/common/miksa/intact/SD/.cache",
)

# HuggingFace (hub, datasets, tokenizers, diffusers)
os.environ.setdefault("HF_HOME", os.path.join(_CACHE_ROOT, "huggingface"))

# PyTorch hub / torchvision pretrained weights / torch-fidelity
os.environ.setdefault("TORCH_HOME", os.path.join(_CACHE_ROOT, "torch"))

# General XDG cache (used by SD/ldm/data/imagenet.py for autoencoders)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_ROOT)

# WandB run files
os.environ.setdefault("WANDB_DIR", os.path.join(_CACHE_ROOT, "wandb"))
os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(_CACHE_ROOT, "wandb"))

# OpenAI CLIP (does not respect XDG_CACHE_HOME, needs its own variable)
# The clip library checks download_root; we patch via env so callers
# can use:  clip.load(..., download_root=os.environ.get("CLIP_CACHE_DIR"))
os.environ.setdefault("CLIP_CACHE_DIR", os.path.join(_CACHE_ROOT, "clip"))

# Ensure the top-level cache directory exists
os.makedirs(_CACHE_ROOT, exist_ok=True)
