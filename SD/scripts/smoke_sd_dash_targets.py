#!/usr/bin/env python
"""Smoke-test DASH target selection on the real CompVis U-Net config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SD_ROOT = REPO_ROOT / "SD"
for path in (REPO_ROOT, SD_ROOT, SD_ROOT / "train-scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dash_sd_targets import select_unet_dash_params, summarize_unet_dash_targets  # noqa: E402
from ldm.util import instantiate_from_config  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


class _UNetWrapper:
    def __init__(self, unet):
        self.model = type("ModelWrapper", (), {"diffusion_model": unet})()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(SD_ROOT / "configs/stable-diffusion/v1-intact.yaml"),
        help="Stable Diffusion CompVis config containing model.params.unet_config.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["unet", "unet_xattn", "unet_attn", "unet_resnet"],
        help="DASH target selectors to summarize.",
    )
    parser.add_argument("--include-bias", action="store_true")
    parser.add_argument(
        "--init-device",
        default="meta",
        choices=["meta", "cpu"],
        help="Use meta to inspect real module structure without allocating full U-Net weights.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    with torch.device(args.init_device):
        unet = instantiate_from_config(cfg.model.params.unet_config)
    wrapped = _UNetWrapper(unet)

    payload = {}
    for target in args.targets:
        selected = select_unet_dash_params(
            wrapped,
            dash_target=target,
            include_bias=args.include_bias,
        )
        summary = summarize_unet_dash_targets(selected)
        summary["example_tensors"] = [item.name for item in selected[:10]]
        payload[target] = summary

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
