"""Stable Diffusion DASH denoising loss helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _label_to_int(label) -> int:
    if isinstance(label, torch.Tensor):
        return int(label.detach().cpu().item())
    return int(label)


def _prompts_from_labels(labels, descriptions: Sequence[str] | None) -> list[str]:
    if descriptions is None:
        raise ValueError("DASH SD batch contains labels but descriptions were not provided.")
    return [descriptions[_label_to_int(label)] for label in labels]


def normalize_sd_batch(batch, *, descriptions: Sequence[str] | None = None):
    """Normalize supported SD batch formats to image tensor and prompt list."""
    if isinstance(batch, dict):
        if "jpg" in batch:
            images = batch["jpg"]
        elif "image" in batch:
            images = batch["image"]
        elif "images" in batch:
            images = batch["images"]
        else:
            raise ValueError(f"DASH SD batch dict is missing image field. Available keys: {sorted(batch.keys())}")

        if "txt" in batch:
            prompts = batch["txt"]
        elif "prompt" in batch:
            prompts = batch["prompt"]
        elif "prompts" in batch:
            prompts = batch["prompts"]
        elif "label" in batch:
            prompts = _prompts_from_labels(batch["label"], descriptions)
        elif "labels" in batch:
            prompts = _prompts_from_labels(batch["labels"], descriptions)
        else:
            raise ValueError(f"DASH SD batch dict is missing prompt/label field. Available keys: {sorted(batch.keys())}")
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, labels = batch
        prompts = _prompts_from_labels(labels, descriptions)
    else:
        raise ValueError(
            "DASH SD batch must be a dict or an (images, labels) tuple. "
            f"Got {type(batch).__name__}."
        )

    if not isinstance(images, torch.Tensor):
        images = torch.stack([item for item in images])
    if isinstance(prompts, str):
        prompts = [prompts] * int(images.shape[0])
    else:
        prompts = [str(prompt) for prompt in prompts]
    if len(prompts) != int(images.shape[0]):
        raise ValueError(
            f"DASH SD prompt/image batch mismatch: {len(prompts)} prompts for {int(images.shape[0])} images."
        )
    return images, prompts


def build_compvis_batch(images: torch.Tensor, prompts: list[str]) -> dict[str, object]:
    if images.ndim != 4:
        raise ValueError(f"DASH SD expects image tensor [B,C,H,W] or [B,H,W,C], got shape {tuple(images.shape)}")
    if images.shape[1] in {1, 3, 4}:
        jpg = images.permute(0, 2, 3, 1)
    else:
        jpg = images
    return {"jpg": jpg, "txt": prompts}


def compute_compvis_denoising_loss(
    model,
    batch,
    *,
    descriptions: Sequence[str] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute the standard CompVis LDM denoising loss for a batch."""
    if device is None:
        device = next(model.model.diffusion_model.parameters()).device
    device = torch.device(device)

    images, prompts = normalize_sd_batch(batch, descriptions=descriptions)
    images = images.to(device)
    batch_dict = build_compvis_batch(images, prompts)

    with torch.no_grad():
        x_start, cond = model.get_input(batch_dict, model.first_stage_key)

    x_start = x_start.to(device)
    cond = cond.to(device) if hasattr(cond, "to") else cond
    timesteps = torch.randint(
        0,
        int(model.num_timesteps),
        (x_start.shape[0],),
        device=device,
    ).long()
    noise = torch.randn_like(x_start, device=device)
    x_noisy = model.q_sample(x_start=x_start, t=timesteps, noise=noise)
    model_output = model.apply_model(x_noisy, timesteps, cond)

    parameterization = str(getattr(model, "parameterization", "eps"))
    if parameterization == "eps":
        target = noise
    elif parameterization == "x0":
        target = x_start
    else:
        raise ValueError(
            f"Unsupported Stable Diffusion parameterization for DASH: {parameterization}. "
            "Expected 'eps' or 'x0'."
        )

    if hasattr(model, "get_loss"):
        loss = model.get_loss(model_output, target, mean=True)
    else:
        loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
    return loss

