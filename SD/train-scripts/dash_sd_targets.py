"""Stable Diffusion DASH target selection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DashSDTargetParam:
    name: str
    param: nn.Parameter
    module_name: str
    param_name: str
    module_type: str


def get_sd_unet(model: nn.Module) -> nn.Module:
    """Return the CompVis U-Net used by the SD training scripts."""
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        return model.model.diffusion_model
    if hasattr(model, "diffusion_model"):
        return model.diffusion_model
    raise ValueError("Could not locate Stable Diffusion U-Net at model.model.diffusion_model")


def _ancestor_names(module_name: str) -> list[str]:
    parts = module_name.split(".")
    return [".".join(parts[:idx]) for idx in range(1, len(parts))]


def _is_resblock_path(module_name: str, module_by_name: dict[str, nn.Module]) -> bool:
    module_name_lower = module_name.lower()
    if "resblock" in module_name_lower or "resnet" in module_name_lower:
        return True
    for ancestor_name in _ancestor_names(module_name):
        ancestor = module_by_name.get(ancestor_name)
        if ancestor is None:
            continue
        type_name = ancestor.__class__.__name__.lower()
        if "resblock" in type_name or "resnet" in type_name:
            return True
    return False


def _target_subset_matches(
    module_name: str,
    module: nn.Module,
    dash_target: str,
    module_by_name: dict[str, nn.Module],
) -> bool:
    dash_target = (dash_target or "unet").lower()
    if dash_target in {"unet", "unet_all", "all"}:
        return True
    if dash_target in {"unet_xattn", "xattn"}:
        return "attn2" in module_name.lower()
    if dash_target == "unet_attn":
        type_name = module.__class__.__name__.lower()
        return "attn" in module_name.lower() or "attention" in type_name
    if dash_target in {"unet_resnet", "unet_resblock"}:
        type_name = module.__class__.__name__.lower()
        return "resblock" in type_name or "resnet" in type_name or _is_resblock_path(module_name, module_by_name)
    raise ValueError(
        f"Unsupported dash_target for Stable Diffusion: {dash_target}. "
        "Expected one of: unet, unet_all, unet_xattn, unet_attn, unet_resnet."
    )


def select_unet_dash_params(
    model: nn.Module,
    *,
    dash_target: str = "unet",
    include_bias: bool = False,
) -> list[DashSDTargetParam]:
    """Select Linear/Conv2d U-Net parameters eligible for DASH."""
    unet = get_sd_unet(model)
    targets: list[DashSDTargetParam] = []
    module_by_name = dict(unet.named_modules())

    for module_name, module in unet.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _target_subset_matches(module_name, module, dash_target, module_by_name):
            continue

        weight_name = f"{module_name}.weight" if module_name else "weight"
        targets.append(
            DashSDTargetParam(
                name=weight_name,
                param=module.weight,
                module_name=module_name,
                param_name="weight",
                module_type=module.__class__.__name__,
            )
        )

        if include_bias and module.bias is not None:
            bias_name = f"{module_name}.bias" if module_name else "bias"
            targets.append(
                DashSDTargetParam(
                    name=bias_name,
                    param=module.bias,
                    module_name=module_name,
                    param_name="bias",
                    module_type=module.__class__.__name__,
                )
            )

    if not targets:
        raise ValueError(
            f"No eligible U-Net DASH parameters found for dash_target={dash_target}, "
            f"include_bias={include_bias}."
        )
    return targets


def summarize_unet_dash_targets(targets: list[DashSDTargetParam]) -> dict[str, float]:
    param_count = sum(int(target.param.numel()) for target in targets)
    linear_count = sum(1 for target in targets if target.module_type == "Linear")
    conv_count = sum(1 for target in targets if target.module_type == "Conv2d")
    bias_count = sum(1 for target in targets if target.param_name == "bias")
    return {
        "target_tensor_count": float(len(targets)),
        "target_param_count": float(param_count),
        "linear_tensor_count": float(linear_count),
        "conv2d_tensor_count": float(conv_count),
        "bias_tensor_count": float(bias_count),
    }


def snapshot_target_params(targets: list[DashSDTargetParam]) -> dict[str, torch.Tensor]:
    return {target.name: target.param.detach().clone() for target in targets}
