"""Stable Diffusion DASH warm-start runtime."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from DASH.plasticity_common import (  # noqa: E402
    as_granularity_matrix,
    expand_decision_values,
    normalize_plasticity_granularity,
    truncate_gradient_global_by_svd_evr,
    truncate_gradient_per_filter_by_bank,
)
from dash_sd_losses import compute_compvis_denoising_loss  # noqa: E402
from dash_sd_targets import get_sd_unet, select_unet_dash_params, summarize_unet_dash_targets  # noqa: E402

log = logging.getLogger(__name__)


_DEFAULT_ATTENTION_LOG_GROUPS = (
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out",
)


def _sync_cuda_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now_seconds():
    _sync_cuda_if_needed()
    return time.perf_counter()


@dataclass
class DashSDConfig:
    warm_start: bool = False
    target: str = "unet"
    signal_mode: str = "preserve_complement"
    plasticity_granularity: str = "per_filter"
    attention_head_wise: bool = False
    grad_aggregation: str = "mean"
    alpha: float = 0.1
    num_aug: int = 10
    aug_mode: str = "none"
    min_shrink: float = 0.004
    svd_truncate_evr: float | None = 0.95
    preserve_forget_evr: float = 0.95
    include_bias: bool = False
    log_cosine_histograms: bool = False
    cosine_hist_bins: int = 50
    retain_batches: int | None = None
    forget_batches: int | None = None
    bn_recalibrate: bool = False
    bn_recalib_batches: int = 200
    attention_logging_enabled: bool = True
    attention_logging_log_histogram: bool = True
    attention_logging_log_cdf: bool = True
    attention_logging_log_median: bool = True
    attention_logging_log_delta_norms: bool = False
    attention_logging_groups: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_ATTENTION_LOG_GROUPS)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "DashSDConfig":
        if mapping is None:
            return cls()
        attention_logging = mapping.get("attention_logging", {}) or {}
        attention_groups = attention_logging.get("groups", _DEFAULT_ATTENTION_LOG_GROUPS)
        if isinstance(attention_groups, str):
            attention_groups = [group.strip() for group in attention_groups.split(",") if group.strip()]
        return cls(
            warm_start=bool(mapping.get("warm_start", mapping.get("dash_warm_start", False))),
            target=str(mapping.get("target", mapping.get("dash_target", "unet"))),
            signal_mode=str(mapping.get("signal_mode", mapping.get("dash_signal_mode", "preserve_complement"))),
            plasticity_granularity=str(mapping.get("plasticity_granularity", "per_filter")),
            attention_head_wise=bool(mapping.get("attention_head_wise", mapping.get("dash_attention_head_wise", False))),
            grad_aggregation=str(mapping.get("grad_aggregation", mapping.get("dash_grad_aggregation", "mean"))),
            alpha=float(mapping.get("alpha", mapping.get("dash_alpha", 0.1))),
            num_aug=int(mapping.get("num_aug", mapping.get("dash_num_aug", 10))),
            aug_mode=str(mapping.get("aug_mode", mapping.get("dash_aug_mode", "none"))),
            min_shrink=float(mapping.get("min_shrink", mapping.get("dash_min_shrink", 0.004))),
            svd_truncate_evr=mapping.get("svd_truncate_evr", mapping.get("dash_svd_truncate_evr", 0.95)),
            preserve_forget_evr=float(mapping.get("preserve_forget_evr", mapping.get("dash_preserve_forget_evr", 0.95))),
            include_bias=bool(mapping.get("include_bias", mapping.get("dash_include_bias", False))),
            log_cosine_histograms=bool(mapping.get("log_cosine_histograms", mapping.get("dash_log_cosine_histograms", False))),
            cosine_hist_bins=int(mapping.get("cosine_hist_bins", mapping.get("dash_cosine_hist_bins", 50))),
            retain_batches=mapping.get("retain_batches", mapping.get("dash_retain_batches")),
            forget_batches=mapping.get("forget_batches", mapping.get("dash_forget_batches")),
            bn_recalibrate=bool(mapping.get("bn_recalibrate", False)),
            bn_recalib_batches=int(mapping.get("bn_recalib_batches", 200)),
            attention_logging_enabled=bool(attention_logging.get("enabled", True)),
            attention_logging_log_histogram=bool(attention_logging.get("log_histogram", True)),
            attention_logging_log_cdf=bool(attention_logging.get("log_cdf", True)),
            attention_logging_log_median=bool(attention_logging.get("log_median", True)),
            attention_logging_log_delta_norms=bool(attention_logging.get("log_delta_norms", False)),
            attention_logging_groups=tuple(str(group) for group in attention_groups),
        )


@dataclass(frozen=True)
class _UnitLayout:
    mode: str
    num_heads: int | None = None
    head_dim: int | None = None
    attention_group: str | None = None
    fallback_reason: str | None = None


def _snapshot_training_mode(model: nn.Module) -> list[tuple[nn.Module, bool]]:
    return [(module, module.training) for module in model.modules()]


def _restore_training_mode(states: list[tuple[nn.Module, bool]]) -> None:
    for module, training in states:
        # Some CompVis submodules replace ``train`` with a plain disabled_train
        # function on the instance. Call the base method directly so restore
        # cannot fail on an unbound replacement.
        nn.Module.train(module, mode=training)


def _snapshot_checkpoint_flags(model: nn.Module) -> list[tuple[nn.Module, str, bool]]:
    states = []
    for module in model.modules():
        for attr in ("use_checkpoint", "checkpoint"):
            if hasattr(module, attr):
                value = getattr(module, attr)
                if isinstance(value, bool):
                    states.append((module, attr, value))
                    setattr(module, attr, False)
    return states


def _restore_checkpoint_flags(states: list[tuple[nn.Module, str, bool]]) -> None:
    for module, attr, value in states:
        setattr(module, attr, value)


def _snapshot_requires_grad(model: nn.Module) -> dict[str, bool]:
    return {name: param.requires_grad for name, param in model.named_parameters()}


def _restore_requires_grad(model: nn.Module, states: dict[str, bool]) -> None:
    for name, param in model.named_parameters():
        if name in states:
            param.requires_grad = states[name]


def _autocast_disabled(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _iter_limited(loader, max_batches: int | None):
    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches < 1:
            return
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        yield batch



def _positive_int_attr(module: nn.Module, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, bool) or value is None:
            continue
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            continue
        if int_value > 0:
            return int_value
    return None


def _attention_group_from_module_name(module_name: str) -> str | None:
    parts = [part.lower() for part in str(module_name or "").split(".")]
    for idx, part in enumerate(parts[:-1]):
        if part not in {"attn1", "attn2"}:
            continue
        projection = parts[idx + 1]
        if projection in {"to_q", "to_k", "to_v", "to_out"}:
            return f"{part}.{projection}"
    return None


def _attention_parent_name(module_name: str) -> str | None:
    parts = str(module_name or "").split(".")
    for idx, part in enumerate(parts):
        if part.lower() in {"attn1", "attn2"}:
            return ".".join(parts[: idx + 1])
    return None


def _infer_attention_head_shape(parent: nn.Module, weight: torch.Tensor, projection: str) -> tuple[int | None, int | None, str | None]:
    if weight.ndim != 2:
        return None, None, "non_2d_weight"
    num_heads = _positive_int_attr(parent, ("heads", "num_heads", "n_heads"))
    if num_heads is None:
        return None, None, "missing_num_heads"

    axis = 1 if projection == "to_out" else 0
    projected_dim = int(weight.shape[axis])
    explicit_head_dim = _positive_int_attr(parent, ("head_dim", "dim_head", "d_head"))
    inner_dim = _positive_int_attr(parent, ("inner_dim",))
    if inner_dim is not None and inner_dim != projected_dim:
        return None, None, "inner_dim_shape_mismatch"
    if explicit_head_dim is not None and explicit_head_dim * num_heads != projected_dim:
        return None, None, "head_dim_shape_mismatch"
    if projected_dim % num_heads != 0:
        return None, None, "head_shape_not_divisible"
    head_dim = explicit_head_dim or (projected_dim // num_heads)
    if head_dim <= 0 or num_heads * head_dim != projected_dim:
        return None, None, "invalid_head_shape"
    return num_heads, head_dim, None


def _build_unit_layouts(model, targets, config: DashSDConfig) -> tuple[dict[str, _UnitLayout], dict[str, float]]:
    module_by_name = dict(get_sd_unet(model).named_modules())
    layouts: dict[str, _UnitLayout] = {}
    stats = {
        "attention_headwise_tensor_count": 0.0,
        "attention_headwise_fallback_tensor_count": 0.0,
    }
    for target in targets:
        group = _attention_group_from_module_name(target.module_name)
        fallback = _UnitLayout(mode=config.plasticity_granularity, attention_group=group)
        if not config.attention_head_wise or group is None:
            layouts[target.name] = fallback
            continue

        projection = group.split(".", 1)[1]
        module = module_by_name.get(target.module_name)
        parent_name = _attention_parent_name(target.module_name)
        parent = module_by_name.get(parent_name or "")
        reason = None
        if target.param_name != "weight":
            reason = "bias"
        elif not isinstance(module, nn.Linear):
            reason = "non_linear_projection"
        elif parent is None:
            reason = "missing_attention_parent"
        elif projection not in {"to_q", "to_k", "to_v", "to_out"}:
            reason = "unsupported_projection"
        else:
            num_heads, head_dim, reason = _infer_attention_head_shape(parent, target.param, projection)
            if reason is None and num_heads is not None and head_dim is not None:
                mode = "head_cols" if projection == "to_out" else "head_rows"
                layouts[target.name] = _UnitLayout(
                    mode=mode,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    attention_group=group,
                )
                stats["attention_headwise_tensor_count"] += 1.0
                continue

        layouts[target.name] = _UnitLayout(
            mode=config.plasticity_granularity,
            attention_group=group,
            fallback_reason=reason or "unknown",
        )
        stats["attention_headwise_fallback_tensor_count"] += 1.0
        stats[f"attention_headwise_fallback_{reason or 'unknown'}"] = (
            stats.get(f"attention_headwise_fallback_{reason or 'unknown'}", 0.0) + 1.0
        )
    return layouts, stats


def _layout_for_name(layouts: dict[str, _UnitLayout], name: str, granularity: str) -> _UnitLayout:
    return layouts.get(name, _UnitLayout(mode=granularity))


def _layout_uses_bank(layout: _UnitLayout) -> bool:
    return layout.mode != "global"


def _as_layout_matrix(tensor: torch.Tensor, layout: _UnitLayout) -> torch.Tensor:
    if layout.mode in {"global", "per_filter"}:
        return as_granularity_matrix(tensor, layout.mode)
    if layout.num_heads is None or layout.head_dim is None or tensor.ndim != 2:
        return as_granularity_matrix(tensor, "per_filter")
    if layout.mode == "head_rows":
        return tensor.reshape(layout.num_heads, layout.head_dim, tensor.shape[1]).reshape(layout.num_heads, -1)
    if layout.mode == "head_cols":
        return (
            tensor.reshape(tensor.shape[0], layout.num_heads, layout.head_dim)
            .permute(1, 0, 2)
            .contiguous()
            .reshape(layout.num_heads, -1)
        )
    raise ValueError(f"Unsupported DASH unit layout mode: {layout.mode}")


def _from_layout_matrix(matrix: torch.Tensor, reference: torch.Tensor, layout: _UnitLayout) -> torch.Tensor:
    if layout.mode in {"global", "per_filter"}:
        return matrix.reshape_as(reference)
    if layout.num_heads is None or layout.head_dim is None or reference.ndim != 2:
        return matrix.reshape_as(reference)
    if layout.mode == "head_rows":
        return matrix.reshape(layout.num_heads, layout.head_dim, reference.shape[1]).reshape_as(reference)
    if layout.mode == "head_cols":
        return (
            matrix.reshape(layout.num_heads, reference.shape[0], layout.head_dim)
            .permute(1, 0, 2)
            .contiguous()
            .reshape_as(reference)
        )
    raise ValueError(f"Unsupported DASH unit layout mode: {layout.mode}")


def _expand_layout_values(values: torch.Tensor, reference: torch.Tensor, layout: _UnitLayout) -> torch.Tensor:
    if reference.ndim == 0:
        return values.view(())
    if layout.mode in {"global", "per_filter"}:
        return expand_decision_values(values, reference, layout.mode)
    if layout.num_heads is None or layout.head_dim is None or reference.ndim != 2:
        return expand_decision_values(values, reference, "per_filter")
    if layout.mode == "head_rows":
        return values.view(layout.num_heads, 1, 1).expand(layout.num_heads, layout.head_dim, reference.shape[1]).reshape_as(reference)
    if layout.mode == "head_cols":
        return values.view(1, layout.num_heads, 1).expand(reference.shape[0], layout.num_heads, layout.head_dim).reshape_as(reference)
    raise ValueError(f"Unsupported DASH unit layout mode: {layout.mode}")


def _project_tensor_perp_reference_layout(
    primary: torch.Tensor,
    reference: torch.Tensor,
    *,
    layout: _UnitLayout,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if primary.shape != reference.shape:
        raise ValueError(
            f"Tensor shape mismatch for projection: primary {tuple(primary.shape)} vs reference {tuple(reference.shape)}"
        )
    primary_matrix = _as_layout_matrix(primary, layout)
    reference_matrix = _as_layout_matrix(reference, layout)
    dot = (primary_matrix * reference_matrix).sum(dim=1, keepdim=True)
    denom = (reference_matrix * reference_matrix).sum(dim=1, keepdim=True)
    coeff = dot / (denom + eps)
    projected = primary_matrix - coeff * reference_matrix
    projected = torch.where(denom > eps, projected, primary_matrix)
    active = (projected.norm(dim=1, keepdim=True) > eps).view(-1)
    return (
        _from_layout_matrix(projected, primary, layout),
        _expand_layout_values(active, primary, layout).bool(),
        int((~active).sum().item()),
        int(active.numel()),
    )


def _cosine_against_negative_gradient_layout(
    weight: torch.Tensor,
    grad: torch.Tensor,
    *,
    layout: _UnitLayout,
    eps: float = 1e-12,
) -> torch.Tensor:
    weight_matrix = _as_layout_matrix(weight, layout)
    grad_matrix = _as_layout_matrix(grad, layout)
    dot = (weight_matrix * (-grad_matrix)).sum(dim=1)
    denom = (weight_matrix.norm(dim=1) * grad_matrix.norm(dim=1)) + eps
    return _expand_layout_values(dot / denom, weight, layout)


def _truncate_gradient(
    grad: torch.Tensor,
    evr_target: float | None,
    *,
    layout: _UnitLayout,
    bank: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if evr_target is None:
        return grad
    if layout.mode == "global":
        return truncate_gradient_global_by_svd_evr(grad, float(evr_target))
    if layout.mode == "per_filter":
        return truncate_gradient_per_filter_by_bank(grad, bank or [], float(evr_target))

    evr_target = float(evr_target)
    if not (0.0 < evr_target <= 1.0):
        raise ValueError(f"svd_truncate_evr must be in (0, 1], got {evr_target}")
    grad_matrix = _as_layout_matrix(grad, layout)
    if grad_matrix.numel() == 0:
        return grad
    bank_matrices = [_as_layout_matrix(sample.detach(), layout) for sample in (bank or []) if sample.shape == grad.shape]
    if not bank_matrices:
        return truncate_gradient_global_by_svd_evr(grad, evr_target)

    truncated = grad_matrix.detach().clone()
    for row_idx in range(grad_matrix.shape[0]):
        row_samples = [sample[row_idx].reshape(1, -1) for sample in bank_matrices if sample.shape == grad_matrix.shape]
        if not row_samples:
            continue
        row_bank = torch.cat(row_samples, dim=0)
        if row_bank.numel() == 0:
            continue
        compute_dtype = torch.float32 if row_bank.dtype in (torch.float16, torch.bfloat16) else row_bank.dtype
        row_bank_compute = row_bank.to(dtype=compute_dtype)
        if float(row_bank_compute.norm().item()) <= 0.0:
            continue
        _, singular_values, vh = torch.linalg.svd(row_bank_compute, full_matrices=False)
        if singular_values.numel() == 0:
            continue
        spectrum = singular_values.square()
        total = spectrum.sum()
        if not torch.isfinite(total) or float(total.item()) <= 0.0:
            continue
        cumulative = torch.cumsum(spectrum, dim=0) / total
        threshold = torch.tensor(evr_target, device=cumulative.device, dtype=cumulative.dtype)
        rank = int(torch.searchsorted(cumulative, threshold).item()) + 1
        rank = max(1, min(rank, int(vh.shape[0])))
        basis = vh[:rank].transpose(0, 1).contiguous()
        row = grad_matrix[row_idx : row_idx + 1].to(dtype=compute_dtype)
        truncated[row_idx : row_idx + 1] = (row @ basis @ basis.transpose(0, 1)).to(dtype=grad.dtype)
    return _from_layout_matrix(truncated, grad, layout)


def _truncate_gradient_dict(
    gradients: dict[str, torch.Tensor] | None,
    evr_target: float | None,
    *,
    layouts: dict[str, _UnitLayout],
    granularity: str,
    bank: dict[str, list[torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor] | None:
    if gradients is None or evr_target is None:
        return gradients
    return {
        name: _truncate_gradient(
            grad,
            float(evr_target),
            layout=_layout_for_name(layouts, name, granularity),
            bank=None if bank is None else bank.get(name),
        )
        for name, grad in gradients.items()
    }


def _project_perp_dict(
    primary: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    layouts: dict[str, _UnitLayout],
    granularity: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    projected = {}
    masks = {}
    skipped_missing = 0
    skipped_zero = 0
    total_units = 0
    for name, primary_grad in primary.items():
        layout = _layout_for_name(layouts, name, granularity)
        reference_grad = reference.get(name)
        if reference_grad is None:
            projected[name] = primary_grad
            active = _as_layout_matrix(primary_grad, layout).norm(dim=1) > 1e-12
            masks[name] = _expand_layout_values(active, primary_grad, layout).bool()
            skipped_missing += 1
            total_units += int(active.numel())
            continue
        proj, mask, skipped, units = _project_tensor_perp_reference_layout(
            primary_grad,
            reference_grad,
            layout=layout,
        )
        projected[name] = proj
        masks[name] = mask
        skipped_zero += skipped
        total_units += units
    stats = {
        "projected_tensors": float(len(projected)),
        "skipped_missing": float(skipped_missing),
        "skipped_zero": float(skipped_zero),
        "total_units": float(total_units),
    }
    return projected, masks, stats


def _full_mask(reference: torch.Tensor, active: bool, layout: _UnitLayout) -> torch.Tensor:
    units = int(_as_layout_matrix(reference, layout).shape[0])
    values = torch.full((units,), active, device=reference.device, dtype=torch.bool)
    return _expand_layout_values(values, reference, layout).bool()


def _gradient_overlap_stats(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    *,
    layouts: dict[str, _UnitLayout],
    granularity: str,
    eps: float = 1e-12,
) -> dict[str, float]:
    overlaps = []
    common_units = 0
    for name, retain_grad in retain_gradients.items():
        forget_grad = forget_gradients.get(name)
        if forget_grad is None or forget_grad.shape != retain_grad.shape:
            continue
        layout = _layout_for_name(layouts, name, granularity)
        retain_matrix = _as_layout_matrix(retain_grad, layout).float()
        forget_matrix = _as_layout_matrix(forget_grad, layout).float()
        dot = (retain_matrix * forget_matrix).sum(dim=1)
        denom = (retain_matrix.norm(dim=1) * forget_matrix.norm(dim=1)) + eps
        overlaps.append(dot / denom)
        common_units += int(retain_matrix.shape[0])
    if not overlaps:
        return {
            "gradient_overlap_common_units": 0.0,
            "gradient_overlap_cos_mean": 0.0,
            "gradient_overlap_cos_min": 0.0,
            "gradient_overlap_cos_max": 0.0,
        }
    overlap_tensor = torch.cat([item.flatten() for item in overlaps], dim=0)
    return {
        "gradient_overlap_common_units": float(common_units),
        "gradient_overlap_cos_mean": float(overlap_tensor.mean().item()),
        "gradient_overlap_cos_min": float(overlap_tensor.min().item()),
        "gradient_overlap_cos_max": float(overlap_tensor.max().item()),
    }

def _norm_stats(gradients: dict[str, torch.Tensor], prefix: str) -> dict[str, float]:
    if not gradients:
        return {prefix: 0.0}
    total_sq = sum(float(grad.float().pow(2).sum().item()) for grad in gradients.values())
    return {prefix: float(total_sq**0.5)}


def _wandb_dash_sd_payload(stats: dict[str, float]) -> dict[str, float]:
    """Map legacy DASH stat keys to flat W&B keys grouped by ____ separators."""
    mapping = {
        "dash_sd_enabled": "dash_sd____status/enabled____warm_start",
        "dash_sd_target_tensor_count": "dash_sd____params/target_tensors____warm_start",
        "dash_sd_target_param_count": "dash_sd____params/target_params____warm_start",
        "dash_sd_updated_tensor_count": "dash_sd____params/updated_tensors____warm_start",
        "dash_sd_updated_param_count": "dash_sd____params/updated_params____warm_start",
        "dash_sd_retain_batches": "dash_sd____data/retain_batches____warm_start",
        "dash_sd_forget_batches": "dash_sd____data/forget_batches____warm_start",
        "dash_sd_alignment_mean": "dash_sd____alignment/mean____warm_start",
        "dash_sd_alignment_std": "dash_sd____alignment/std____warm_start",
        "dash_sd_alignment_min": "dash_sd____alignment/min____warm_start",
        "dash_sd_alignment_max": "dash_sd____alignment/max____warm_start",
        "dash_sd_shrink_mean": "dash_sd____shrink/mean____warm_start",
        "dash_sd_shrink_std": "dash_sd____shrink/std____warm_start",
        "dash_sd_shrink_min": "dash_sd____shrink/min____warm_start",
        "dash_sd_shrink_max": "dash_sd____shrink/max____warm_start",
        "dash_sd_preserve_evr": "dash_sd____config/preserve_evr____warm_start",
        "dash_sd_delta_norm": "dash_sd____update/delta_norm____warm_start",
        "dash_sd_relative_delta_norm": "dash_sd____update/relative_delta_norm____warm_start",
        "dash_sd_grad_norm_retain": "dash_sd____grad_norm/retain____warm_start",
        "dash_sd_grad_norm_forget": "dash_sd____grad_norm/forget____warm_start",
        "dash_sd_gradient_overlap_common_units": "dash_sd____overlap/common_units____warm_start",
        "dash_sd_gradient_overlap_cos_mean": "dash_sd____overlap/cos_mean____warm_start",
        "dash_sd_gradient_overlap_cos_min": "dash_sd____overlap/cos_min____warm_start",
        "dash_sd_gradient_overlap_cos_max": "dash_sd____overlap/cos_max____warm_start",
        "dash_sd_bn_recalib_batches": "dash_sd____bn/recalib_batches____warm_start",
        "dash_sd_bn_module_count": "dash_sd____bn/module_count____warm_start",
        "time____dash/total_seconds____warm_start": "time____dash/total_seconds____warm_start",
        "time____dash/target_selection_seconds____warm_start": "time____dash/target_selection_seconds____warm_start",
        "time____dash/collect_retain_grad_seconds____warm_start": "time____dash/collect_retain_grad_seconds____warm_start",
        "time____dash/collect_forget_grad_seconds____warm_start": "time____dash/collect_forget_grad_seconds____warm_start",
        "time____dash/projection_seconds____warm_start": "time____dash/projection_seconds____warm_start",
        "time____dash/apply_shrink_seconds____warm_start": "time____dash/apply_shrink_seconds____warm_start",
        "time____dash/bn_recalibration_seconds____warm_start": "time____dash/bn_recalibration_seconds____warm_start",
    }
    payload = {}
    for key, value in stats.items():
        if key in mapping:
            payload[mapping[key]] = value
        elif key.startswith("dash_sd_projection/"):
            payload[f"dash_sd____projection/{key.removeprefix('dash_sd_projection/')}____warm_start"] = value
        elif key.startswith("dash_sd_update_module/"):
            payload[f"dash_sd____update_module/{key.removeprefix('dash_sd_update_module/')}____warm_start"] = value
        elif key.startswith("dash_sd_"):
            payload[f"dash_sd____other/{key.removeprefix('dash_sd_')}____warm_start"] = value
    return payload


def _sanitize_wandb_module_name(module_name: str) -> str:
    module_name = str(module_name or "root")
    return module_name.replace("/", "_").replace(" ", "_").replace(":", "_")


def _histogram_edges(counts: torch.Tensor) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, int(counts.numel()) + 1)


def _histogram_median(counts: torch.Tensor, edges: torch.Tensor) -> float:
    counts = counts.detach().float().cpu()
    total = float(counts.sum().item())
    if total <= 0.0:
        return float("nan")
    cdf = counts.cumsum(dim=0)
    median_idx = int(torch.searchsorted(cdf, torch.tensor(0.5 * total)).item())
    median_idx = max(0, min(median_idx, int(counts.numel()) - 1))
    return float((0.5 * (edges[median_idx] + edges[median_idx + 1])).item())


def _add_cdf_payload(payload, prefix: str, counts: torch.Tensor, total: float):
    # CDF is logged as a plot/table; scalar per-bin CDF keys make W&B panels noisy.
    return None


def _cdf_line_plot(wandb, title: str, counts: torch.Tensor, edges: torch.Tensor, total: float):
    cdf = counts.cumsum(dim=0) / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    table = wandb.Table(
        data=[
            [float(x), float(y)]
            for x, y in zip(centers.tolist(), cdf.tolist())
        ],
        columns=["alignment", "cdf"],
    )
    return wandb.plot.line(
        table,
        "alignment",
        "cdf",
        title=title,
    )


def _wandb_dash_sd_histogram_payload(
    config: DashSDConfig,
    cosine_hist_counts,
    cosine_hist_counts_by_module=None,
    cosine_hist_counts_by_attention_group=None,
    cosine_medians_by_module=None,
    cosine_medians_by_attention_group=None,
):
    if not config.log_cosine_histograms or cosine_hist_counts is None:
        return {}
    try:
        import wandb
    except Exception:
        return {}
    if float(cosine_hist_counts.sum().item()) <= 0.0:
        return {}
    edges = _histogram_edges(cosine_hist_counts)
    counts = cosine_hist_counts.detach().float().cpu()
    total = float(counts.sum().item())
    centers = 0.5 * (edges[:-1] + edges[1:])
    payload = {
        "dash_sd____alignment/cosine_histogram____warm_start": wandb.Histogram(
            np_histogram=(
                counts.numpy(),
                edges.detach().cpu().numpy(),
            )
        )
    }
    for idx, count in enumerate(counts.tolist()):
        payload[f"dash_sd____alignment_hist/bin_{idx:02d}____warm_start"] = float(count) / total
    _add_cdf_payload(payload, "dash_sd____alignment_cdf", counts, total)
    payload["dash_sd____alignment_cdf/plot____warm_start"] = _cdf_line_plot(
        wandb,
        "DASH alignment CDF",
        counts,
        edges,
        total,
    )
    payload["dash_sd____alignment_hist/negative_fraction____warm_start"] = float(counts[centers < -0.05].sum().item()) / total
    payload["dash_sd____alignment_hist/near_zero_fraction____warm_start"] = float(
        counts[(centers >= -0.05) & (centers <= 0.05)].sum().item()
    ) / total
    payload["dash_sd____alignment_hist/positive_fraction____warm_start"] = float(counts[centers > 0.05].sum().item()) / total
    for module_name, module_counts in sorted((cosine_hist_counts_by_module or {}).items()):
        module_counts = module_counts.detach().float().cpu()
        module_total = float(module_counts.sum().item())
        if module_total <= 0.0:
            continue
        module_edges = _histogram_edges(module_counts)
        module_key = _sanitize_wandb_module_name(module_name)
        module_prefix = f"dash_sd____alignment_module/{module_key}"
        payload[f"{module_prefix}/cosine_histogram____warm_start"] = wandb.Histogram(
            np_histogram=(
                module_counts.numpy(),
                module_edges.detach().cpu().numpy(),
            )
        )
        _add_cdf_payload(payload, f"{module_prefix}/cdf", module_counts, module_total)
        payload[f"{module_prefix}/cdf/plot____warm_start"] = _cdf_line_plot(
            wandb,
            f"DASH alignment CDF: {module_name}",
            module_counts,
            module_edges,
            module_total,
        )
        if cosine_medians_by_module and module_name in cosine_medians_by_module:
            payload[f"{module_prefix}/median____warm_start"] = float(cosine_medians_by_module[module_name])
        else:
            payload[f"{module_prefix}/median____warm_start"] = _histogram_median(module_counts, module_edges)
    if config.attention_logging_enabled:
        enabled_groups = set(str(group) for group in config.attention_logging_groups)
        for group_name, group_counts in sorted((cosine_hist_counts_by_attention_group or {}).items()):
            if group_name not in enabled_groups:
                continue
            group_counts = group_counts.detach().float().cpu()
            group_total = float(group_counts.sum().item())
            if group_total <= 0.0:
                continue
            group_edges = _histogram_edges(group_counts)
            group_key = _sanitize_wandb_module_name(group_name)
            group_prefix = f"dash_sd____alignment_attention/{group_key}"
            if config.attention_logging_log_histogram:
                payload[f"{group_prefix}/cosine_histogram____warm_start"] = wandb.Histogram(
                    np_histogram=(
                        group_counts.numpy(),
                        group_edges.detach().cpu().numpy(),
                    )
                )
            if config.attention_logging_log_cdf:
                _add_cdf_payload(payload, f"{group_prefix}/cdf", group_counts, group_total)
                payload[f"{group_prefix}/cdf/plot____warm_start"] = _cdf_line_plot(
                    wandb,
                    f"DASH attention alignment CDF: {group_name}",
                    group_counts,
                    group_edges,
                    group_total,
                )
            if config.attention_logging_log_median:
                if cosine_medians_by_attention_group and group_name in cosine_medians_by_attention_group:
                    payload[f"{group_prefix}/median____warm_start"] = float(cosine_medians_by_attention_group[group_name])
                else:
                    payload[f"{group_prefix}/median____warm_start"] = _histogram_median(group_counts, group_edges)
    return payload


def _collect_sd_gradients(
    *,
    model,
    data_loader,
    descriptions,
    targets,
    config: DashSDConfig,
    device: torch.device,
    max_batches: int | None,
    collect_filter_bank: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]], int]:
    grad_aggregation = (config.grad_aggregation or "mean").lower()
    if grad_aggregation not in {"mean", "ema"}:
        raise ValueError(f"Unsupported dash_grad_aggregation for SD DASH: {config.grad_aggregation}")
    if grad_aggregation == "ema" and not (0.0 < float(config.alpha) <= 1.0):
        raise ValueError(f"dash_alpha must be in (0, 1], got {config.alpha}")
    if int(config.num_aug) < 1:
        raise ValueError(f"dash_num_aug must be >= 1, got {config.num_aug}")

    params = [target.param for target in targets]
    names = [target.name for target in targets]
    gradients: dict[str, torch.Tensor] = {}
    sum_gradients: dict[str, torch.Tensor] = {}
    filter_bank: dict[str, list[torch.Tensor]] = {name: [] for name in names} if collect_filter_bank else {}
    processed_batches = 0

    model.zero_grad(set_to_none=True)
    for batch in _iter_limited(data_loader, max_batches):
        batch_accum: dict[str, torch.Tensor] = {}
        for _ in range(int(config.num_aug)):
            with _autocast_disabled(device), torch.enable_grad():
                loss = compute_compvis_denoising_loss(
                    model,
                    batch,
                    descriptions=descriptions,
                    device=device,
                )
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
            for name, grad in zip(names, grads):
                if grad is None:
                    continue
                grad = grad.detach()
                if name in batch_accum:
                    batch_accum[name] = batch_accum[name] + grad
                else:
                    batch_accum[name] = grad.clone()
            model.zero_grad(set_to_none=True)

        if not batch_accum:
            continue
        for name in list(batch_accum.keys()):
            batch_accum[name] = batch_accum[name] / float(config.num_aug)
            if collect_filter_bank and len(filter_bank[name]) < 128:
                filter_bank[name].append(batch_accum[name].detach().clone())

        if grad_aggregation == "mean":
            for name, grad in batch_accum.items():
                if name in sum_gradients:
                    sum_gradients[name] = sum_gradients[name] + grad
                else:
                    sum_gradients[name] = grad.clone()
        else:
            for name, grad in batch_accum.items():
                if name in gradients:
                    gradients[name] = (1.0 - float(config.alpha)) * gradients[name] + float(config.alpha) * grad
                else:
                    gradients[name] = grad.clone()
        processed_batches += 1

    if processed_batches == 0:
        return {}, filter_bank, 0
    if grad_aggregation == "mean":
        gradients = {name: grad / float(processed_batches) for name, grad in sum_gradients.items()}
    return gradients, filter_bank, processed_batches


def _recalibrate_bn(model, retain_loader, descriptions, device: torch.device, num_batches: int) -> dict[str, float]:
    bn_modules = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        log.info("DASH SD BN recalibration skipped: no BatchNorm modules found.")
        return {"bn_recalib_batches": 0.0, "bn_module_count": 0.0}

    training_states = _snapshot_training_mode(model)
    try:
        model.train()
        with torch.no_grad():
            processed = 0
            for batch in _iter_limited(retain_loader, int(num_batches)):
                compute_compvis_denoising_loss(model, batch, descriptions=descriptions, device=device)
                processed += 1
    finally:
        _restore_training_mode(training_states)
    return {"bn_recalib_batches": float(processed), "bn_module_count": float(len(bn_modules))}


def run_dash_sd_warm_start(
    *,
    model,
    retain_loader,
    forget_loader=None,
    descriptions=None,
    dash_config: dict[str, Any] | DashSDConfig | None = None,
    logger=None,
) -> dict[str, float]:
    """Run DASH as an in-place warm-start on the Stable Diffusion U-Net."""
    total_start = _now_seconds()
    timing_stats = {
        "time____dash/target_selection_seconds____warm_start": 0.0,
        "time____dash/collect_retain_grad_seconds____warm_start": 0.0,
        "time____dash/collect_forget_grad_seconds____warm_start": 0.0,
        "time____dash/projection_seconds____warm_start": 0.0,
        "time____dash/apply_shrink_seconds____warm_start": 0.0,
        "time____dash/bn_recalibration_seconds____warm_start": 0.0,
    }
    config = dash_config if isinstance(dash_config, DashSDConfig) else DashSDConfig.from_mapping(dash_config)
    if not config.warm_start:
        timing_stats["time____dash/total_seconds____warm_start"] = _now_seconds() - total_start
        return {"dash_sd_enabled": 0.0, **timing_stats}

    config.signal_mode = str(config.signal_mode).lower()
    if config.signal_mode not in {"retain_only", "forget_perp_retain", "preserve_complement"}:
        raise ValueError(f"Unsupported dash_signal_mode for SD DASH: {config.signal_mode}")
    config.plasticity_granularity = normalize_plasticity_granularity(config.plasticity_granularity)
    config.aug_mode = str(config.aug_mode or "none").lower()
    if config.aug_mode not in {"none", "default"}:
        raise ValueError(f"Unsupported dash_aug_mode for SD DASH: {config.aug_mode}")
    if config.signal_mode != "retain_only" and forget_loader is None:
        raise ValueError(f"forget_loader must be provided when dash_signal_mode={config.signal_mode}")

    active_logger = logger or log
    device = next(model.model.diffusion_model.parameters()).device
    target_start = _now_seconds()
    targets = select_unet_dash_params(
        model,
        dash_target=config.target,
        include_bias=config.include_bias,
    )
    unit_layouts, unit_layout_stats = _build_unit_layouts(model, targets, config)
    needs_filter_bank = (
        any(_layout_uses_bank(_layout_for_name(unit_layouts, target.name, config.plasticity_granularity)) for target in targets)
        and (
            config.svd_truncate_evr is not None
            or config.signal_mode == "preserve_complement"
        )
    )
    target_summary = summarize_unet_dash_targets(targets)
    timing_stats["time____dash/target_selection_seconds____warm_start"] = _now_seconds() - target_start
    if unit_layout_stats.get("attention_headwise_fallback_tensor_count", 0.0) > 0:
        active_logger.warning(
            "DASH SD attention_head_wise fell back to %s for %d tensors.",
            config.plasticity_granularity,
            int(unit_layout_stats["attention_headwise_fallback_tensor_count"]),
        )
    active_logger.info(
        "DASH SD warm-start: target=%s, tensors=%d, params=%d, signal=%s, granularity=%s, attention_head_wise=%s",
        config.target,
        int(target_summary["target_tensor_count"]),
        int(target_summary["target_param_count"]),
        config.signal_mode,
        config.plasticity_granularity,
        bool(config.attention_head_wise),
    )

    requires_grad_state = _snapshot_requires_grad(model)
    training_states = _snapshot_training_mode(model)
    checkpoint_states = _snapshot_checkpoint_flags(model)
    for _, param in model.named_parameters():
        param.requires_grad = False
    for target in targets:
        target.param.requires_grad = True
    model.eval()

    try:
        retain_start = _now_seconds()
        retain_gradients, retain_bank, retain_batches = _collect_sd_gradients(
            model=model,
            data_loader=retain_loader,
            descriptions=descriptions,
            targets=targets,
            config=config,
            device=device,
            max_batches=config.retain_batches,
            collect_filter_bank=needs_filter_bank,
        )
        timing_stats["time____dash/collect_retain_grad_seconds____warm_start"] = _now_seconds() - retain_start
        if not retain_gradients:
            raise ValueError("DASH SD warm-start computed no retain gradients.")

        forget_gradients = None
        forget_bank = None
        projection_stats = {}
        overlap = {}
        invert_gradients = None
        invert_masks = None
        preserve_gradients = None
        preserve_masks = None
        forget_batches = 0

        if config.signal_mode != "retain_only":
            forget_start = _now_seconds()
            forget_gradients, forget_bank, forget_batches = _collect_sd_gradients(
                model=model,
                data_loader=forget_loader,
                descriptions=descriptions,
                targets=targets,
                config=config,
                device=device,
                max_batches=config.forget_batches,
                collect_filter_bank=needs_filter_bank,
            )
            timing_stats["time____dash/collect_forget_grad_seconds____warm_start"] = _now_seconds() - forget_start
            if not forget_gradients:
                raise ValueError("DASH SD warm-start computed no forget gradients.")
            projection_start = _now_seconds()
            overlap = _gradient_overlap_stats(
                retain_gradients,
                forget_gradients,
                layouts=unit_layouts,
                granularity=config.plasticity_granularity,
            )

            if config.signal_mode == "forget_perp_retain":
                invert_gradients, invert_masks, stats = _project_perp_dict(
                    forget_gradients,
                    retain_gradients,
                    layouts=unit_layouts,
                    granularity=config.plasticity_granularity,
                )
                projection_stats.update({f"forget_perp_retain/{key}": val for key, val in stats.items()})
                invert_gradients = _truncate_gradient_dict(
                    invert_gradients,
                    config.svd_truncate_evr,
                    layouts=unit_layouts,
                    granularity=config.plasticity_granularity,
                    bank=forget_bank,
                )
            else:
                forget_reference = {
                    name: _truncate_gradient(
                        grad,
                        config.preserve_forget_evr,
                        layout=_layout_for_name(unit_layouts, name, config.plasticity_granularity),
                        bank=None if forget_bank is None else forget_bank.get(name),
                    )
                    for name, grad in forget_gradients.items()
                }
                preserve_gradients, preserve_masks, stats = _project_perp_dict(
                    retain_gradients,
                    forget_reference,
                    layouts=unit_layouts,
                    granularity=config.plasticity_granularity,
                )
                projection_stats.update({f"retain_perp_forget/{key}": val for key, val in stats.items()})
                preserve_gradients = _truncate_gradient_dict(
                    preserve_gradients,
                    config.svd_truncate_evr,
                    layouts=unit_layouts,
                    granularity=config.plasticity_granularity,
                    bank=retain_bank,
                )
            timing_stats["time____dash/projection_seconds____warm_start"] += _now_seconds() - projection_start
        elif config.svd_truncate_evr is not None:
            projection_start = _now_seconds()
            retain_gradients = _truncate_gradient_dict(
                retain_gradients,
                config.svd_truncate_evr,
                layouts=unit_layouts,
                granularity=config.plasticity_granularity,
                bank=retain_bank,
            )
            timing_stats["time____dash/projection_seconds____warm_start"] += _now_seconds() - projection_start

        updated_tensors = 0
        updated_params = 0
        alignment_count = 0
        alignment_sum = 0.0
        alignment_sq_sum = 0.0
        alignment_min = None
        alignment_max = None
        shrink_count = 0
        shrink_sum = 0.0
        shrink_sq_sum = 0.0
        shrink_min = None
        shrink_max = None
        delta_norm_sq = 0.0
        weight_norm_sq = 0.0
        min_shrink_tensor = None
        cosine_hist_counts = None
        cosine_hist_counts_by_module = {}
        cosine_hist_counts_by_attention_group = {}
        cosine_values_by_module = {}
        cosine_values_by_attention_group = {}
        module_update_norms = {}

        apply_start = _now_seconds()
        with torch.no_grad():
            for target in targets:
                name = target.name
                param = target.param
                layout = _layout_for_name(unit_layouts, name, config.plasticity_granularity)
                if min_shrink_tensor is None or min_shrink_tensor.device != param.device or min_shrink_tensor.dtype != param.dtype:
                    min_shrink_tensor = torch.tensor(float(config.min_shrink), device=param.device, dtype=param.dtype)
                before = param.detach().clone()

                if config.signal_mode == "preserve_complement":
                    preserve_grad = None if preserve_gradients is None else preserve_gradients.get(name)
                    preserve_mask = (
                        preserve_masks.get(name, _full_mask(param.data, False, layout))
                        if preserve_masks is not None
                        else _full_mask(param.data, False, layout)
                    )
                    shrink = torch.full_like(param.data, float(config.min_shrink))
                    cosine = torch.zeros_like(param.data)
                    if preserve_grad is not None:
                        cosine = _cosine_against_negative_gradient_layout(
                            param.data,
                            preserve_grad,
                            layout=layout,
                        ).clamp(-1.0, 1.0)
                        shrink = torch.where(preserve_mask, torch.maximum(cosine, min_shrink_tensor), shrink)
                elif config.signal_mode == "forget_perp_retain":
                    grad = None if invert_gradients is None else invert_gradients.get(name)
                    if grad is None:
                        continue
                    mask = (
                        invert_masks.get(name, _full_mask(param.data, True, layout))
                        if invert_masks is not None
                        else _full_mask(param.data, True, layout)
                    )
                    cosine = _cosine_against_negative_gradient_layout(
                        param.data,
                        grad,
                        layout=layout,
                    ).clamp(-1.0, 1.0)
                    shrink_inv = min_shrink_tensor + (1.0 - min_shrink_tensor) * (1.0 - cosine) * 0.5
                    shrink = torch.where(mask, shrink_inv, torch.ones_like(shrink_inv))
                else:
                    grad = retain_gradients.get(name)
                    if grad is None:
                        continue
                    cosine = _cosine_against_negative_gradient_layout(
                        param.data,
                        grad,
                        layout=layout,
                    ).clamp(-1.0, 1.0)
                    shrink = torch.maximum(cosine, min_shrink_tensor)

                param.mul_(shrink)
                delta = param.detach() - before
                target_delta_norm_sq = float(delta.float().pow(2).sum().item())
                target_weight_norm_sq = float(before.float().pow(2).sum().item())
                delta_norm_sq += target_delta_norm_sq
                weight_norm_sq += target_weight_norm_sq
                module_name = target.module_name or "root"
                module_entry = module_update_norms.setdefault(module_name, {"delta_norm_sq": 0.0, "weight_norm_sq": 0.0})
                module_entry["delta_norm_sq"] += target_delta_norm_sq
                module_entry["weight_norm_sq"] += target_weight_norm_sq
                cosine_float = cosine.detach().float()
                shrink_float = shrink.detach().float()
                if config.log_cosine_histograms:
                    hist = torch.histc(
                        cosine_float.flatten().cpu(),
                        bins=int(config.cosine_hist_bins),
                        min=-1.0,
                        max=1.0,
                    )
                    cosine_hist_counts = hist if cosine_hist_counts is None else cosine_hist_counts + hist
                    module_hist = cosine_hist_counts_by_module.get(module_name)
                    cosine_hist_counts_by_module[module_name] = hist if module_hist is None else module_hist + hist
                    cosine_values_by_module.setdefault(module_name, []).append(cosine_float.flatten().cpu())
                    if layout.attention_group is not None and config.attention_logging_enabled:
                        attention_hist = cosine_hist_counts_by_attention_group.get(layout.attention_group)
                        cosine_hist_counts_by_attention_group[layout.attention_group] = (
                            hist if attention_hist is None else attention_hist + hist
                        )
                        cosine_values_by_attention_group.setdefault(layout.attention_group, []).append(cosine_float.flatten().cpu())
                alignment_count += int(cosine_float.numel())
                alignment_sum += float(cosine_float.sum().item())
                alignment_sq_sum += float(cosine_float.pow(2).sum().item())
                alignment_min = (
                    float(cosine_float.min().item())
                    if alignment_min is None
                    else min(alignment_min, float(cosine_float.min().item()))
                )
                alignment_max = (
                    float(cosine_float.max().item())
                    if alignment_max is None
                    else max(alignment_max, float(cosine_float.max().item()))
                )
                shrink_count += int(shrink_float.numel())
                shrink_sum += float(shrink_float.sum().item())
                shrink_sq_sum += float(shrink_float.pow(2).sum().item())
                shrink_min = (
                    float(shrink_float.min().item())
                    if shrink_min is None
                    else min(shrink_min, float(shrink_float.min().item()))
                )
                shrink_max = (
                    float(shrink_float.max().item())
                    if shrink_max is None
                    else max(shrink_max, float(shrink_float.max().item()))
                )
                updated_tensors += 1
                updated_params += int(param.numel())
        timing_stats["time____dash/apply_shrink_seconds____warm_start"] = _now_seconds() - apply_start
    finally:
        _restore_requires_grad(model, requires_grad_state)
        _restore_checkpoint_flags(checkpoint_states)
        _restore_training_mode(training_states)
        model.zero_grad(set_to_none=True)

    if updated_tensors == 0:
        raise ValueError("DASH SD warm-start did not update any selected U-Net tensors.")

    alignment_mean = alignment_sum / max(alignment_count, 1)
    alignment_var = max((alignment_sq_sum / max(alignment_count, 1)) - alignment_mean**2, 0.0)
    shrink_mean = shrink_sum / max(shrink_count, 1)
    shrink_var = max((shrink_sq_sum / max(shrink_count, 1)) - shrink_mean**2, 0.0)
    cosine_medians_by_module = {
        name: float(torch.cat(chunks).median().item())
        for name, chunks in cosine_values_by_module.items()
        if chunks
    }
    cosine_medians_by_attention_group = {
        name: float(torch.cat(chunks).median().item())
        for name, chunks in cosine_values_by_attention_group.items()
        if chunks
    }
    stats = {
        "dash_sd_enabled": 1.0,
        "dash_sd_target_tensor_count": target_summary["target_tensor_count"],
        "dash_sd_target_param_count": target_summary["target_param_count"],
        "dash_sd_updated_tensor_count": float(updated_tensors),
        "dash_sd_updated_param_count": float(updated_params),
        "dash_sd_retain_batches": float(retain_batches),
        "dash_sd_forget_batches": float(forget_batches),
        "dash_sd_alignment_mean": float(alignment_mean),
        "dash_sd_alignment_std": float(alignment_var**0.5),
        "dash_sd_alignment_min": float(alignment_min if alignment_min is not None else 0.0),
        "dash_sd_alignment_max": float(alignment_max if alignment_max is not None else 0.0),
        "dash_sd_shrink_mean": float(shrink_mean),
        "dash_sd_shrink_std": float(shrink_var**0.5),
        "dash_sd_shrink_min": float(shrink_min if shrink_min is not None else 1.0),
        "dash_sd_shrink_max": float(shrink_max if shrink_max is not None else 1.0),
        "dash_sd_preserve_evr": float(config.preserve_forget_evr),
        "dash_sd_delta_norm": float(delta_norm_sq**0.5),
        "dash_sd_relative_delta_norm": float((delta_norm_sq**0.5) / max(weight_norm_sq**0.5, 1e-12)),
        **timing_stats,
    }
    stats.update({f"dash_sd_{key}": value for key, value in _norm_stats(retain_gradients, "grad_norm_retain").items()})
    if forget_gradients is not None:
        stats.update({f"dash_sd_{key}": value for key, value in _norm_stats(forget_gradients, "grad_norm_forget").items()})
    stats.update({f"dash_sd_{key}": value for key, value in unit_layout_stats.items()})
    stats.update({f"dash_sd_{key}": float(value) for key, value in overlap.items()})
    stats.update({f"dash_sd_projection/{key}": float(value) for key, value in projection_stats.items()})
    for module_name, values in sorted(module_update_norms.items()):
        module_key = _sanitize_wandb_module_name(module_name)
        module_delta_norm = float(values["delta_norm_sq"] ** 0.5)
        module_weight_norm = float(values["weight_norm_sq"] ** 0.5)
        stats[f"dash_sd_update_module/{module_key}/delta_norm"] = module_delta_norm
        stats[f"dash_sd_update_module/{module_key}/relative_delta_norm"] = module_delta_norm / max(module_weight_norm, 1e-12)

    active_logger.info(
        "DASH SD applied: tensors=%d, params=%d, shrink_mean=%.4f, rel_delta=%.4e",
        updated_tensors,
        updated_params,
        stats["dash_sd_shrink_mean"],
        stats["dash_sd_relative_delta_norm"],
    )

    if config.bn_recalibrate:
        bn_start = _now_seconds()
        stats.update(
            {
                f"dash_sd_{key}": value
                for key, value in _recalibrate_bn(
                    model,
                    retain_loader,
                    descriptions,
                    device,
                    config.bn_recalib_batches,
                ).items()
            }
        )
        stats["time____dash/bn_recalibration_seconds____warm_start"] = _now_seconds() - bn_start

    stats["time____dash/total_seconds____warm_start"] = _now_seconds() - total_start

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    **_wandb_dash_sd_payload(stats),
                    **_wandb_dash_sd_histogram_payload(
                        config,
                        cosine_hist_counts,
                        cosine_hist_counts_by_module,
                        cosine_hist_counts_by_attention_group,
                        cosine_medians_by_module,
                        cosine_medians_by_attention_group,
                    ),
                }
            )
    except Exception:
        pass
    return stats
