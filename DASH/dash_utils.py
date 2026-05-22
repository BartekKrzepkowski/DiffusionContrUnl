"""DASH-specific gradient and warm-start utilities."""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..plasticity_common import (
    as_filter_matrix,
    as_granularity_matrix,
    cosine_against_negative_gradient,
    expand_decision_values,
    gradient_overlap_stats,
    normalize_plasticity_granularity,
    project_tensor_onto_reference,
    project_tensor_perp_reference,
    truncate_gradient_global_by_svd_evr,
    truncate_gradient_per_filter_by_bank,
)


def _snapshot_batchnorm_stats(
    model: nn.Module,
) -> list[tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    bn_states: list[tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.running_mean is None or module.running_var is None:
                continue
            num_batches = None
            if hasattr(module, "num_batches_tracked") and module.num_batches_tracked is not None:
                num_batches = module.num_batches_tracked.detach().clone()
            bn_states.append(
                (
                    module,
                    module.running_mean.detach().clone(),
                    module.running_var.detach().clone(),
                    num_batches,
                )
            )
    return bn_states


def _restore_batchnorm_stats(
    bn_states: list[tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor | None]]
) -> None:
    for module, running_mean, running_var, num_batches in bn_states:
        module.running_mean.data.copy_(running_mean)
        module.running_var.data.copy_(running_var)
        if num_batches is not None and hasattr(module, "num_batches_tracked"):
            module.num_batches_tracked.data.copy_(num_batches)


def _snapshot_training_mode(model: nn.Module) -> list[tuple[nn.Module, bool]]:
    states: list[tuple[nn.Module, bool]] = []
    for module in model.modules():
        states.append((module, module.training))
    return states


def _restore_training_mode(states: list[tuple[nn.Module, bool]]) -> None:
    for module, training in states:
        module.train(mode=training)


def _compute_batch_gradients_full(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    include_bias: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute gradients for a single batch, returning per-parameter tensors.

    Args:
        model: Neural network model
        data: Input data
        target: Target labels
        criterion: Loss function
        include_bias: Whether to include bias gradients

    Returns:
        Dictionary of gradients keyed by parameter name
    """
    output = model(data)
    loss = criterion(output, target)

    params = []
    names = []
    for name, param in model.named_parameters():
        if not include_bias and name.endswith("bias"):
            continue
        if not param.requires_grad:
            continue
        params.append(param)
        names.append(name)

    if not params:
        return {}

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    gradients_dict = {}
    for name, grad in zip(names, grads):
        if grad is None:
            continue
        gradients_dict[name] = grad.detach().clone()

    return gradients_dict


DashAugFn = Callable[[torch.Tensor], torch.Tensor]


def _default_dash_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Lightweight tensor-space augmentation for DASH multi-augmentation.

    - For image-like tensors [B, C, H, W]: random horizontal flip + random crop.
    - For other tensors: small Gaussian noise jitter.
    """
    if x.ndim != 4:
        return x + 0.01 * torch.randn_like(x)

    flip_mask = (torch.rand(x.shape[0], device=x.device) < 0.5).view(-1, 1, 1, 1)
    x_flip = torch.flip(x, dims=[3])
    x_aug = torch.where(flip_mask, x_flip, x)

    _, _, h, w = x_aug.shape
    pad = min(4, max(1, min(h, w) // 8))
    x_pad = torch.nn.functional.pad(x_aug, (pad, pad, pad, pad), mode="replicate")
    yy = torch.randint(0, 2 * pad + 1, (x.shape[0],), device=x.device)
    xx = torch.randint(0, 2 * pad + 1, (x.shape[0],), device=x.device)
    crops = []
    for b in range(x.shape[0]):
        crops.append(x_pad[b : b + 1, :, yy[b] : yy[b] + h, xx[b] : xx[b] + w])
    return torch.cat(crops, dim=0)



def _as_output_matrix(tensor: torch.Tensor) -> torch.Tensor:
    return as_filter_matrix(tensor)


def _output_mask_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    if tensor.ndim == 0:
        return ()
    if tensor.ndim == 1:
        return tuple(tensor.shape)
    return (tensor.shape[0],) + (1,) * (tensor.ndim - 1)


def _expand_output_values(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return expand_decision_values(values, reference, "per_filter")


def _project_per_output(
    primary_grad: torch.Tensor,
    reference_grad: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    projected, mask, skipped, _ = project_tensor_perp_reference(
        primary_grad,
        reference_grad,
        granularity="per_filter",
        eps=eps,
    )
    return projected, mask, skipped


def _project_gradients_perp(
    primary_gradients: dict[str, torch.Tensor],
    reference_gradients: dict[str, torch.Tensor],
    *,
    granularity: str,
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
):
    granularity = normalize_plasticity_granularity(granularity)
    projected: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_zero = 0
    total_units = 0

    for name, primary_grad in primary_gradients.items():
        reference_grad = reference_gradients.get(name)
        if reference_grad is None:
            projected[name] = primary_grad
            active = as_granularity_matrix(primary_grad, granularity).norm(dim=1) > eps
            masks[name] = expand_decision_values(active, primary_grad, granularity).bool()
            skipped_missing += 1
            total_units += int(active.numel())
            continue
        proj, mask, skipped, units = project_tensor_perp_reference(
            primary_grad,
            reference_grad,
            granularity=granularity,
            eps=eps,
        )
        projected[name] = proj
        masks[name] = mask
        total_units += units
        skipped_zero += skipped

    stats = {
        "projected_tensors": float(len(projected)),
        "skipped_missing": float(skipped_missing),
        "skipped_zero": float(skipped_zero),
        "total_units": float(total_units),
    }
    if return_stats and return_masks:
        return projected, stats, masks
    if return_stats:
        return projected, stats
    if return_masks:
        return projected, {}, masks
    return projected


def _project_gradients_onto(
    primary_gradients: dict[str, torch.Tensor],
    reference_gradients: dict[str, torch.Tensor],
    *,
    granularity: str,
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
):
    granularity = normalize_plasticity_granularity(granularity)
    projected: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_zero = 0
    total_units = 0

    for name, primary_grad in primary_gradients.items():
        reference_grad = reference_gradients.get(name)
        if reference_grad is None:
            projected[name] = torch.zeros_like(primary_grad)
            masks[name] = torch.zeros_like(expand_decision_values(torch.ones(1, device=primary_grad.device), primary_grad, granularity)).bool()
            skipped_missing += 1
            total_units += int(as_granularity_matrix(primary_grad, granularity).shape[0])
            continue
        proj, mask, skipped, units = project_tensor_onto_reference(
            primary_grad,
            reference_grad,
            granularity=granularity,
            eps=eps,
        )
        projected[name] = proj
        masks[name] = mask
        total_units += units
        skipped_zero += skipped

    stats = {
        "projected_tensors": float(len(projected)),
        "skipped_missing": float(skipped_missing),
        "skipped_zero": float(skipped_zero),
        "total_units": float(total_units),
    }
    if return_stats and return_masks:
        return projected, stats, masks
    if return_stats:
        return projected, stats
    if return_masks:
        return projected, {}, masks
    return projected


def project_forget_perp_retain(
    forget_gradients: dict[str, torch.Tensor],
    retain_gradients: dict[str, torch.Tensor],
    *,
    granularity: str = "per_filter",
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
):
    return _project_gradients_perp(
        forget_gradients,
        retain_gradients,
        granularity=granularity,
        eps=eps,
        return_stats=return_stats,
        return_masks=return_masks,
    )


def project_forget_perp_retain_per_output(
    forget_gradients: dict[str, torch.Tensor],
    retain_gradients: dict[str, torch.Tensor],
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
):
    projected, stats, masks = project_forget_perp_retain(
        forget_gradients,
        retain_gradients,
        granularity="per_filter",
        eps=eps,
        return_stats=True,
        return_masks=True,
    )
    stats["total_outputs"] = stats.pop("total_units")
    if return_stats and return_masks:
        return projected, stats, masks
    if return_stats:
        return projected, stats
    if return_masks:
        return projected, {}, masks
    return projected


def project_retain_perp_forget(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    *,
    granularity: str = "per_filter",
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
):
    return _project_gradients_perp(
        retain_gradients,
        forget_gradients,
        granularity=granularity,
        eps=eps,
        return_stats=return_stats,
        return_masks=return_masks,
    )


def project_retain_perp_forget_per_output(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
):
    projected, stats, masks = project_retain_perp_forget(
        retain_gradients,
        forget_gradients,
        granularity="per_filter",
        eps=eps,
        return_stats=True,
        return_masks=True,
    )
    stats["total_outputs"] = stats.pop("total_units")
    if return_stats and return_masks:
        return projected, stats, masks
    if return_stats:
        return projected, stats
    if return_masks:
        return projected, {}, masks
    return projected


def project_retain_onto_forget(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    *,
    granularity: str = "per_filter",
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
):
    return _project_gradients_onto(
        retain_gradients,
        forget_gradients,
        granularity=granularity,
        eps=eps,
        return_stats=return_stats,
        return_masks=return_masks,
    )


_PROJECT_FORGET_PERP_RETAIN = project_forget_perp_retain
_PROJECT_RETAIN_PERP_FORGET = project_retain_perp_forget
_PROJECT_RETAIN_ONTO_FORGET = project_retain_onto_forget


def _dash_cosine_per_output(weight: torch.Tensor, grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return cosine_against_negative_gradient(weight, grad, granularity="per_filter", eps=eps)


def gradient_overlap_stats_for_granularity(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    *,
    granularity: str,
    eps: float = 1e-12,
) -> dict[str, float]:
    return gradient_overlap_stats(
        retain_gradients,
        forget_gradients,
        granularity=granularity,
        eps=eps,
    )


def gradient_overlap_stats_per_output(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> dict[str, float]:
    stats = gradient_overlap_stats_for_granularity(
        retain_gradients,
        forget_gradients,
        granularity="per_filter",
        eps=eps,
    )
    stats["gradient_overlap_common_outputs"] = stats.pop("gradient_overlap_common_units")
    return stats


def _truncate_gradient_by_svd_evr(grad: torch.Tensor, evr_target: float) -> torch.Tensor:
    return truncate_gradient_global_by_svd_evr(grad, evr_target)


def _truncate_gradient_by_svd_evr_for_granularity(
    grad: torch.Tensor,
    evr_target: float,
    *,
    granularity: str,
    bank: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    granularity = normalize_plasticity_granularity(granularity)
    if granularity == "global":
        return truncate_gradient_global_by_svd_evr(grad, evr_target)
    return truncate_gradient_per_filter_by_bank(grad, bank or [], evr_target)


def _unwrap_gradient_result(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
        return result[0], result[1]
    return result, None


def _project_filter_bank(
    bank: dict[str, list[torch.Tensor]] | None,
    reference_gradients: dict[str, torch.Tensor],
    *,
    mode: str,
) -> dict[str, list[torch.Tensor]] | None:
    if not bank:
        return None
    projected_bank: dict[str, list[torch.Tensor]] = {}
    for name, entries in bank.items():
        reference_grad = reference_gradients.get(name)
        if reference_grad is None:
            if mode == "onto":
                projected_bank[name] = [torch.zeros_like(entry) for entry in entries]
            else:
                projected_bank[name] = [entry.clone() for entry in entries]
            continue
        reference_matrix = as_filter_matrix(reference_grad)
        bucket = []
        for entry in entries:
            if mode == "perp":
                projected_entry, _, _, _ = project_tensor_perp_reference(
                    entry,
                    reference_matrix,
                    granularity="per_filter",
                )
            elif mode == "onto":
                projected_entry, _, _, _ = project_tensor_onto_reference(
                    entry,
                    reference_matrix,
                    granularity="per_filter",
                )
            else:
                raise ValueError(f"Unsupported filter-bank projection mode: {mode}")
            bucket.append(projected_entry)
        projected_bank[name] = bucket
    return projected_bank


def compute_dash_gradients(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    changed_layers_class: list[str] | None = None,
    changed_layers_name: list[str] | None = None,
    include_bias: bool = False,
    grad_mode: str = "eval",
    grad_aggregation: str = "ema",
    ema_alpha: float = 0.05,
    force_all_params: bool = False,
    num_aug: int = 1,
    aug_mode: str = "default",
    aug_fn: DashAugFn | None = None,
    svd_truncate_evr: float | None = None,
    return_filter_bank: bool = False,
    filter_bank_cap: int = 128,
    plasticity_granularity: str = "global",
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
    """
    Compute DASH gradients over a loader using EMA or mean aggregation.

    When return_filter_bank=True and plasticity_granularity=per_filter, the function also
    returns a capped bank of batch-level row/filter gradients used for row-local EVR.
    """
    if changed_layers_class is None:
        changed_layers_class = ["linear", "conv2d"]
    if changed_layers_name is None:
        changed_layers_name = []

    grad_mode = (grad_mode or "eval").lower()
    if grad_mode not in ("eval", "train_preserve_bn"):
        raise ValueError(f"Unsupported grad_mode: {grad_mode}")

    grad_aggregation = (grad_aggregation or "ema").lower()
    if grad_aggregation not in ("ema", "mean"):
        raise ValueError(f"Unsupported grad_aggregation: {grad_aggregation}")
    if grad_aggregation == "ema" and not (0.0 < ema_alpha <= 1.0):
        raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")

    num_aug = max(1, int(num_aug))
    aug_mode = (aug_mode or "default").lower()
    if aug_mode not in ("default", "none"):
        raise ValueError(f"Unsupported aug_mode: {aug_mode}")

    plasticity_granularity = normalize_plasticity_granularity(plasticity_granularity)
    filter_bank_cap = max(1, int(filter_bank_cap))

    requires_grad_state = {name: param.requires_grad for name, param in model.named_parameters()}

    if force_all_params:
        for name, param in model.named_parameters():
            if not include_bias and name.endswith("bias"):
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        from ..utils import set_requires_grad

        set_requires_grad(
            model,
            changed_layers_class=changed_layers_class,
            changed_layers_name=changed_layers_name,
            include_bias=include_bias,
        )

    gradients = {}
    sum_gradients = {}
    filter_bank: dict[str, list[torch.Tensor]] | None = {} if return_filter_bank else None
    num_batches = 0

    training_states = _snapshot_training_mode(model)
    if grad_mode == "eval":
        model.eval()
    else:
        model.train()

    bn_states = None
    if grad_mode == "train_preserve_bn":
        bn_states = _snapshot_batchnorm_stats(model)

    try:
        with torch.enable_grad():
            for images, targets in data_loader:
                images, targets = images.to(device), targets.to(device)
                batch_grads_accum: dict[str, torch.Tensor] = {}
                for aug_idx in range(num_aug):
                    if aug_idx == 0:
                        images_aug = images
                    elif aug_fn is not None:
                        images_aug = aug_fn(images)
                    elif aug_mode == "default":
                        images_aug = _default_dash_augment(images)
                    else:
                        images_aug = images

                    batch_grads = _compute_batch_gradients_full(
                        model=model,
                        data=images_aug,
                        target=targets,
                        criterion=criterion,
                        include_bias=include_bias,
                    )
                    for name, grad in batch_grads.items():
                        if name in batch_grads_accum:
                            batch_grads_accum[name] = batch_grads_accum[name] + grad
                        else:
                            batch_grads_accum[name] = grad

                if num_aug > 1:
                    for name in list(batch_grads_accum.keys()):
                        batch_grads_accum[name] = batch_grads_accum[name] / float(num_aug)

                if filter_bank is not None and plasticity_granularity == "per_filter":
                    for name, grad in batch_grads_accum.items():
                        entries = filter_bank.setdefault(name, [])
                        if len(entries) < filter_bank_cap:
                            entries.append(as_filter_matrix(grad.detach().clone()))

                if grad_aggregation == "mean":
                    for name, grad in batch_grads_accum.items():
                        if name in sum_gradients:
                            sum_gradients[name] += grad
                        else:
                            sum_gradients[name] = grad
                else:
                    for name, grad in batch_grads_accum.items():
                        if name in gradients:
                            gradients[name] = (1.0 - ema_alpha) * gradients[name] + ema_alpha * grad
                        else:
                            gradients[name] = grad

                num_batches += 1
    finally:
        if bn_states is not None:
            _restore_batchnorm_stats(bn_states)
        _restore_training_mode(training_states)
        for name, param in model.named_parameters():
            if name in requires_grad_state:
                param.requires_grad = requires_grad_state[name]

    if num_batches == 0:
        return ({}, filter_bank or {}) if return_filter_bank else {}

    if grad_aggregation == "mean":
        gradients = {name: grad / float(num_batches) for name, grad in sum_gradients.items()}

    if svd_truncate_evr is not None:
        gradients = {
            name: _truncate_gradient_by_svd_evr_for_granularity(
                grad,
                svd_truncate_evr,
                granularity=plasticity_granularity,
                bank=None if filter_bank is None else filter_bank.get(name),
            )
            for name, grad in gradients.items()
        }

    if return_filter_bank:
        return gradients, (filter_bank or {})
    return gradients


def _dash_cosine_per_vector(
    weight: torch.Tensor,
    grad: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute cosine similarity between weight vectors and negative gradients.

    Returns a tensor of cosine similarities shaped for broadcasting to weight.
    """
    if weight.dim() == 0:
        w = weight.view(1)
        g = grad.view(1)
        denom = (w.abs() * g.abs()) + eps
        cos = (w * (-g)) / denom
        return cos.view(())

    if weight.dim() == 1:
        w = weight.view(-1, 1)
        g = grad.view(-1, 1)
        dot = (w * (-g)).sum(dim=1)
        denom = (w.norm(dim=1) * g.norm(dim=1)) + eps
        cos = dot / denom
        return cos.view(weight.shape)

    if weight.dim() == 2:
        w = weight
        g = grad
        dot = (w * (-g)).sum(dim=1)
        denom = (w.norm(dim=1) * g.norm(dim=1)) + eps
        cos = dot / denom
        return cos.view(weight.shape[0], 1)

    if weight.dim() == 3:
        w = weight.reshape(weight.shape[0], -1)
        g = grad.reshape(grad.shape[0], -1)
        dot = (w * (-g)).sum(dim=1)
        denom = (w.norm(dim=1) * g.norm(dim=1)) + eps
        cos = dot / denom
        return cos.view(weight.shape[0], 1, 1)

    if weight.dim() == 4:
        w = weight.reshape(weight.shape[0] * weight.shape[1], -1)
        g = grad.reshape(grad.shape[0] * grad.shape[1], -1)
        dot = (w * (-g)).sum(dim=1)
        denom = (w.norm(dim=1) * g.norm(dim=1)) + eps
        cos = dot / denom
        return cos.view(weight.shape[0], weight.shape[1], 1, 1)

    # Fallback: group by first dimension
    w = weight.reshape(weight.shape[0], -1)
    g = grad.reshape(grad.shape[0], -1)
    dot = (w * (-g)).sum(dim=1)
    denom = (w.norm(dim=1) * g.norm(dim=1)) + eps
    cos = dot / denom
    shape = [weight.shape[0]] + [1] * (weight.dim() - 1)
    return cos.view(*shape)


def _project_forget_perp_retain_per_vector(
    forget_grad: torch.Tensor,
    retain_grad: torch.Tensor,
    eps: float = 1e-12,
    return_mask: bool = False,
) -> tuple[torch.Tensor, int, int] | tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Project forget gradients onto the space perpendicular to retain gradients per vector.

    Returns (projected_gradient, num_vectors, num_skipped_zero).
    """
    if forget_grad.shape != retain_grad.shape:
        raise ValueError(
            f"Gradient shape mismatch: forget {tuple(forget_grad.shape)} vs retain {tuple(retain_grad.shape)}"
        )

    def _project_matrix(g_f: torch.Tensor, g_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        dot = (g_f * g_r).sum(dim=1, keepdim=True)
        denom = (g_r * g_r).sum(dim=1, keepdim=True)
        coeff = dot / (denom + eps)
        proj = g_f - coeff * g_r
        mask = denom > eps
        if mask.any():
            proj = torch.where(mask, proj, g_f)
            skipped = int((~mask).sum().item())
        else:
            skipped = int(mask.numel())
        return proj, mask, int(g_f.shape[0]), skipped

    if forget_grad.dim() == 0:
        g_f = forget_grad.view(1, 1)
        g_r = retain_grad.view(1, 1)
        proj, mask, n_vec, skipped = _project_matrix(g_f, g_r)
        if return_mask:
            return proj.view(()), mask.view(()), n_vec, skipped
        return proj.view(()), n_vec, skipped

    if forget_grad.dim() == 1:
        g_f = forget_grad.view(-1, 1)
        g_r = retain_grad.view(-1, 1)
        proj, mask, n_vec, skipped = _project_matrix(g_f, g_r)
        if return_mask:
            return proj.view_as(forget_grad), mask.view_as(forget_grad), n_vec, skipped
        return proj.view_as(forget_grad), n_vec, skipped

    if forget_grad.dim() == 2:
        proj, mask, n_vec, skipped = _project_matrix(forget_grad, retain_grad)
        if return_mask:
            return proj.view_as(forget_grad), mask.view(forget_grad.shape[0], 1), n_vec, skipped
        return proj.view_as(forget_grad), n_vec, skipped

    if forget_grad.dim() == 3:
        g_f = forget_grad.reshape(forget_grad.shape[0], -1)
        g_r = retain_grad.reshape(retain_grad.shape[0], -1)
        proj, mask, n_vec, skipped = _project_matrix(g_f, g_r)
        if return_mask:
            return proj.view_as(forget_grad), mask.view(forget_grad.shape[0], 1, 1), n_vec, skipped
        return proj.view_as(forget_grad), n_vec, skipped

    if forget_grad.dim() == 4:
        g_f = forget_grad.reshape(forget_grad.shape[0] * forget_grad.shape[1], -1)
        g_r = retain_grad.reshape(retain_grad.shape[0] * retain_grad.shape[1], -1)
        proj, mask, n_vec, skipped = _project_matrix(g_f, g_r)
        if return_mask:
            return (
                proj.view_as(forget_grad),
                mask.view(forget_grad.shape[0], forget_grad.shape[1], 1, 1),
                n_vec,
                skipped,
            )
        return proj.view_as(forget_grad), n_vec, skipped

    g_f = forget_grad.reshape(forget_grad.shape[0], -1)
    g_r = retain_grad.reshape(retain_grad.shape[0], -1)
    proj, mask, n_vec, skipped = _project_matrix(g_f, g_r)
    if return_mask:
        shape = [forget_grad.shape[0]] + [1] * (forget_grad.dim() - 1)
        return proj.view_as(forget_grad), mask.view(*shape), n_vec, skipped
    return proj.view_as(forget_grad), n_vec, skipped


def _project_retain_onto_forget_per_vector(
    retain_grad: torch.Tensor,
    forget_grad: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Project retain gradients onto forget gradients per vector.

    Returns (projected_gradient, valid_mask, num_vectors, num_skipped_zero).
    """
    if retain_grad.shape != forget_grad.shape:
        raise ValueError(
            f"Gradient shape mismatch: retain {tuple(retain_grad.shape)} vs forget {tuple(forget_grad.shape)}"
        )

    def _project_matrix(g_r: torch.Tensor, g_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        dot = (g_r * g_f).sum(dim=1, keepdim=True)
        denom = (g_f * g_f).sum(dim=1, keepdim=True)
        coeff = dot / (denom + eps)
        proj = coeff * g_f
        mask = denom > eps
        if mask.any():
            proj = torch.where(mask, proj, torch.zeros_like(proj))
            skipped = int((~mask).sum().item())
        else:
            skipped = int(mask.numel())
        return proj, mask, int(g_f.shape[0]), skipped

    if retain_grad.dim() == 0:
        g_r = retain_grad.view(1, 1)
        g_f = forget_grad.view(1, 1)
        proj, mask, n_vec, skipped = _project_matrix(g_r, g_f)
        return proj.view(()), mask.view(()), n_vec, skipped

    if retain_grad.dim() == 1:
        g_r = retain_grad.view(-1, 1)
        g_f = forget_grad.view(-1, 1)
        proj, mask, n_vec, skipped = _project_matrix(g_r, g_f)
        return proj.view_as(retain_grad), mask.view_as(retain_grad), n_vec, skipped

    if retain_grad.dim() == 2:
        proj, mask, n_vec, skipped = _project_matrix(retain_grad, forget_grad)
        return proj.view_as(retain_grad), mask.view(retain_grad.shape[0], 1), n_vec, skipped

    if retain_grad.dim() == 3:
        g_r = retain_grad.reshape(retain_grad.shape[0], -1)
        g_f = forget_grad.reshape(forget_grad.shape[0], -1)
        proj, mask, n_vec, skipped = _project_matrix(g_r, g_f)
        return proj.view_as(retain_grad), mask.view(retain_grad.shape[0], 1, 1), n_vec, skipped

    if retain_grad.dim() == 4:
        g_r = retain_grad.reshape(retain_grad.shape[0] * retain_grad.shape[1], -1)
        g_f = forget_grad.reshape(forget_grad.shape[0] * forget_grad.shape[1], -1)
        proj, mask, n_vec, skipped = _project_matrix(g_r, g_f)
        return (
            proj.view_as(retain_grad),
            mask.view(retain_grad.shape[0], retain_grad.shape[1], 1, 1),
            n_vec,
            skipped,
        )

    g_r = retain_grad.reshape(retain_grad.shape[0], -1)
    g_f = forget_grad.reshape(forget_grad.shape[0], -1)
    proj, mask, n_vec, skipped = _project_matrix(g_r, g_f)
    shape = [retain_grad.shape[0]] + [1] * (retain_grad.dim() - 1)
    return proj.view_as(retain_grad), mask.view(*shape), n_vec, skipped


def project_retain_onto_forget_per_vector(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
) -> (
    dict[str, torch.Tensor]
    | tuple[dict[str, torch.Tensor], dict[str, float]]
    | tuple[dict[str, torch.Tensor], dict[str, float], dict[str, torch.Tensor]]
):
    """
    Project retain gradients onto forget gradients per-neuron.

    If forget gradient is missing, mark the tensor as invalid (no shrink).
    """
    projected: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_zero = 0
    total_vectors = 0

    for name, g_r in retain_gradients.items():
        g_f = forget_gradients.get(name)
        if g_f is None:
            projected[name] = torch.zeros_like(g_r)
            masks[name] = torch.zeros_like(_dash_cosine_per_vector(g_r, g_r)).bool()
            skipped_missing += 1
            continue
        proj, mask, n_vec, skipped = _project_retain_onto_forget_per_vector(g_r, g_f, eps=eps)
        projected[name] = proj
        masks[name] = mask
        total_vectors += n_vec
        skipped_zero += skipped

    print(
        "✓ Projected retain gradients onto forget gradients (per-vector): "
        f"{len(projected)} tensors, skipped_missing={skipped_missing}, skipped_zero={skipped_zero}",
        flush=True,
    )
    stats = {
        "projected_tensors": float(len(projected)),
        "skipped_missing": float(skipped_missing),
        "skipped_zero": float(skipped_zero),
        "total_vectors": float(total_vectors),
    }

    if return_stats and return_masks:
        return projected, stats, masks
    if return_stats:
        return projected, stats
    if return_masks:
        return projected, {}, masks
    return projected


def project_forget_perp_retain_per_vector(
    forget_gradients: dict[str, torch.Tensor],
    retain_gradients: dict[str, torch.Tensor],
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
) -> (
    dict[str, torch.Tensor]
    | tuple[dict[str, torch.Tensor], dict[str, float]]
    | tuple[dict[str, torch.Tensor], dict[str, float], dict[str, torch.Tensor]]
):
    """
    Project forget gradients onto the space perpendicular to retain gradients (per-neuron vectors).

    If retain gradient is missing or near-zero, the forget gradient is kept as-is.
    """
    projected: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_zero = 0
    total_vectors = 0

    for name, g_f in forget_gradients.items():
        g_r = retain_gradients.get(name)
        if g_r is None:
            projected[name] = g_f
            skipped_missing += 1
            continue
        proj, mask, n_vec, skipped = _project_forget_perp_retain_per_vector(g_f, g_r, eps=eps, return_mask=True)
        projected[name] = proj
        masks[name] = mask
        total_vectors += n_vec
        skipped_zero += skipped

    print(
        "✓ Projected forget gradients perpendicular to retain gradients "
        f"(per-vector): {len(projected)} tensors, skipped_missing={skipped_missing}, skipped_zero={skipped_zero}",
        flush=True,
    )
    stats = {
        "projected_tensors": float(len(projected)),
        "skipped_missing": float(skipped_missing),
        "skipped_zero": float(skipped_zero),
        "total_vectors": float(total_vectors),
    }
    if return_stats and return_masks:
        return projected, stats, masks
    if return_stats:
        return projected, stats
    if return_masks:
        return projected, {}, masks
    return projected


def project_retain_perp_forget_per_vector(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
) -> (
    dict[str, torch.Tensor]
    | tuple[dict[str, torch.Tensor], dict[str, float]]
    | tuple[dict[str, torch.Tensor], dict[str, float], dict[str, torch.Tensor]]
):
    """
    Project retain gradients onto the space perpendicular to forget gradients (per-neuron vectors).

    If forget gradient is missing or near-zero, the retain gradient is kept as-is.
    """
    projected: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_zero = 0
    total_vectors = 0

    for name, g_r in retain_gradients.items():
        g_f = forget_gradients.get(name)
        if g_f is None:
            projected[name] = g_r
            skipped_missing += 1
            continue
        proj, mask, n_vec, skipped = _project_forget_perp_retain_per_vector(g_r, g_f, eps=eps, return_mask=True)
        projected[name] = proj
        masks[name] = mask
        total_vectors += n_vec
        skipped_zero += skipped

    print(
        "✓ Projected retain gradients perpendicular to forget gradients "
        f"(per-vector): {len(projected)} tensors, skipped_missing={skipped_missing}, skipped_zero={skipped_zero}",
        flush=True,
    )
    stats = {
        "projected_tensors": float(len(projected)),
        "skipped_missing": float(skipped_missing),
        "skipped_zero": float(skipped_zero),
        "total_vectors": float(total_vectors),
    }
    if return_stats and return_masks:
        return projected, stats, masks
    if return_stats:
        return projected, stats
    if return_masks:
        return projected, {}, masks
    return projected


def apply_dash_warm_start(
    model: nn.Module,
    data_loader: DataLoader,
    forget_loader: DataLoader | None = None,
    project_retain_to_forget: bool = False,
    project_forget_perp_retain: bool = False,
    project_retain_perp_forget: bool = False,
    criterion: nn.Module = None,
    device: torch.device = None,
    changed_layers_class: list[str] | None = None,
    changed_layers_name: list[str] | None = None,
    include_bias: bool = False,
    grad_mode: str = "eval",
    grad_aggregation: str = "ema",
    signal_mode: str = "retain_only",
    ema_alpha: float = 0.05,
    min_shrink: float = 0.05,
    force_all_params: bool = False,
    wandb_logger=None,
    log_cosine_histograms: bool = False,
    cosine_hist_bins: int = 50,
    log_prefix: str = "dash",
    log_step: int = -1,
    compute_gradients_fn=None,
    num_aug: int = 1,
    aug_mode: str = "default",
    aug_fn: DashAugFn | None = None,
    svd_truncate_evr: float | None = None,
    preserve_forget_evr: float = 0.95,
    plasticity_granularity: str = "per_filter",
) -> dict[str, float]:
    """
    Apply DASH-style direction-aware shrinking.

    The shared plasticity_granularity flag is the single source of truth for projection,
    overlap, truncation, preserve/complement masks, and shrink semantics.
    """
    if data_loader is None:
        raise ValueError("data_loader must be provided for DASH warm-start")
    if compute_gradients_fn is None:
        compute_gradients_fn = compute_dash_gradients

    signal_mode = str(signal_mode or "retain_only").lower()
    if signal_mode not in {"retain_only", "forget_perp_retain", "preserve_complement"}:
        raise ValueError(f"Unsupported DASH signal_mode: {signal_mode}")
    plasticity_granularity = normalize_plasticity_granularity(plasticity_granularity)

    legacy_projection_requested = any(
        (project_retain_to_forget, project_forget_perp_retain, project_retain_perp_forget)
    )
    if signal_mode != "retain_only" and legacy_projection_requested:
        print(
            f"⚠️  DASH warm-start: ignoring legacy projection flags because dash_signal_mode={signal_mode}.",
            flush=True,
        )

    if signal_mode == "retain_only" and project_retain_to_forget and (
        project_forget_perp_retain or project_retain_perp_forget
    ):
        raise ValueError("DASH warm-start: project_retain_to_forget cannot be combined with other projections.")
    allow_dual_projection = signal_mode == "retain_only" and project_forget_perp_retain and project_retain_perp_forget

    request_filter_bank = plasticity_granularity == "per_filter"

    def _compute_gradients(loader):
        kwargs = dict(
            model=model,
            data_loader=loader,
            criterion=criterion,
            device=device,
            changed_layers_class=changed_layers_class,
            changed_layers_name=changed_layers_name,
            include_bias=include_bias,
            grad_mode=grad_mode,
            grad_aggregation=grad_aggregation,
            ema_alpha=ema_alpha,
            force_all_params=force_all_params,
            num_aug=num_aug,
            aug_mode=aug_mode,
            aug_fn=aug_fn,
        )
        if request_filter_bank:
            try:
                result = compute_gradients_fn(
                    **kwargs,
                    return_filter_bank=True,
                    plasticity_granularity=plasticity_granularity,
                )
            except TypeError:
                result = compute_gradients_fn(**kwargs)
        else:
            result = compute_gradients_fn(**kwargs)
        return _unwrap_gradient_result(result)

    def _full_mask(reference: torch.Tensor, active: bool) -> torch.Tensor:
        units = int(as_granularity_matrix(reference, plasticity_granularity).shape[0])
        values = torch.full((units,), active, device=reference.device, dtype=torch.bool)
        return expand_decision_values(values, reference, plasticity_granularity).bool()

    retain_gradients, retain_filter_bank = _compute_gradients(data_loader)

    projection_stats = {}
    overlap_stats = {}
    base_gradients = retain_gradients
    base_filter_bank = retain_filter_bank
    invert_gradients = None
    invert_filter_bank = None
    normal_gradients = None
    normal_filter_bank = None
    invert_masks = None
    normal_masks = None
    forget_gradients = None
    forget_filter_bank = None

    if signal_mode != "retain_only":
        if forget_loader is None:
            raise ValueError(f"forget_loader must be provided when dash_signal_mode={signal_mode}")
        forget_gradients, forget_filter_bank = _compute_gradients(forget_loader)
        overlap_stats = gradient_overlap_stats_for_granularity(
            retain_gradients,
            forget_gradients,
            granularity=plasticity_granularity,
        )
        if signal_mode == "forget_perp_retain":
            invert_gradients, stats, invert_masks = _PROJECT_FORGET_PERP_RETAIN(
                forget_gradients=forget_gradients,
                retain_gradients=retain_gradients,
                granularity=plasticity_granularity,
                return_stats=True,
                return_masks=True,
            )
            projection_stats.update({f"forget_perp_retain/{k}": v for k, v in stats.items()})
            if request_filter_bank:
                invert_filter_bank = _project_filter_bank(forget_filter_bank, retain_gradients, mode="perp")
        else:
            forget_preserve_reference = {
                name: _truncate_gradient_by_svd_evr_for_granularity(
                    grad,
                    preserve_forget_evr,
                    granularity=plasticity_granularity,
                    bank=None if forget_filter_bank is None else forget_filter_bank.get(name),
                )
                for name, grad in forget_gradients.items()
            }
            normal_gradients, stats, normal_masks = _PROJECT_RETAIN_PERP_FORGET(
                retain_gradients=retain_gradients,
                forget_gradients=forget_preserve_reference,
                granularity=plasticity_granularity,
                return_stats=True,
                return_masks=True,
            )
            projection_stats.update({f"retain_perp_forget/{k}": v for k, v in stats.items()})
            if request_filter_bank:
                normal_filter_bank = _project_filter_bank(retain_filter_bank, forget_preserve_reference, mode="perp")
    elif project_forget_perp_retain:
        if forget_loader is None:
            raise ValueError("forget_loader must be provided when project_forget_perp_retain=True")
        forget_gradients, forget_filter_bank = _compute_gradients(forget_loader)
        invert_gradients, stats, invert_masks = project_forget_perp_retain(
            forget_gradients=forget_gradients,
            retain_gradients=retain_gradients,
            granularity=plasticity_granularity,
            return_stats=True,
            return_masks=True,
        )
        projection_stats.update({f"forget_perp_retain/{k}": v for k, v in stats.items()})
        if request_filter_bank:
            invert_filter_bank = _project_filter_bank(forget_filter_bank, retain_gradients, mode="perp")
    elif project_retain_to_forget:
        if forget_loader is None:
            print("⚠️  DASH warm-start: forget_loader missing; skipping projection.", flush=True)
        else:
            forget_gradients, forget_filter_bank = _compute_gradients(forget_loader)
            normal_gradients, stats, normal_masks = _PROJECT_RETAIN_ONTO_FORGET(
                retain_gradients=retain_gradients,
                forget_gradients=forget_gradients,
                granularity=plasticity_granularity,
                return_stats=True,
                return_masks=True,
            )
            projection_stats.update({f"retain_to_forget/{k}": v for k, v in stats.items()})
            if request_filter_bank:
                normal_filter_bank = _project_filter_bank(retain_filter_bank, forget_gradients, mode="onto")
    elif project_retain_perp_forget:
        if forget_loader is None:
            raise ValueError("forget_loader must be provided when project_retain_perp_forget=True")
        forget_gradients, forget_filter_bank = _compute_gradients(forget_loader)
        normal_gradients, stats, normal_masks = project_retain_perp_forget(
            retain_gradients=retain_gradients,
            forget_gradients=forget_gradients,
            granularity=plasticity_granularity,
            return_stats=True,
            return_masks=True,
        )
        projection_stats.update({f"retain_perp_forget/{k}": v for k, v in stats.items()})
        if request_filter_bank:
            normal_filter_bank = _project_filter_bank(retain_filter_bank, forget_gradients, mode="perp")

    if allow_dual_projection:
        if forget_loader is None:
            raise ValueError("forget_loader must be provided for dual projection.")
        if forget_gradients is None:
            forget_gradients, forget_filter_bank = _compute_gradients(forget_loader)
        if normal_gradients is None:
            normal_gradients, stats, normal_masks = _PROJECT_RETAIN_PERP_FORGET(
                retain_gradients=retain_gradients,
                forget_gradients=forget_gradients,
                granularity=plasticity_granularity,
                return_stats=True,
                return_masks=True,
            )
            projection_stats.update({f"retain_perp_forget/{k}": v for k, v in stats.items()})
            if request_filter_bank:
                normal_filter_bank = _project_filter_bank(retain_filter_bank, forget_gradients, mode="perp")
        if invert_gradients is None:
            invert_gradients, stats, invert_masks = _PROJECT_FORGET_PERP_RETAIN(
                forget_gradients=forget_gradients,
                retain_gradients=retain_gradients,
                granularity=plasticity_granularity,
                return_stats=True,
                return_masks=True,
            )
            projection_stats.update({f"forget_perp_retain/{k}": v for k, v in stats.items()})
            if request_filter_bank:
                invert_filter_bank = _project_filter_bank(forget_filter_bank, retain_gradients, mode="perp")

    if signal_mode == "retain_only" and invert_masks is not None and normal_masks is not None:
        for name, mask in normal_masks.items():
            if name in invert_masks:
                normal_masks[name] = mask & ~invert_masks[name]

    if svd_truncate_evr is not None:
        if invert_gradients is not None:
            invert_gradients = {
                name: _truncate_gradient_by_svd_evr_for_granularity(
                    grad,
                    svd_truncate_evr,
                    granularity=plasticity_granularity,
                    bank=None if invert_filter_bank is None else invert_filter_bank.get(name),
                )
                for name, grad in invert_gradients.items()
            }
        if normal_gradients is not None:
            normal_gradients = {
                name: _truncate_gradient_by_svd_evr_for_granularity(
                    grad,
                    svd_truncate_evr,
                    granularity=plasticity_granularity,
                    bank=None if normal_filter_bank is None else normal_filter_bank.get(name),
                )
                for name, grad in normal_gradients.items()
            }
        if invert_gradients is None and normal_gradients is None:
            base_gradients = {
                name: _truncate_gradient_by_svd_evr_for_granularity(
                    grad,
                    svd_truncate_evr,
                    granularity=plasticity_granularity,
                    bank=None if base_filter_bank is None else base_filter_bank.get(name),
                )
                for name, grad in base_gradients.items()
            }

    if invert_gradients is None and normal_gradients is None and not base_gradients:
        print("⚠️  DASH warm-start: no gradients computed (empty selection).", flush=True)
        return {"updated_params": 0.0, "updated_tensors": 0.0}

    updated_params = 0
    updated_tensors = 0
    cos_total = 0.0
    cos_count = 0
    cos_global_min = None
    cos_global_max = None
    preserve_fraction_total = 0.0
    preserve_fraction_count = 0

    min_shrink = float(min_shrink)
    if min_shrink < 0:
        raise ValueError(f"min_shrink must be >= 0, got {min_shrink}")

    log_prefix = (log_prefix or "dash").rstrip("/")
    cosine_hist_bins = max(int(cosine_hist_bins), 1)
    cosine_stats = {}
    cosine_histograms = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if (
                (invert_gradients is not None and name in invert_gradients)
                or (normal_gradients is not None and name in normal_gradients)
                or (invert_gradients is None and normal_gradients is None and name in base_gradients)
            ):
                pass
            else:
                continue
            if not include_bias and name.endswith("bias"):
                continue

            min_shrink_tensor = torch.tensor(min_shrink, device=param.device, dtype=param.dtype)

            if signal_mode == "preserve_complement":
                preserve_grad = None if normal_gradients is None else normal_gradients.get(name)
                if preserve_grad is not None and preserve_grad.shape != param.shape:
                    print(
                        f"⚠️  DASH warm-start: shape mismatch for {name} "
                        f"(param {tuple(param.shape)} vs preserve_grad {tuple(preserve_grad.shape)}), skipping.",
                        flush=True,
                    )
                    continue
                preserve_mask = normal_masks.get(name, _full_mask(param.data, False)) if normal_masks is not None else _full_mask(param.data, False)
                shrink = torch.full_like(param.data, float(min_shrink))
                cos_used = torch.zeros_like(param.data)
                if preserve_grad is not None:
                    cos_preserve = cosine_against_negative_gradient(
                        param.data,
                        preserve_grad,
                        granularity=plasticity_granularity,
                    )
                    cos_preserve = torch.clamp(cos_preserve, min=-1.0, max=1.0)
                    shrink_preserve = torch.maximum(cos_preserve, min_shrink_tensor)
                    shrink = torch.where(preserve_mask, shrink_preserve, shrink)
                    cos_used = torch.where(preserve_mask, cos_preserve, cos_used)
                preserve_fraction_total += float(preserve_mask.float().mean().item())
                preserve_fraction_count += 1
            elif signal_mode == "forget_perp_retain":
                grad = None if invert_gradients is None else invert_gradients.get(name)
                if grad is None:
                    continue
                if grad.shape != param.shape:
                    print(
                        f"⚠️  DASH warm-start: shape mismatch for {name} "
                        f"(param {tuple(param.shape)} vs grad {tuple(grad.shape)}), skipping.",
                        flush=True,
                    )
                    continue
                mask = invert_masks.get(name) if invert_masks is not None else _full_mask(param.data, True)
                cos_inv = cosine_against_negative_gradient(
                    param.data,
                    grad,
                    granularity=plasticity_granularity,
                )
                cos_inv = torch.clamp(cos_inv, min=-1.0, max=1.0)
                shrink_inv = min_shrink_tensor + (1.0 - min_shrink_tensor) * (1.0 - cos_inv) * 0.5
                shrink = torch.where(mask, shrink_inv, torch.ones_like(shrink_inv))
                cos_used = torch.where(mask, cos_inv, torch.ones_like(cos_inv))
            else:
                if invert_gradients is not None and name in invert_gradients:
                    grad = invert_gradients[name]
                elif normal_gradients is not None and name in normal_gradients:
                    grad = normal_gradients[name]
                else:
                    grad = base_gradients[name]
                if grad is None or grad.shape != param.shape:
                    print(
                        f"⚠️  DASH warm-start: shape mismatch for {name} "
                        f"(param {tuple(param.shape)} vs grad {tuple(grad.shape)}), skipping.",
                        flush=True,
                    )
                    continue

                if invert_gradients is None and normal_gradients is None:
                    cos = cosine_against_negative_gradient(
                        param.data,
                        grad,
                        granularity=plasticity_granularity,
                    )
                    cos = torch.clamp(cos, min=-1.0, max=1.0)
                    shrink = torch.maximum(cos, min_shrink_tensor)
                    cos_used = cos
                else:
                    cos_template = None
                    shrink = None
                    cos_used = None

                    if invert_gradients is not None and name in invert_gradients:
                        cos_inv = cosine_against_negative_gradient(
                            param.data,
                            invert_gradients[name],
                            granularity=plasticity_granularity,
                        )
                        cos_inv = torch.clamp(cos_inv, min=-1.0, max=1.0)
                        cos_template = cos_inv
                        shrink_inv = min_shrink_tensor + (1.0 - min_shrink_tensor) * (1.0 - cos_inv) * 0.5
                        if invert_masks is not None and name in invert_masks:
                            mask = invert_masks[name]
                            cos_used = torch.where(mask, cos_inv, torch.ones_like(cos_inv))
                            shrink = torch.where(mask, shrink_inv, torch.ones_like(shrink_inv))
                        else:
                            cos_used = cos_inv
                            shrink = shrink_inv

                    if normal_gradients is not None and name in normal_gradients:
                        cos_norm = cosine_against_negative_gradient(
                            param.data,
                            normal_gradients[name],
                            granularity=plasticity_granularity,
                        )
                        cos_norm = torch.clamp(cos_norm, min=-1.0, max=1.0)
                        if cos_template is None:
                            cos_template = cos_norm
                        shrink_norm = torch.maximum(cos_norm, min_shrink_tensor)
                        if normal_masks is not None and name in normal_masks:
                            mask = normal_masks[name]
                            if cos_used is None:
                                cos_used = torch.where(mask, cos_norm, torch.ones_like(cos_norm))
                                shrink = torch.where(mask, shrink_norm, torch.ones_like(shrink_norm))
                            else:
                                cos_used = torch.where(mask, cos_norm, cos_used)
                                shrink = torch.where(mask, shrink_norm, shrink)
                        else:
                            cos_used = cos_norm
                            shrink = shrink_norm

                    if cos_used is None:
                        cos_used = torch.ones_like(cos_template)
                    if shrink is None:
                        shrink = torch.ones_like(cos_template)

            param.data.mul_(shrink)
            updated_tensors += 1
            updated_params += param.numel()

            cos_flat = cos_used.detach().flatten()
            cos_sum = float(cos_flat.sum().item())
            cos_numel = int(cos_flat.numel())
            cos_total += cos_sum
            cos_count += cos_numel
            cos_min = float(cos_flat.min().item())
            cos_max = float(cos_flat.max().item())
            cos_mean = cos_sum / max(cos_numel, 1)
            cos_std = float(cos_flat.std(unbiased=False).item()) if cos_numel > 1 else 0.0

            if cos_global_min is None or cos_min < cos_global_min:
                cos_global_min = cos_min
            if cos_global_max is None or cos_max > cos_global_max:
                cos_global_max = cos_max

            if wandb_logger is not None:
                cosine_stats[f"{log_prefix}_cosine_mean/{name}"] = cos_mean
                cosine_stats[f"{log_prefix}_cosine_std/{name}"] = cos_std
                cosine_stats[f"{log_prefix}_cosine_min/{name}"] = cos_min
                cosine_stats[f"{log_prefix}_cosine_max/{name}"] = cos_max
                if log_cosine_histograms:
                    try:
                        import wandb

                        hist = wandb.Histogram(cos_flat.cpu().numpy(), num_bins=int(cosine_hist_bins))
                    except Exception:
                        hist = cos_flat.cpu().numpy()
                    cosine_histograms[f"{log_prefix}_cosine_hist/{name}"] = hist

    cos_mean = cos_total / max(cos_count, 1)
    stats = {
        "updated_params": float(updated_params),
        "updated_tensors": float(updated_tensors),
        "cos_mean": float(cos_mean),
        "cos_min": float(cos_global_min) if cos_global_min is not None else 0.0,
        "cos_max": float(cos_global_max) if cos_global_max is not None else 0.0,
        "granularity_is_global": float(plasticity_granularity == "global"),
        "granularity_is_per_filter": float(plasticity_granularity == "per_filter"),
    }
    if projection_stats is not None:
        stats.update({f"projection/{k}": v for k, v in projection_stats.items()})
    if overlap_stats:
        stats.update(overlap_stats)
    if preserve_fraction_count > 0:
        preserve_fraction = preserve_fraction_total / float(preserve_fraction_count)
        stats["preserve_fraction"] = float(preserve_fraction)
        stats["complement_fraction"] = float(1.0 - preserve_fraction)
    if signal_mode == "preserve_complement":
        stats["preserve_forget_evr"] = float(preserve_forget_evr)

    print(
        f"✓ DASH warm-start applied: {updated_tensors} tensors, "
        f"{updated_params:,} params, mean cos={stats['cos_mean']:.4f}",
        flush=True,
    )

    if wandb_logger is not None:
        summary_stats = {
            f"{log_prefix}_summary/updated_params": float(updated_params),
            f"{log_prefix}_summary/updated_tensors": float(updated_tensors),
            f"{log_prefix}_summary/cos_mean": float(stats["cos_mean"]),
            f"{log_prefix}_summary/cos_min": float(stats["cos_min"]),
            f"{log_prefix}_summary/cos_max": float(stats["cos_max"]),
            f"{log_prefix}_summary/granularity_is_global": float(plasticity_granularity == "global"),
            f"{log_prefix}_summary/granularity_is_per_filter": float(plasticity_granularity == "per_filter"),
        }
        for key, val in stats.items():
            if key in {"updated_params", "updated_tensors", "cos_mean", "cos_min", "cos_max", "granularity_is_global", "granularity_is_per_filter"}:
                continue
            if isinstance(val, (int, float)):
                summary_stats[f"{log_prefix}_summary/{key}"] = float(val)
        if projection_stats is not None:
            for key, val in projection_stats.items():
                summary_stats[f"{log_prefix}_projection/{key}"] = float(val)

        if cosine_stats:
            summary_stats.update(cosine_stats)
        wandb_logger.log_scalars(summary_stats, step=log_step)
        if cosine_histograms:
            wandb_logger.log_histograms(cosine_histograms)

    return stats
