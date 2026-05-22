"""Shared helpers for DASH/RS-FIRE plasticity granularity handling."""

from __future__ import annotations

from typing import Iterable

import torch


def normalize_plasticity_granularity(granularity: str | None) -> str:
    granularity = str(granularity or "per_filter").lower()
    if granularity not in {"global", "per_filter"}:
        raise ValueError(
            f"Unsupported plasticity_granularity: {granularity}. Expected 'global' or 'per_filter'."
        )
    return granularity


def as_filter_matrix(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.view(1, 1)
    if tensor.ndim == 1:
        return tensor.view(-1, 1)
    return tensor.reshape(tensor.shape[0], -1)


def as_granularity_matrix(tensor: torch.Tensor, granularity: str | None) -> torch.Tensor:
    granularity = normalize_plasticity_granularity(granularity)
    if granularity == "global":
        return tensor.reshape(1, -1)
    return as_filter_matrix(tensor)


def _decision_shape(reference: torch.Tensor, granularity: str | None) -> tuple[int, ...]:
    granularity = normalize_plasticity_granularity(granularity)
    if reference.ndim == 0:
        return ()
    if granularity == "global":
        return (1,) * reference.ndim
    if reference.ndim == 1:
        return tuple(reference.shape)
    return (reference.shape[0],) + (1,) * (reference.ndim - 1)


def expand_decision_values(
    values: torch.Tensor,
    reference: torch.Tensor,
    granularity: str | None,
) -> torch.Tensor:
    if reference.ndim == 0:
        return values.view(())
    return values.view(*_decision_shape(reference, granularity))


def project_tensor_perp_reference(
    primary: torch.Tensor,
    reference: torch.Tensor,
    *,
    granularity: str | None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if primary.shape != reference.shape:
        raise ValueError(
            f"Tensor shape mismatch for projection: primary {tuple(primary.shape)} vs reference {tuple(reference.shape)}"
        )

    primary_matrix = as_granularity_matrix(primary, granularity)
    reference_matrix = as_granularity_matrix(reference, granularity)
    dot = (primary_matrix * reference_matrix).sum(dim=1, keepdim=True)
    denom = (reference_matrix * reference_matrix).sum(dim=1, keepdim=True)
    coeff = dot / (denom + eps)
    projected = primary_matrix - coeff * reference_matrix
    projected = torch.where(denom > eps, projected, primary_matrix)
    active = (projected.norm(dim=1, keepdim=True) > eps).view(-1)
    return (
        projected.reshape_as(primary),
        expand_decision_values(active, primary, granularity).bool(),
        int((~active).sum().item()),
        int(active.numel()),
    )


def project_tensor_onto_reference(
    primary: torch.Tensor,
    reference: torch.Tensor,
    *,
    granularity: str | None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if primary.shape != reference.shape:
        raise ValueError(
            f"Tensor shape mismatch for projection: primary {tuple(primary.shape)} vs reference {tuple(reference.shape)}"
        )

    primary_matrix = as_granularity_matrix(primary, granularity)
    reference_matrix = as_granularity_matrix(reference, granularity)
    dot = (primary_matrix * reference_matrix).sum(dim=1, keepdim=True)
    denom = (reference_matrix * reference_matrix).sum(dim=1, keepdim=True)
    coeff = dot / (denom + eps)
    projected = coeff * reference_matrix
    active = denom.view(-1) > eps
    projected = torch.where(denom > eps, projected, torch.zeros_like(projected))
    return (
        projected.reshape_as(primary),
        expand_decision_values(active, primary, granularity).bool(),
        int((~active).sum().item()),
        int(active.numel()),
    )


def cosine_against_negative_gradient(
    weight: torch.Tensor,
    grad: torch.Tensor,
    *,
    granularity: str | None,
    eps: float = 1e-12,
) -> torch.Tensor:
    weight_matrix = as_granularity_matrix(weight, granularity)
    grad_matrix = as_granularity_matrix(grad, granularity)
    dot = (weight_matrix * (-grad_matrix)).sum(dim=1)
    denom = (weight_matrix.norm(dim=1) * grad_matrix.norm(dim=1)) + eps
    return expand_decision_values(dot / denom, weight, granularity)


def gradient_overlap_stats(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    *,
    granularity: str | None,
    eps: float = 1e-12,
) -> dict[str, float]:
    overlaps = []
    common_units = 0

    for name, retain_grad in retain_gradients.items():
        forget_grad = forget_gradients.get(name)
        if forget_grad is None or forget_grad.shape != retain_grad.shape:
            continue
        retain_matrix = as_granularity_matrix(retain_grad, granularity).float()
        forget_matrix = as_granularity_matrix(forget_grad, granularity).float()
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


def truncate_gradient_global_by_svd_evr(grad: torch.Tensor, evr_target: float) -> torch.Tensor:
    evr_target = float(evr_target)
    if not (0.0 < evr_target <= 1.0):
        raise ValueError(f"svd_truncate_evr must be in (0, 1], got {evr_target}")
    if grad.ndim <= 1:
        return grad

    original_shape = grad.shape
    matrix = grad if grad.ndim == 2 else grad.reshape(grad.shape[0], -1)
    if matrix.numel() == 0:
        return grad

    compute_dtype = torch.float32 if matrix.dtype in (torch.float16, torch.bfloat16) else matrix.dtype
    matrix_compute = matrix.detach().to(dtype=compute_dtype)
    u, s, vh = torch.linalg.svd(matrix_compute, full_matrices=False)
    if s.numel() == 0:
        return grad

    spectrum = s.square()
    total = spectrum.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return grad

    cumulative = torch.cumsum(spectrum, dim=0) / total
    threshold = torch.tensor(evr_target, device=cumulative.device, dtype=cumulative.dtype)
    rank = int(torch.searchsorted(cumulative, threshold).item()) + 1
    rank = max(1, min(rank, int(s.shape[0])))
    truncated = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    return truncated.to(dtype=grad.dtype).reshape(original_shape)


def truncate_gradient_per_filter_by_bank(
    grad: torch.Tensor,
    bank: Iterable[torch.Tensor],
    evr_target: float,
) -> torch.Tensor:
    evr_target = float(evr_target)
    if not (0.0 < evr_target <= 1.0):
        raise ValueError(f"svd_truncate_evr must be in (0, 1], got {evr_target}")

    grad_matrix = as_filter_matrix(grad)
    if grad_matrix.numel() == 0:
        return grad

    bank_matrices = [as_filter_matrix(sample).detach() for sample in bank]
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

    return truncated.reshape_as(grad)
