"""Gradient computation utilities for model unlearning."""

import copy
import inspect

import torch
import torch.nn as nn
from torch import linalg as LA
from torch.utils.data import DataLoader

from ..dash.dash_utils import apply_dash_warm_start as _apply_dash_warm_start_impl
from ..dash.dash_utils import compute_dash_gradients
from ..dash.dash_utils import project_retain_onto_forget_per_vector as _project_retain_onto_forget_per_vector_impl
from ..dash.dash_utils import project_retain_perp_forget_per_vector as _project_retain_perp_forget_per_vector_impl
from ..utils import get_module_by_name


def compute_sum_gradients_from_class(
    model: nn.Module,
    full_train_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_class: int,
    changed_layers_class: list[str] = None,
    use_projection: bool = False,
    gradient_checkpoint_path: str = None,
    clone_model_for_gradients: bool = False,
    per_channel_projection: bool = False,
    channelwise_info: dict[str, bool] = None,
    grad_mode: str = "eval",
) -> dict[str, torch.Tensor]:
    """
    Compute gradients using only samples from a specific class.

    This function filters the full training loader to only include samples
    from the target class, then computes gradients on those samples.
    Useful for cross-class gradient experiments where SVD uses gradients
    from class X but the model forgets class Y.

    Args:
        model: The neural network model
        full_train_loader: FULL training data loader (not filtered to forget set)
        criterion: Loss function
        device: Device to compute on
        target_class: Class ID to use for gradient computation
        changed_layers_class: List of layer class names for which parameters should remain trainable
        use_projection: If True, project gradients onto space perpendicular to weights
        gradient_checkpoint_path: Path to checkpoint with weights for gradient computation
        clone_model_for_gradients: If True, clone model before loading checkpoint weights
        per_channel_projection: If True, use per-channel projection for Conv2d layers
        channelwise_info: Dictionary mapping layer names to whether they use channelwise SVD

    Returns:
        Dictionary mapping layer names to their summed gradients (computed on target_class only)
    """
    print(f"\n=== Computing gradients from CLASS {target_class} ===", flush=True)

    # Get the underlying dataset from the loader
    if hasattr(full_train_loader.dataset, "dataset"):
        # It's a Subset or wrapped dataset
        base_dataset = full_train_loader.dataset.dataset
        existing_indices = (
            full_train_loader.dataset.indices
            if hasattr(full_train_loader.dataset, "indices")
            else range(len(base_dataset))
        )
    else:
        base_dataset = full_train_loader.dataset
        existing_indices = range(len(base_dataset))

    # Filter to only target_class
    class_indices = []
    for idx in existing_indices:
        _, label = base_dataset[idx]
        if label == target_class:
            class_indices.append(idx)

    if len(class_indices) == 0:
        raise ValueError(
            f"No samples found for class {target_class} in full_train_loader. "
            f"Dataset has {len(base_dataset)} samples total."
        )

    print(
        f"  Found {len(class_indices)} samples for class {target_class} "
        f"(out of {len(list(existing_indices))} total in loader)",
        flush=True,
    )

    # Create subset
    class_subset = torch.utils.data.Subset(base_dataset, class_indices)

    # Create filtered loader
    class_loader = torch.utils.data.DataLoader(
        class_subset,
        batch_size=full_train_loader.batch_size,
        shuffle=False,
        num_workers=full_train_loader.num_workers,
        pin_memory=full_train_loader.pin_memory,
    )

    # Use existing compute_sum_gradients
    gradients = compute_sum_gradients(
        model=model,
        data_loader=class_loader,
        criterion=criterion,
        device=device,
        changed_layers_class=changed_layers_class,
        use_projection=use_projection,
        gradient_checkpoint_path=gradient_checkpoint_path,
        clone_model_for_gradients=clone_model_for_gradients,
        per_channel_projection=per_channel_projection,
        channelwise_info=channelwise_info,
        grad_mode=grad_mode,
    )

    print(f"  ✓ Computed gradients from {len(class_indices)} samples of class {target_class}", flush=True)

    return gradients


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


def compute_sum_gradients(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    changed_layers_class: list[str] = None,
    use_projection: bool = False,
    gradient_checkpoint_path: str = None,
    clone_model_for_gradients: bool = False,
    per_channel_projection: bool = False,
    channelwise_info: dict[str, bool] = None,
    grad_mode: str = "eval",
) -> dict[str, torch.Tensor]:
    """
    Compute sum of gradients across all batches in the data loader.

    Args:
        model: The neural network model
        data_loader: DataLoader with the dataset
        criterion: Loss function
        device: Device to compute on
        changed_layers_class: List of layer class names for which parameters should remain trainable
        use_projection: If True, project gradients onto space perpendicular to weights
        gradient_checkpoint_path: Path to checkpoint with weights for gradient computation
        clone_model_for_gradients: If True, clone model before loading checkpoint weights
        per_channel_projection: If True, use per-channel projection for Conv2d layers
        channelwise_info: Dictionary mapping layer names to whether they use channelwise SVD

    Returns:
        Dictionary mapping layer names to their summed gradients
    """
    if changed_layers_class is None:
        changed_layers_class = ["linear", "conv2d"]

    # Prepare model for gradient computation
    gradient_model = _prepare_gradient_model(
        model=model,
        device=device,
        gradient_checkpoint_path=gradient_checkpoint_path,
        clone_model_for_gradients=clone_model_for_gradients,
    )

    # Set requires_grad for specified layers
    from ..utils import set_requires_grad

    set_requires_grad(gradient_model, changed_layers_class=changed_layers_class)

    # Compute gradients
    print(f"\n=== Computing gradients ===", flush=True)
    print(
        f"  Using {'cloned' if clone_model_for_gradients and gradient_checkpoint_path else 'original'} model",
        flush=True,
    )

    sum_gradients = None
    num_batches = 0

    grad_mode = (grad_mode or "eval").lower()
    if grad_mode not in ("eval", "train_preserve_bn"):
        raise ValueError(f"Unsupported grad_mode: {grad_mode}")

    training_states = _snapshot_training_mode(gradient_model)
    if grad_mode == "eval":
        gradient_model.eval()

    bn_states = None
    if grad_mode == "train_preserve_bn":
        bn_states = _snapshot_batchnorm_stats(gradient_model)
    try:
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            gradients = _compute_batch_gradients(gradient_model, images, targets, criterion)

            if sum_gradients is None:
                sum_gradients = gradients
            else:
                for key, val in gradients.items():
                    sum_gradients[key] += val
            num_batches += 1

        print(f"  Processed {num_batches} batches", flush=True)
    finally:
        if bn_states is not None:
            _restore_batchnorm_stats(bn_states)
        _restore_training_mode(training_states)

    # Optional: project gradients onto space perpendicular to weights
    if use_projection and sum_gradients is not None:
        sum_gradients = project_gradients_perpendicular(
            model=gradient_model,
            gradients=sum_gradients,
            per_channel_projection=per_channel_projection,
            channelwise_info=channelwise_info,
        )

    # Clean up cloned model if it was created
    if clone_model_for_gradients and gradient_checkpoint_path is not None:
        print("\n  Cleaning up cloned model...", flush=True)
        del gradient_model
        torch.cuda.empty_cache()
        print("  Memory freed", flush=True)

    return sum_gradients


def project_retain_onto_forget_per_vector(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
):
    """Thin wrapper for DASH per-vector projection (retain onto forget)."""
    return _project_retain_onto_forget_per_vector_impl(
        retain_gradients=retain_gradients,
        forget_gradients=forget_gradients,
        eps=eps,
        return_stats=return_stats,
        return_masks=return_masks,
    )


def project_retain_perp_forget_per_vector(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    eps: float = 1e-12,
    return_stats: bool = False,
    return_masks: bool = False,
):
    """Thin wrapper for DASH per-vector projection (retain perp forget)."""
    return _project_retain_perp_forget_per_vector_impl(
        retain_gradients=retain_gradients,
        forget_gradients=forget_gradients,
        eps=eps,
        return_stats=return_stats,
        return_masks=return_masks,
    )


def _prepare_gradient_model(
    model: nn.Module,
    device: torch.device,
    gradient_checkpoint_path: str = None,
    clone_model_for_gradients: bool = False,
) -> nn.Module:
    """
    Prepare model for gradient computation, optionally loading from checkpoint.

    Args:
        model: Original model
        device: Device to use
        gradient_checkpoint_path: Path to checkpoint
        clone_model_for_gradients: Whether to clone model

    Returns:
        Model ready for gradient computation
    """
    gradient_model = model

    if gradient_checkpoint_path is not None:
        print(f"\n=== Loading checkpoint for gradient computation ===", flush=True)
        print(f"  Checkpoint path: {gradient_checkpoint_path}", flush=True)
        print(f"  Clone model: {clone_model_for_gradients}", flush=True)

        if clone_model_for_gradients:
            print("  Cloning model...", flush=True)
            gradient_model = copy.deepcopy(model)
            print("  Model cloned successfully", flush=True)

        # Load checkpoint with safe loading (consistent with main_forget.py)
        load_kwargs = {"map_location": device}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False

        checkpoint = torch.load(gradient_checkpoint_path, **load_kwargs)

        # Handle different checkpoint formats (consistent with main_forget.py)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint.keys():
            state_dict = checkpoint["state_dict"]
            print("  Using 'state_dict' key from checkpoint", flush=True)
        else:
            state_dict = checkpoint
            print("  Using checkpoint directly as state_dict", flush=True)

        # Load weights with strict=False to allow partial loading
        gradient_model.load_state_dict(state_dict, strict=False)
        gradient_model.to(device)
        print(f"  Checkpoint loaded successfully", flush=True)

        # Log additional checkpoint information if available
        if isinstance(checkpoint, dict):
            if "epoch" in checkpoint:
                print(f"  Checkpoint epoch: {checkpoint['epoch']}", flush=True)
            if "accuracy" in checkpoint:
                print(f"  Checkpoint accuracy: {checkpoint['accuracy']}", flush=True)
            if "acc" in checkpoint:
                print(f"  Checkpoint accuracy: {checkpoint['acc']}", flush=True)

    return gradient_model


def _compute_batch_gradients(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
) -> dict[str, torch.Tensor]:
    """
    Compute gradients for a single batch.

    Args:
        model: Neural network model
        data: Input data
        target: Target labels
        criterion: Loss function

    Returns:
        Dictionary of gradients per layer
    """
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    gradients_dict = {}
    for name, param in model.named_parameters():
        if name.endswith("bias"):
            continue
        if not param.requires_grad or param.grad is None:
            continue
        prefix = name.rsplit(".", 1)[0] if "." in name else ""
        grad_value = param.grad.detach().clone()

        if prefix in gradients_dict:
            gradients_dict[prefix] += grad_value
        else:
            gradients_dict[prefix] = grad_value

    return gradients_dict


def apply_dash_warm_start(*args, **kwargs):
    """
    Backward-compatible wrapper for DASH warm-start.

    Keeps monkeypatchability via gradient_utils.compute_dash_gradients in tests and callers.
    """
    return _apply_dash_warm_start_impl(
        *args,
        compute_gradients_fn=compute_dash_gradients,
        **kwargs,
    )


def project_gradients_onto_forget(
    retain_gradients: dict[str, torch.Tensor],
    forget_gradients: dict[str, torch.Tensor],
    eps: float = 1e-12,
    return_stats: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, float]]:
    """
    Project retain gradients onto forget gradients (per-parameter tensor).

    If forget gradient is missing or near-zero, the parameter is skipped to
    avoid shrinking in non-forget directions.
    """
    projected: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_zero = 0
    for name, g_r in retain_gradients.items():
        g_f = forget_gradients.get(name)
        if g_f is None:
            skipped_missing += 1
            continue
        g_f_flat = g_f.flatten()
        denom = torch.dot(g_f_flat, g_f_flat)
        if denom <= eps:
            skipped_zero += 1
            continue
        coeff = torch.dot(g_r.flatten(), g_f_flat) / (denom + eps)
        projected[name] = coeff * g_f

    skipped = skipped_missing + skipped_zero
    print(
        f"✓ Projected retain gradients onto forget gradients: " f"{len(projected)} tensors, skipped {skipped}",
        flush=True,
    )
    stats = {
        "projected_tensors": float(len(projected)),
        "skipped_tensors": float(skipped),
        "skipped_missing": float(skipped_missing),
        "skipped_zero": float(skipped_zero),
    }
    if return_stats:
        return projected, stats
    return projected


def project_gradients_perpendicular(
    model: nn.Module,
    gradients: dict[str, torch.Tensor],
    per_channel_projection: bool = False,
    channelwise_info: dict[str, bool] = None,
) -> dict[str, torch.Tensor]:
    """
    Project gradients onto the space perpendicular to the weights.

    Args:
        model: The neural network model
        gradients: Dictionary of gradients to project
        per_channel_projection: If True, use per-channel projection for Conv2d layers with channelwise SVD
        channelwise_info: Dictionary mapping layer names to whether they use channelwise SVD

    Returns:
        Dictionary of projected gradients
    """
    if channelwise_info is None:
        channelwise_info = {}

    projected_gradients = {}
    projection_info = []
    eps = torch.finfo(torch.float32).eps

    mode_str = "per-channel" if per_channel_projection else "global"
    print(f"Using {mode_str} projection of gradients onto the space perpendicular to the weights.", flush=True)

    for layer_name, grad in gradients.items():
        weight_tensor = get_module_by_name(model, layer_name).weight
        is_channelwise = channelwise_info.get(layer_name, False)

        # Decide whether to use per-channel projection
        use_per_channel = per_channel_projection and is_channelwise and grad.dim() == 4 and weight_tensor.dim() == 4

        if use_per_channel:
            proj_grad, info = _project_grad_perp_per_channel(
                grad=grad, weight_tensor=weight_tensor, layer_name=layer_name, eps=eps
            )
        else:
            proj_grad, info = _project_grad_perp_global(
                grad=grad, weight_tensor=weight_tensor, layer_name=layer_name, eps=eps
            )

        projected_gradients[layer_name] = proj_grad
        if info is not None:
            projection_info.append(info)

    # Print projection information
    if projection_info:
        print("\nProjection coefficients (numerator, denominator) for layers:", flush=True)
        for info in projection_info:
            if info["mode"] == "global":
                print(
                    f"- {info['name']} [Global]: {info['num']:.4e} / {info['denom']:.4e} = {info['coeff']:.4e}",
                    flush=True,
                )
            else:
                print(
                    f"- {info['name']} [PerCh]: mean={info['coeff_mean']:.4e} ± {info['coeff_std']:.4e} (across {info['num_channels']} channels)",
                    flush=True,
                )

    return projected_gradients


def project_forget_perpendicular_to_retain(
    forget_gradients: dict[str, torch.Tensor],
    retain_gradients: dict[str, torch.Tensor],
    model: nn.Module = None,
    project_retain_perpendicular_weights: bool = False,
    per_channel_projection: bool = False,
    channelwise_info: dict[str, bool] = None,
    eps: float = 1e-10,
) -> dict[str, torch.Tensor]:
    """
    Project forget gradients onto space perpendicular to retain gradients.

    For each layer:
        If project_retain_perpendicular_weights:
            G_r_ortho = G_r - (G_r · W / ||W||²) * W  [make G_r ⊥ W first]
        else:
            G_r_ortho = G_r

        G_f_proj = G_f - (G_f · G_r_ortho / ||G_r_ortho||²) * G_r_ortho

    Projection modes:
    - Global (per_channel_projection=False): Single projection for entire layer
    - Per-channel (per_channel_projection=True): Separate projection for each input channel (Conv2d only)

    This ensures:
    - Without flag: G_f_proj ⊥ G_r
    - With flag (Gram-Schmidt): G_f_proj ⊥ {G_r, W}

    Args:
        forget_gradients: Gradients computed on forget set
        retain_gradients: Gradients computed on retain set
        model: Model (required if project_retain_perpendicular_weights=True or per_channel_projection=True)
        project_retain_perpendicular_weights: First make retain gradient ⊥ weights (Gram-Schmidt)
        per_channel_projection: Use per-channel projection for Conv2d with channelwise SVD
        channelwise_info: Dict indicating which layers used channelwise SVD
        eps: Small value to prevent division by zero

    Returns:
        Projected forget gradients (perpendicular to retain [and optionally weights])
    """
    # Validation
    if project_retain_perpendicular_weights and model is None:
        raise ValueError("model must be provided when project_retain_perpendicular_weights=True")

    if per_channel_projection and model is None:
        raise ValueError("model must be provided when per_channel_projection=True")

    projected = {}
    projection_stats = []

    for layer_name, G_f in forget_gradients.items():
        if layer_name not in retain_gradients:
            print(f"⚠️  No retain gradient for {layer_name}, skipping projection", flush=True)
            projected[layer_name] = G_f
            continue

        G_r = retain_gradients[layer_name]

        # Determine if this layer should use per-channel projection
        is_channelwise = channelwise_info.get(layer_name, False) if channelwise_info else False
        use_per_channel = per_channel_projection and is_channelwise and G_f.dim() == 4

        if use_per_channel:
            # PER-CHANNEL PROJECTION for Conv2d with channelwise SVD
            G_f_proj, stat = _project_per_channel(
                G_f, G_r, layer_name, model, project_retain_perpendicular_weights, eps
            )
            projected[layer_name] = G_f_proj
            projection_stats.append(stat)
        else:
            # GLOBAL PROJECTION (existing code)
            G_f_proj, stat = _project_global(G_f, G_r, layer_name, model, project_retain_perpendicular_weights, eps)
            projected[layer_name] = G_f_proj
            projection_stats.append(stat)

    # Print statistics
    print("\n" + "=" * 100, flush=True)
    if per_channel_projection and any(
        channelwise_info.get(s["layer"], False) for s in projection_stats if channelwise_info
    ):
        print("FORGET GRADIENT PROJECTION STATISTICS (PER-CHANNEL MODE)", flush=True)
        if project_retain_perpendicular_weights:
            print("Using Gram-Schmidt: G_f ⊥ {G_r, W} per input channel", flush=True)
        else:
            print("Projecting G_f ⊥ G_r per input channel", flush=True)
    elif project_retain_perpendicular_weights:
        print("FORGET GRADIENT PROJECTION STATISTICS (G_f ⊥ {G_r, W}) - GRAM-SCHMIDT", flush=True)
        print("Step 1: Project G_r ⊥ W", flush=True)
        print("Step 2: Project G_f ⊥ G_r_ortho", flush=True)
    else:
        print("FORGET GRADIENT PROJECTION STATISTICS (G_f ⊥ G_r)", flush=True)
    print("=" * 100, flush=True)

    if project_retain_perpendicular_weights:
        print(
            f"{'Layer':<30} {'Mode':<8} {'Cos(G_r,W)':<12} {'||G_r||':<12} {'||G_r_⊥||':<12} {'Cos(G_f,G_r_⊥)':<15} {'||G_f||':<12} {'||G_f_⊥||':<12}",
            flush=True,
        )
        print("-" * 110, flush=True)
        for stat in projection_stats:
            mode = "PerCh" if stat.get("per_channel", False) else "Global"
            cos_gr_w = stat["cosine_G_r_W"]
            cos_gr_w_str = f"{cos_gr_w:.4f}"
            if stat.get("per_channel", False) and "cosine_G_r_W_std" in stat:
                cos_gr_w_str += f"±{stat['cosine_G_r_W_std']:.3f}"

            cos_sim = stat["cosine_similarity"]
            cos_sim_str = f"{cos_sim:.4f}"
            if stat.get("per_channel", False) and "cosine_similarity_std" in stat:
                cos_sim_str += f"±{stat['cosine_similarity_std']:.3f}"

            print(
                f"{stat['layer']:<30} "
                f"{mode:<8} "
                f"{cos_gr_w_str:>11} "
                f"{stat['norm_retain_original']:>11.4e} "
                f"{stat['norm_retain_ortho']:>11.4e} "
                f"{cos_sim_str:>14} "
                f"{stat['norm_before']:>11.4e} "
                f"{stat['norm_after']:>11.4e}",
                flush=True,
            )
    else:
        print(
            f"{'Layer':<35} {'Mode':<8} {'Cos Sim':<12} {'Proj Coef':<12} {'||G_f||':<12} {'||G_f_⊥||':<12} {'||G_r||':<12}",
            flush=True,
        )
        print("-" * 105, flush=True)
        for stat in projection_stats:
            mode = "PerCh" if stat.get("per_channel", False) else "Global"
            cos_sim_str = f"{stat['cosine_similarity']:.4f}"
            if stat.get("per_channel", False) and "cosine_similarity_std" in stat:
                cos_sim_str += f"±{stat['cosine_similarity_std']:.3f}"

            print(
                f"{stat['layer']:<35} "
                f"{mode:<8} "
                f"{cos_sim_str:>11} "
                f"{stat['projection_coeff']:>11.4e} "
                f"{stat['norm_before']:>11.4e} "
                f"{stat['norm_after']:>11.4e} "
                f"{stat['norm_retain_ortho']:>11.4e}",
                flush=True,
            )

    # Summary statistics
    avg_cosine = sum(s["cosine_similarity"] for s in projection_stats) / len(projection_stats)
    avg_norm_ratio = sum(s["norm_after"] / s["norm_before"] for s in projection_stats) / len(projection_stats)
    print("-" * 110, flush=True)
    print(f"Average cosine similarity G_f vs G_r_ortho (before projection): {avg_cosine:.4f}", flush=True)
    print(f"Average norm ratio (after/before): {avg_norm_ratio:.4f}", flush=True)

    if project_retain_perpendicular_weights:
        avg_cosine_G_r_W = sum(s["cosine_G_r_W"] for s in projection_stats) / len(projection_stats)
        avg_retain_norm_ratio = sum(s["norm_retain_ortho"] / s["norm_retain_original"] for s in projection_stats) / len(
            projection_stats
        )
        print(f"Average cosine similarity G_r vs W: {avg_cosine_G_r_W:.4f}", flush=True)
        print(f"Average retain norm ratio (ortho/original): {avg_retain_norm_ratio:.4f}", flush=True)

    # Per-channel specific stats
    per_ch_stats = [s for s in projection_stats if s.get("per_channel", False)]
    if per_ch_stats:
        print(f"Per-channel layers: {len(per_ch_stats)}/{len(projection_stats)}", flush=True)
        avg_channels = sum(s.get("num_channels", 0) for s in per_ch_stats) / len(per_ch_stats)
        print(f"Average channels per layer: {avg_channels:.1f}", flush=True)

    print(f"Total layers projected: {len(projection_stats)}", flush=True)
    print("=" * 110 + "\n", flush=True)

    return projected


def extract_weights_from_model(
    model: nn.Module,
    changed_layers_class: list[str],
) -> dict[str, torch.Tensor]:
    """
    Extract weights from model layers for SVD.

    Args:
        model: Neural network model
        changed_layers_class: List of layer class names to extract weights from

    Returns:
        Dictionary mapping layer names to their weights
    """
    weights_dict = {}

    for layer_name, module in model.named_modules():
        if layer_name == "":
            continue

        # Check if layer type matches
        module_type = type(module).__name__.lower()
        if module_type not in [cls.lower() for cls in changed_layers_class]:
            continue

        # Extract weight
        if hasattr(module, "weight") and module.weight is not None:
            weights_dict[layer_name] = module.weight.detach().clone()
            print(f"  Extracted weights from {layer_name}: shape={tuple(module.weight.shape)}", flush=True)

    print(f"  Total layers extracted: {len(weights_dict)}", flush=True)
    return weights_dict


def _project_global(
    G_f: torch.Tensor,
    G_r: torch.Tensor,
    layer_name: str,
    model: nn.Module,
    project_retain_perpendicular_weights: bool,
    eps: float,
) -> tuple[torch.Tensor, dict]:
    """
    Global projection (flatten entire tensor and project once).

    Args:
        G_f: Forget gradient
        G_r: Retain gradient
        layer_name: Name of the layer
        model: Model (for weight access if needed)
        project_retain_perpendicular_weights: Use Gram-Schmidt
        eps: Small value to prevent division by zero

    Returns:
        Tuple of (projected_gradient, statistics_dict)
    """
    # Flatten for dot product
    G_f_flat = G_f.flatten()
    G_r_flat = G_r.flatten()

    # Step 1: Optionally make G_r orthogonal to weights (Gram-Schmidt)
    if project_retain_perpendicular_weights:
        W = get_module_by_name(model, layer_name).weight
        W_flat = W.flatten()

        # Compute G_r projection onto W
        dot_G_r_W = torch.dot(G_r_flat, W_flat)
        W_norm_sq = torch.dot(W_flat, W_flat) + eps
        proj_coeff_W = dot_G_r_W / W_norm_sq

        # Make G_r orthogonal to W
        G_r_ortho_flat = G_r_flat - proj_coeff_W * W_flat
        G_r_ortho = G_r_ortho_flat.reshape(G_r.shape)

        # Statistics for Gram-Schmidt
        norm_G_r_original = torch.norm(G_r_flat)
        cosine_G_r_W = dot_G_r_W / (norm_G_r_original * torch.norm(W_flat) + eps)
    else:
        G_r_ortho_flat = G_r_flat
        G_r_ortho = G_r

    # Step 2: Project G_f perpendicular to G_r_ortho
    dot_product = torch.dot(G_f_flat, G_r_ortho_flat)
    G_r_ortho_norm_sq = torch.dot(G_r_ortho_flat, G_r_ortho_flat) + eps
    projection_coeff = dot_product / G_r_ortho_norm_sq

    # Project: G_f_proj = G_f - projection_coeff * G_r_ortho
    G_f_proj = G_f - projection_coeff * G_r_ortho

    # Statistics
    G_f_norm = torch.norm(G_f_flat)
    G_r_ortho_norm = torch.norm(G_r_ortho_flat)
    cosine_sim = dot_product / (G_f_norm * G_r_ortho_norm + eps)

    stat = {
        "layer": layer_name,
        "cosine_similarity": cosine_sim.item(),
        "projection_coeff": projection_coeff.item(),
        "norm_before": G_f_norm.item(),
        "norm_after": torch.norm(G_f_proj.flatten()).item(),
        "norm_retain_ortho": G_r_ortho_norm.item(),
        "per_channel": False,
    }

    # Add weight projection stats if enabled
    if project_retain_perpendicular_weights:
        stat["cosine_G_r_W"] = cosine_G_r_W.item()
        stat["proj_coeff_W"] = proj_coeff_W.item()
        stat["norm_retain_original"] = norm_G_r_original.item()

    return G_f_proj, stat


def _project_per_channel(
    G_f: torch.Tensor,
    G_r: torch.Tensor,
    layer_name: str,
    model: nn.Module,
    project_retain_perpendicular_weights: bool,
    eps: float,
) -> tuple[torch.Tensor, dict]:
    """
    Per-channel projection for Conv2d layers with channelwise SVD.
    Projects separately for each input channel to respect channel structure.

    Args:
        G_f: Forget gradient (C_out, C_in, K_h, K_w)
        G_r: Retain gradient (C_out, C_in, K_h, K_w)
        layer_name: Name of the layer
        model: Model (for weight access if needed)
        project_retain_perpendicular_weights: Use Gram-Schmidt per channel
        eps: Small value to prevent division by zero

    Returns:
        Tuple of (projected_gradient, statistics_dict)
    """
    if G_f.dim() != 4:
        # Fallback to global for non-Conv2d
        return _project_global(G_f, G_r, layer_name, model, project_retain_perpendicular_weights, eps)

    C_out, C_in, K_h, K_w = G_f.shape
    G_f_proj = G_f.clone()

    # Get weights if needed
    W = get_module_by_name(model, layer_name).weight if project_retain_perpendicular_weights else None

    # Statistics accumulators
    cosine_sims = []
    proj_coeffs = []
    norms_before = []
    norms_after = []
    cosines_G_r_W = [] if project_retain_perpendicular_weights else None

    # Project per input channel
    for c_in in range(C_in):
        # Extract channel slices (C_out, K_h, K_w)
        G_f_c = G_f[:, c_in, :, :].flatten()  # (C_out * K_h * K_w,)
        G_r_c = G_r[:, c_in, :, :].flatten()

        # Step 1: Optionally make G_r_c orthogonal to W_c (Gram-Schmidt per channel)
        if project_retain_perpendicular_weights:
            W_c = W[:, c_in, :, :].flatten()

            dot_G_r_W = torch.dot(G_r_c, W_c)
            W_norm_sq = torch.dot(W_c, W_c) + eps
            proj_coeff_W = dot_G_r_W / W_norm_sq

            # Make G_r_c orthogonal to W_c
            G_r_c_ortho = G_r_c - proj_coeff_W * W_c

            # Statistics
            cosine_G_r_W = dot_G_r_W / (torch.norm(G_r_c) * torch.norm(W_c) + eps)
            cosines_G_r_W.append(cosine_G_r_W.item())
        else:
            G_r_c_ortho = G_r_c

        # Step 2: Project G_f_c perpendicular to G_r_c_ortho
        dot_prod = torch.dot(G_f_c, G_r_c_ortho)
        G_r_c_norm_sq = torch.dot(G_r_c_ortho, G_r_c_ortho) + eps
        proj_coeff = dot_prod / G_r_c_norm_sq

        # Compute projection
        G_f_c_proj = G_f_c - proj_coeff * G_r_c_ortho

        # Update projected gradient (reshape back)
        G_f_proj[:, c_in, :, :] = G_f_c_proj.reshape(C_out, K_h, K_w)

        # Collect statistics
        G_f_norm = torch.norm(G_f_c)
        cosine_sim = dot_prod / (G_f_norm * torch.norm(G_r_c_ortho) + eps)

        cosine_sims.append(cosine_sim.item())
        proj_coeffs.append(proj_coeff.item())
        norms_before.append(G_f_norm.item())
        norms_after.append(torch.norm(G_f_c_proj).item())

    # Aggregate statistics across channels
    import numpy as np

    stat = {
        "layer": layer_name,
        "cosine_similarity": np.mean(cosine_sims),
        "cosine_similarity_std": np.std(cosine_sims),
        "projection_coeff": np.mean(proj_coeffs),
        "norm_before": np.mean(norms_before),
        "norm_after": np.mean(norms_after),
        "norm_retain_ortho": torch.norm(G_r).item(),  # Overall norm
        "per_channel": True,
        "num_channels": C_in,
    }

    if project_retain_perpendicular_weights:
        stat["cosine_G_r_W"] = np.mean(cosines_G_r_W)
        stat["cosine_G_r_W_std"] = np.std(cosines_G_r_W)
        stat["norm_retain_original"] = torch.norm(G_r).item()

    return G_f_proj, stat


def _project_grad_perp_global(
    grad: torch.Tensor,
    weight_tensor: torch.Tensor,
    layer_name: str,
    eps: float,
) -> tuple[torch.Tensor, dict]:
    """
    Project gradient perpendicular to weight using global projection.

    Args:
        grad: Gradient tensor to project
        weight_tensor: Weight tensor
        layer_name: Name of the layer
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (projected_gradient, info_dict)
    """
    weight_norm = LA.norm(weight_tensor)

    if weight_norm <= eps:
        return grad, None

    # Compute projection coefficient
    projection_coeff_num = torch.dot(grad.flatten(), weight_tensor.flatten())
    weight_norm_sq = weight_norm.pow(2)
    projection_coeff = projection_coeff_num / (weight_norm_sq + eps)

    # Project gradient
    proj_grad = grad - projection_coeff * weight_tensor

    info = {
        "mode": "global",
        "name": layer_name,
        "num": projection_coeff_num.item(),
        "denom": weight_norm_sq.item(),
        "coeff": projection_coeff.item(),
    }

    return proj_grad, info


def _project_grad_perp_per_channel(
    grad: torch.Tensor,
    weight_tensor: torch.Tensor,
    layer_name: str,
    eps: float,
) -> tuple[torch.Tensor, dict]:
    """
    Project gradient perpendicular to weight using per-channel projection.

    Args:
        grad: Gradient tensor (C_out, C_in, K_h, K_w)
        weight_tensor: Weight tensor (C_out, C_in, K_h, K_w)
        layer_name: Name of the layer
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (projected_gradient, info_dict)
    """
    C_out, C_in, K_h, K_w = grad.shape
    proj_grad = torch.zeros_like(grad)

    projection_coeffs = []
    projection_nums = []
    projection_denoms = []

    for c_in in range(C_in):
        # Extract per-channel slices and flatten
        grad_c = grad[:, c_in, :, :].flatten()
        weight_c = weight_tensor[:, c_in, :, :].flatten()

        weight_c_norm = torch.norm(weight_c)

        if weight_c_norm <= eps:
            proj_grad[:, c_in, :, :] = grad[:, c_in, :, :]
            continue

        # Compute projection coefficient
        proj_coeff_num = torch.dot(grad_c, weight_c)
        weight_c_norm_sq = weight_c_norm.pow(2)
        proj_coeff = proj_coeff_num / (weight_c_norm_sq + eps)

        # Project gradient
        grad_c_proj = grad_c - proj_coeff * weight_c

        # Update projected gradient (reshape back)
        proj_grad[:, c_in, :, :] = grad_c_proj.reshape(C_out, K_h, K_w)

        # Collect statistics
        projection_coeffs.append(proj_coeff.item())
        projection_nums.append(proj_coeff_num.item())
        projection_denoms.append(weight_c_norm_sq.item())

    # Aggregate statistics across channels
    import numpy as np

    info = {
        "mode": "per_channel",
        "name": layer_name,
        "coeff_mean": np.mean(projection_coeffs),
        "coeff_std": np.std(projection_coeffs),
        "num_mean": np.mean(projection_nums),
        "denom_mean": np.mean(projection_denoms),
        "num_channels": C_in,
    }

    return proj_grad, info
