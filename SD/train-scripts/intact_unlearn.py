"""
InTAct Unlearning for Stable Diffusion

This script implements InTAct (Interval-based Task Activation Consolidation) unlearning for Stable Diffusion,
composable with multiple base unlearning methods:
- GA (Gradient Ascent)
- RL (Random Label)  
- NSFW (NSFW concept removal)
- ESD (Erased Stable Diffusion)

InTAct adds interval protection loss on top of any base method:
    total_loss = base_loss + lambda_interval * intact_loss

Usage:
    python train-scripts/intact_unlearn.py --base_method ga --class_to_forget 0 --targets to_q to_k to_v
    python train-scripts/intact_unlearn.py --base_method rl --class_to_forget 0 --targets to_q to_k to_v
    python train-scripts/intact_unlearn.py --base_method nsfw --targets attn1
    python train-scripts/intact_unlearn.py --base_method esd --prompt "nudity" --targets to_q to_k to_v
"""

import argparse
import csv
import logging
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # For InTAct
sys.path.insert(0, str(Path(__file__).parent.parent))  # For ldm and SD modules

from InTAct.intact import UnlearnIntervalProtection
from convertModels import savemodelDiffusers
from dataset import (
    setup_class_forgetting_data,
    setup_forget_nsfw_data,
    setup_model,
)
from dash_sd_runtime import run_dash_sd_warm_start
from ldm.models.diffusion.ddim import DDIMSampler
from random_label import (
    _collect_unet_grad_norms,
    _loader_data_stats,
    _loss_mean_payload,
    _normalize_rl_loss_mode,
    _should_log_grad_norm,
    _update_loss_accumulator,
)
from run_naming import build_sd_unlearn_name
from training_eval import (
    compute_named_parameter_change_stats,
    compute_unet_change_stats,
    run_training_eval,
    should_run_pre_epoch_train_eval,
    should_run_train_eval,
    snapshot_named_parameter_baseline,
    write_metric_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def _set_seed(seed):
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    log.info("Seed set to %d", int(seed))


def _log_wandb_scalars(payload, step=None):
    try:
        import wandb
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return
    payload = {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }
    if payload:
        wandb.log(payload, step=step)


def _next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _next_or_none(iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        return None, iterator


def _unlearn_epoch_batch_count(retain_dl, forget_dl, full_retain_per_epoch=False):
    if full_retain_per_epoch:
        return max(len(retain_dl), len(forget_dl))
    return len(forget_dl)


def _retain_label_pool(forget_indices, descriptions, device=None):
    forget_set = {int(idx) for idx in forget_indices}
    retain_labels = [idx for idx in range(len(descriptions)) if idx not in forget_set]
    if not retain_labels:
        raise ValueError("Cannot sample pseudo labels: every description is marked as forgotten.")
    return torch.tensor(retain_labels, dtype=torch.long, device=device)


def _sample_pseudo_prompts(batch_size, forget_indices, descriptions, device=None):
    retain_labels = _retain_label_pool(forget_indices, descriptions, device=device)
    sample_indices = torch.randint(
        0,
        int(retain_labels.numel()),
        (int(batch_size),),
        device=retain_labels.device,
    )
    return [descriptions[int(label)] for label in retain_labels[sample_indices].tolist()]


# ============================================================================
# Config Loading
# ============================================================================

def load_training_config(config_path):
    """Load training configuration from YAML file."""
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        if hasattr(config, 'training'):
            log.info(f"Loaded training config from {config_path}")
            return config.training
    return None


# ============================================================================
# SD Forward Function for InTAct (model-agnostic activation collection)
# ============================================================================

def sd_forward_fn(model, batch, device, prompts=None, data_transform_fn=None, betas=None, num_timesteps=1000):
    """
    SD-specific forward function for InTAct activation collection.
    Takes raw image batches and handles full encoding/forward pipeline.
    
    Args:
        model: Full SD model (LatentDiffusion) - needed for get_input()
        batch: Either tuple (images, labels) or just images (for NSFW datasets)
        device: CUDA device
        prompts: List of text prompts (indexed by labels if available)
        betas: Noise schedule betas tensor
        num_timesteps: Number of diffusion timesteps
    """
    # Handle both (images, labels) and images-only batches
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], torch.Tensor):
        # batch is (images, labels) tuple from DataLoader
        images, labels = batch
    else:
        # batch is just images (NSFW datasets)
        images = batch
        labels = None
    
    images = torch.stack([item for item in images])
    images = images.to(device)
    n = images.size(0)
    
    # Get text prompts
    if prompts is not None and labels is not None:
        txt = [prompts[label] for label in labels]
    elif prompts is not None:
        # No labels (e.g. NSFW datasets) — repeat first prompt for all images
        txt = [prompts[0]] * n
    else:
        txt = [""] * n
    
    # Create batch dict for SD
    batch_dict = {
        "jpg": images.permute(0, 2, 3, 1),
        "txt": txt
    }
    
    # Encode to latent and get conditioning embeddings
    with torch.no_grad():
        x, c = model.get_input(batch_dict, model.first_stage_key)
    
    if data_transform_fn is not None:
        x = data_transform_fn(x)
    
    # Create timesteps
    t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
    t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
    
    # Add noise if betas provided
    if betas is not None:
        e = torch.randn_like(x)
        a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x_noisy = x * a.sqrt() + e * (1.0 - a).sqrt()
    else:
        x_noisy = x
    
    # Forward through UNet (triggers hooks for activation collection)
    model.model.diffusion_model(x_noisy, t.float(), context=c)


# ============================================================================
# Model Loading (from existing scripts)
# ============================================================================

def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt (from train-esd.py)"""
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf
    
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    # SD v1 checkpoints are trusted local Lightning pickles; PyTorch >=2.6
    # defaults to weights_only=True, which rejects their callback metadata.
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if m:
        log.warning("Missing keys while loading SD checkpoint %s: %s", ckpt, m)
    if u:
        log.warning("Unexpected keys while loading SD checkpoint %s: %s", ckpt, u)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


def get_models_esd(config_path, ckpt_path, devices):
    """Load original and training models for ESD (from train-esd.py)"""
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler


@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta,
                 start_code=None, n_samples=1, t_start=-1, log_every_t=None,
                 till_T=None, verbose=True):
    """Sample the model (from train-esd.py)"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(
        S=ddim_steps,
        conditioning=c,
        batch_size=n_samples,
        shape=shape,
        verbose=False,
        x_T=start_code,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
        verbose_iter=verbose,
        t_start=t_start,
        log_every_t=log_t,
        till_T=till_T,
    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim


# ============================================================================
# InTAct Setup for SD
# ============================================================================

def setup_intact_protection(
    model,
    forget_dl,
    remain_dl,
    descriptions,
    device,
    targets,
    lambda_interval=1.0,
    lower_percentile=0.05,
    upper_percentile=0.95,
    reduced_dim=32,
    infinity_scale=20.0,
    use_actual_bounds=False,
    normalize_protection=True,
):
    """
    Setup InTAct protection for SD model.
    
    Args:
        model: SD model (LatentDiffusion)
        forget_dl: Forget dataloader (raw, yields images/labels)
        remain_dl: Remain dataloader (optional)
        descriptions: List of class descriptions (prompts indexed by label)
        device: CUDA device
        targets: List of target layer patterns (e.g., ["to_q", "to_k", "to_v"])
    
    Returns:
        protection: UnlearnIntervalProtection instance
    """
    log.info(f"Setting up InTAct protection with targets: {targets}")
    
    # Create protection instance
    protection = UnlearnIntervalProtection(
        targets=targets,
        lambda_interval=lambda_interval,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        reduced_dim=reduced_dim,
        infinity_scale=infinity_scale,
        use_actual_bounds=use_actual_bounds,
        normalize_protection=normalize_protection,
    )
    
    # Create forward function with prompts pre-bound
    # Note: Capture full model for encoding, but forward_fn receives diffusion_model
    def forward_fn(diffusion_model, batch, dev, **kwargs):
        return sd_forward_fn(model, batch, dev, prompts=descriptions, **kwargs)
    
    # Setup protection on diffusion_model, but pass raw dataloaders
    protection.setup_protection(
        model.model.diffusion_model,
        forget_dl,
        device,
        remain_dataloader=remain_dl,
        forward_fn=forward_fn,
        betas=model.betas.to(device) if hasattr(model, 'betas') else None,
        num_timesteps=model.num_timesteps if hasattr(model, 'num_timesteps') else 1000,
    )
    
    return protection


# ============================================================================
# Base Method Loss Functions
# ============================================================================

def compute_ga_loss(model, forget_batch, remain_batch, alpha, device):
    """
    Gradient Ascent loss (from gradient_ascent.py)
    Loss = alpha * (-forget_loss) + remain_loss
    """
    loss_terms = []
    forget_loss = None
    remain_loss = None

    if forget_batch is not None:
        # Forget loss is negative so gradient descent performs ascent.
        forget_loss = -model.shared_step(forget_batch)[0]
        loss_terms.append(alpha * forget_loss)

    if remain_batch is not None:
        remain_loss = model.shared_step(remain_batch)[0]
        loss_terms.append(remain_loss)

    if not loss_terms:
        raise ValueError("Internal error: both forget and retain batches are empty.")

    return sum(loss_terms) / len(loss_terms), forget_loss, remain_loss


def compute_rl_loss(model, forget_images, forget_prompts, pseudo_prompts,
                    remain_batch, alpha, criteria, device, rl_loss_mode="output_matching"):
    """
    Random Label loss (from random_label.py)
    Train forget images to predict pseudo/random label output instead of actual.
    """
    loss_terms = []
    forget_loss = None
    remain_loss = None

    if forget_images is not None:
        forget_batch = {
            "jpg": forget_images.permute(0, 2, 3, 1),
            "txt": forget_prompts,
        }
        pseudo_batch = {
            "jpg": forget_images.permute(0, 2, 3, 1),
            "txt": pseudo_prompts,
        }

        forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
        pseudo_input, pseudo_emb = model.get_input(pseudo_batch, model.first_stage_key)

        t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=device).long()
        noise = torch.randn_like(forget_input, device=device)

        forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
        if rl_loss_mode == "output_matching":
            pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
            forget_out = model.apply_model(forget_noisy, t, forget_emb)
            pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()
            forget_loss = criteria(forget_out, pseudo_out)
        elif rl_loss_mode == "denoise_pseudo":
            pseudo_out = model.apply_model(forget_noisy, t, pseudo_emb)
            forget_loss = criteria(pseudo_out, noise)
        else:
            raise ValueError(f"Unsupported rl_loss_mode={rl_loss_mode!r}")
        loss_terms.append(alpha * forget_loss)

    if remain_batch is not None:
        remain_loss = model.shared_step(remain_batch)[0]
        loss_terms.append(remain_loss)

    if not loss_terms:
        raise ValueError("Internal error: both forget and retain batches are empty.")

    return sum(loss_terms) / len(loss_terms), forget_loss, remain_loss


def compute_nsfw_loss(model, forget_images, remain_images, word_nude, word_wear,
                      alpha, criteria, device):
    """
    NSFW removal loss (from nsfw_removal.py)
    Similar to EL but with specific nude/wear prompts.
    """
    batch_size = forget_images.shape[0]
    
    forget_prompts = [word_nude] * batch_size
    pseudo_prompts = [word_wear] * batch_size
    remain_prompts = [word_wear] * batch_size
    
    # Remain stage
    remain_batch = {
        "jpg": remain_images.permute(0, 2, 3, 1),
        "txt": remain_prompts,
    }
    remain_loss = model.shared_step(remain_batch)[0]
    
    # Forget stage
    forget_batch = {
        "jpg": forget_images.permute(0, 2, 3, 1),
        "txt": forget_prompts,
    }
    pseudo_batch = {
        "jpg": forget_images.permute(0, 2, 3, 1),
        "txt": pseudo_prompts,
    }
    
    forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
    pseudo_input, pseudo_emb = model.get_input(pseudo_batch, model.first_stage_key)
    
    t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=device).long()
    noise = torch.randn_like(forget_input, device=device)
    
    forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
    pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
    
    forget_out = model.apply_model(forget_noisy, t, forget_emb)
    pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()
    
    forget_loss = criteria(forget_out, pseudo_out)
    
    return forget_loss + alpha * remain_loss, None, None


def compute_esd_loss(model, model_orig, sampler, word, emb_0, emb_p, emb_n,
                     t_enc, t_enc_ddpm, start_code, criteria, devices,
                     start_guidance, negative_guidance, image_size, ddim_steps, ddim_eta):
    """
    ESD loss (from train-esd.py)
    """
    quick_sample_till_t = lambda x, s, code, t: sample_model(
        model, sampler, x, image_size, image_size, ddim_steps, s, ddim_eta,
        start_code=code, till_T=t, verbose=False
    )
    
    with torch.no_grad():
        # Generate image with concept from ESD model
        z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, int(t_enc))
        # Get scores from frozen model
        e_0 = model_orig.apply_model(
            z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1])
        )
        e_p = model_orig.apply_model(
            z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_p.to(devices[1])
        )
    
    # Get conditional score from ESD model
    e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_n.to(devices[0]))
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    # ESD objective
    loss = criteria(
        e_n.to(devices[0]),
        e_0.to(devices[0]) - (negative_guidance * (e_p.to(devices[0]) - e_0.to(devices[0])))
    )
    
    return loss, None, None


# ============================================================================
# Main Training Functions
# ============================================================================

def intact_unlearn_class(
    class_to_forget,
    base_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    # InTAct parameters
    targets,
    lambda_interval=1.0,
    lower_percentile=0.05,
    upper_percentile=0.95,
    reduced_dim=32,
    infinity_scale=20.0,
    use_actual_bounds=False,
    normalize_protection=True,
    # SD parameters
    image_size=512,
    ddim_steps=50,
    # Save paths
    model_save_dir="models",
    logs_dir="models",
    dash_config=None,
    seed=None,
    forget_classes=None,
    forget_concepts=None,
    full_retain_per_epoch=False,
    train_eval_config=None,
    wandb_log_interval=1,
    rl_loss_mode="output_matching",
    log_grad_norms=True,
    grad_norm_log_interval=3,
):
    """
    InTAct unlearning for class forgetting (GA/RL methods).
    """
    _set_seed(seed)
    rl_loss_mode = _normalize_rl_loss_mode(rl_loss_mode)
    wandb_log_interval = int(wandb_log_interval or 0)
    grad_norm_log_interval = int(grad_norm_log_interval or 0)
    log.info(f"InTAct Unlearning: base_method={base_method}, class={class_to_forget}, targets={targets}")
    
    # Setup model
    model = setup_model(config_path, ckpt_path, device)
    
    # Ensure all model buffers (including logvar) are on the correct device
    model = model.to(device)
    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(device)
    
    criteria = torch.nn.MSELoss()
    
    # Setup data
    remain_dl, forget_dl, descriptions, forget_indices = setup_class_forgetting_data(
        class_to_forget=class_to_forget,
        batch_size=batch_size,
        image_size=image_size,
        forget_classes=forget_classes,
        forget_concepts=forget_concepts,
    )
    primary_forget_class = int(forget_indices[0])
    forget_name = "_".join(str(idx) for idx in forget_indices)
    data_stats = _loader_data_stats(
        remain_dl,
        forget_dl,
        dash_config=dash_config,
        full_retain_per_epoch=full_retain_per_epoch,
        use_forget_in_unlearn=True,
    )
    uc_for_name = {
        "method": "intact",
        "class_to_forget": class_to_forget,
        "forget_classes": forget_classes,
        "forget_concepts": forget_concepts,
        "alpha": alpha,
        "epochs": epochs,
        "lr": lr,
        "full_retain_per_epoch": full_retain_per_epoch,
    }
    ic_for_name = {
        "base_method": base_method,
        "targets": targets,
        "lambda_interval": lambda_interval,
    }
    name = build_sd_unlearn_name(
        setting="sd",
        uc=uc_for_name,
        ic=ic_for_name,
        dash_cfg=dash_config,
        seed=seed,
    )
    folder_path = f"{logs_dir}/{name}"
    os.makedirs(folder_path, exist_ok=True)

    if should_run_pre_epoch_train_eval(train_eval_config):
        run_training_eval(
            model=model,
            name=name,
            epoch=-1,
            class_to_forget=primary_forget_class,
            config_path=config_path,
            ckpt_path=ckpt_path,
            diffusers_config_path=diffusers_config_path,
            device=device,
            image_size=image_size,
            logs_dir=logs_dir,
            eval_config=train_eval_config,
        )

    dash_start = time.perf_counter()
    dash_stats = run_dash_sd_warm_start(
        model=model,
        retain_loader=remain_dl,
        forget_loader=forget_dl,
        descriptions=descriptions,
        dash_config=dash_config,
        logger=log,
    )
    dash_total_seconds = time.perf_counter() - dash_start
    dash_stats["time____dash/total_seconds____warm_start"] = float(dash_total_seconds)
    _log_wandb_scalars({"time____dash/total_seconds____warm_start": float(dash_total_seconds)})
    dash_change_stats = compute_unet_change_stats(
        model,
        ckpt_path,
        dash_config=dash_config,
        prefix="after_dash_vs_base",
        target=(dash_config or {}).get("target", "unet"),
    )
    write_metric_dict(
        f"{folder_path}/dash_warm_start_stats.json",
        {
            "seed": seed if seed is not None else None,
            **data_stats,
            "rl_loss_mode": rl_loss_mode,
            **dash_stats,
            **dash_change_stats,
        },
    )
    
    # Setup InTAct protection (operates directly on diffusion_model)
    protection = setup_intact_protection(
        model, forget_dl, remain_dl, descriptions, device,
        targets=targets,
        lambda_interval=lambda_interval,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        reduced_dim=reduced_dim,
        infinity_scale=infinity_scale,
        use_actual_bounds=use_actual_bounds,
        normalize_protection=normalize_protection,
    )
    
    # Get reference to diffusion_model for InTAct operations
    diffusion_model = model.model.diffusion_model
    
    # Mark non-target parameters (doesn't freeze to avoid breaking checkpointing)
    protection.freeze_non_target_params(diffusion_model)
    
    # Get only trainable parameters for optimizer
    trainable_params = protection.get_trainable_params(diffusion_model)
    log.info(f"Training {len(trainable_params)} parameters")
    if not trainable_params:
        raise ValueError(
            f"InTAct selected no trainable parameters for targets={targets}. "
            "Check intact.targets against diffusion_model parameter names."
        )
    trainable_param_ids = {id(param) for param in trainable_params}
    trainable_named_params = [
        (param_name, param)
        for param_name, param in diffusion_model.named_parameters()
        if id(param) in trainable_param_ids
    ]
    post_dash_train_baseline = snapshot_named_parameter_baseline(trainable_named_params)
    
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    model.train()
    
    losses = []
    history_rows = []
    selected_param_ids = {id(param) for param in trainable_params}
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        epoch_loss_accumulator = {}
        forget_iter = iter(forget_dl)
        remain_iter = iter(remain_dl)
        epoch_batches = _unlearn_epoch_batch_count(
            remain_dl,
            forget_dl,
            full_retain_per_epoch=full_retain_per_epoch,
        )
        with tqdm(total=epoch_batches, desc=f"Epoch {epoch}") as pbar:
            for i in range(epoch_batches):
                optimizer.zero_grad()

                if full_retain_per_epoch:
                    forget_batch_data, forget_iter = _next_or_none(forget_iter)
                    remain_batch_data, remain_iter = _next_or_none(remain_iter)
                else:
                    forget_batch_data, forget_iter = _next_or_restart(forget_iter, forget_dl)
                    remain_batch_data, remain_iter = _next_or_restart(remain_iter, remain_dl)

                forget_images = None
                forget_prompts = None
                remain_batch = None

                if forget_batch_data is not None:
                    forget_images, forget_labels = forget_batch_data
                    forget_images = forget_images.to(device)
                    forget_prompts = [descriptions[label] for label in forget_labels]

                if remain_batch_data is not None:
                    remain_images, remain_labels = remain_batch_data
                    remain_images = remain_images.to(device)
                    remain_prompts = [descriptions[label] for label in remain_labels]
                    remain_batch = {
                        "jpg": remain_images.permute(0, 2, 3, 1),
                        "txt": remain_prompts,
                    }

                # Compute base method loss
                if base_method == "ga":
                    forget_batch = None
                    if forget_images is not None:
                        forget_batch = {
                            "jpg": forget_images.permute(0, 2, 3, 1),
                            "txt": forget_prompts,
                        }
                    base_loss, forget_loss_val, remain_loss_val = compute_ga_loss(
                        model, forget_batch, remain_batch, alpha, device
                    )
                elif base_method == "rl":
                    pseudo_prompts = None
                    if forget_images is not None:
                        pseudo_prompts = _sample_pseudo_prompts(
                            len(forget_labels),
                            forget_indices,
                            descriptions,
                            device=forget_labels.device,
                        )
                    base_loss, forget_loss_val, remain_loss_val = compute_rl_loss(
                        model, forget_images, forget_prompts, pseudo_prompts,
                        remain_batch, alpha, criteria, device, rl_loss_mode=rl_loss_mode,
                    )
                else:
                    raise ValueError(f"Unknown base_method for class unlearning: {base_method}")
                
                # Compute InTAct protection loss
                intact_loss = protection.compute_protection_loss(diffusion_model, device)
                
                # Total loss
                total_loss = base_loss + intact_loss
                total_loss.backward()

                base_loss_value = float(base_loss.detach().cpu())
                forget_loss_value = float(forget_loss_val.detach().cpu()) if forget_loss_val is not None else float("nan")
                weighted_forget_loss_value = (
                    float((alpha * forget_loss_val).detach().cpu()) if forget_loss_val is not None else float("nan")
                )
                remain_loss_value = float(remain_loss_val.detach().cpu()) if remain_loss_val is not None else float("nan")
                intact_loss_value = float(intact_loss.detach().cpu())
                total_loss_value = float(total_loss.detach().cpu())
                loss_values = {
                    "total": total_loss_value,
                    "base": base_loss_value,
                    "forget": forget_loss_value,
                    "weighted_forget": weighted_forget_loss_value,
                    "remain": remain_loss_value,
                    "intact": intact_loss_value,
                }
                _update_loss_accumulator(epoch_loss_accumulator, loss_values)
                grad_payload = {}
                if log_grad_norms and _should_log_grad_norm(i, epoch_batches, grad_norm_log_interval):
                    grad_payload = _collect_unet_grad_norms(model, selected_param_ids)
                
                optimizer.step()

                losses.append(total_loss_value)
                history_rows.append(
                    {
                        "step": len(history_rows),
                        "epoch": epoch,
                        "batch": i,
                        "total_loss": total_loss_value,
                        "base_loss": base_loss_value,
                        "forget_loss": forget_loss_value,
                        "weighted_forget_loss": weighted_forget_loss_value,
                        "remain_loss": remain_loss_value,
                        "intact_loss": intact_loss_value,
                        "loss_term_count": (1 if forget_loss_val is not None else 0) + (1 if remain_loss_val is not None else 0),
                        "has_forget_batch": 1 if forget_loss_val is not None else 0,
                        "has_retain_batch": 1 if remain_loss_val is not None else 0,
                    }
                )
                global_step = len(history_rows) - 1
                if wandb_log_interval > 0 and global_step % wandb_log_interval == 0:
                    history_row = history_rows[-1]
                    _log_wandb_scalars(
                        {
                            "loss____train/total____step": total_loss_value,
                            "loss____train/base____step": base_loss_value,
                            "loss____train/forget____step": forget_loss_value,
                            "loss____train/weighted_forget____step": weighted_forget_loss_value,
                            "loss____train/remain____step": remain_loss_value,
                            "loss____train/intact____step": intact_loss_value,
                            **_loss_mean_payload(epoch_loss_accumulator, "running"),
                            "meta____loss_term_count____train": float(history_row["loss_term_count"]),
                            "batch____has_forget____train": float(history_row["has_forget_batch"]),
                            "batch____has_retain____train": float(history_row["has_retain_batch"]),
                            "progress____epoch____train": float(epoch),
                            "progress____batch____train": float(i),
                            **grad_payload,
                        },
                        step=global_step,
                    )
                pbar.set_postfix({
                    "base": base_loss_value,
                    "intact": intact_loss_value,
                    "total": total_loss_value,
                })
                pbar.update(1)

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_payload = {
            "time____train/epoch_seconds____epoch": float(epoch_seconds),
            "progress____epoch____train": float(epoch),
        }
        if epoch_batches > 0:
            epoch_payload["time____train/step_seconds____epoch_mean"] = float(epoch_seconds) / float(epoch_batches)
        if epoch == 0 and epoch_seconds > 0:
            epoch_payload["time_pct____dash/total_vs_epoch____warm_start"] = (
                100.0 * float(dash_total_seconds) / float(epoch_seconds)
            )
        _log_wandb_scalars(
            {
                **epoch_payload,
                **_loss_mean_payload(epoch_loss_accumulator, "epoch"),
            },
            step=len(history_rows) - 1 if history_rows else None,
        )

        if should_run_train_eval(epoch, epochs, train_eval_config):
            eval_metrics = run_training_eval(
                model=model,
                name=name,
                epoch=epoch,
                class_to_forget=forget_indices[0],
                config_path=config_path,
                ckpt_path=ckpt_path,
                diffusers_config_path=diffusers_config_path,
                device=device,
                image_size=image_size,
                logs_dir=logs_dir,
                eval_config=train_eval_config,
            )
            if eval_metrics and epoch_seconds > 0:
                pct_payload = {}
                for key, pct_name in {
                    "time____eval_train/total_seconds____epoch": "time_pct____eval_train/total_vs_epoch____epoch",
                    "time____eval_train/generate_images_seconds____epoch": "time_pct____eval_train/generate_images_vs_epoch____epoch",
                    "time____eval_train/fid_seconds____epoch": "time_pct____eval_train/fid_vs_epoch____epoch",
                }.items():
                    value = eval_metrics.get(key)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        pct_payload[pct_name] = 100.0 * float(value) / float(epoch_seconds)
                if pct_payload:
                    pct_payload["progress____epoch____train_eval"] = float(epoch)
                    _log_wandb_scalars(pct_payload)
    
    model.eval()
    final_change_stats = compute_named_parameter_change_stats(
        trainable_named_params,
        ckpt_path=ckpt_path,
        prefix="after_unlearn_vs_base",
        selector="intact_trainable",
    )
    final_since_dash_stats = compute_named_parameter_change_stats(
        trainable_named_params,
        baseline=post_dash_train_baseline,
        prefix="after_unlearn_vs_after_dash",
        selector="intact_trainable",
    )
    write_metric_dict(
        f"{folder_path}/final_unlearn_delta_stats.json",
        {
            "seed": seed if seed is not None else None,
            **data_stats,
            "rl_loss_mode": rl_loss_mode,
            **final_change_stats,
            **final_since_dash_stats,
        },
    )
    save_model(model, name, None, config_path, diffusers_config_path, 
               model_save_dir=model_save_dir, device=device)
    save_history(losses, name, f"class_{forget_name}", history_rows=history_rows, logs_dir=logs_dir)
    
    return model


def intact_unlearn_nsfw(
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    # InTAct parameters
    targets,
    lambda_interval=1.0,
    lower_percentile=0.05,
    upper_percentile=0.95,
    reduced_dim=32,
    infinity_scale=20.0,
    use_actual_bounds=False,
    normalize_protection=True,
    # SD parameters
    image_size=512,
    ddim_steps=50,
    # Data paths
    nsfw_data_path="data/nsfw",
    not_nsfw_data_path="data/not-nsfw",
    # Save paths
    model_save_dir="models",
    logs_dir="models",
):
    """
    InTAct unlearning for NSFW concept removal.
    """
    log.info(f"InTAct NSFW Unlearning: targets={targets}")
    
    # Setup model
    model = setup_model(config_path, ckpt_path, device)
    
    # Ensure all model buffers (including logvar) are on the correct device
    model = model.to(device)
    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(device)
    
    sampler = DDIMSampler(model)
    criteria = torch.nn.MSELoss()
    
    # Setup data
    forget_dl, remain_dl = setup_forget_nsfw_data(batch_size, image_size, nsfw_data_path=nsfw_data_path, not_nsfw_data_path=not_nsfw_data_path)
    
    # NSFW prompts
    word_nude = "a photo of a nude person"
    word_wear = "a photo of a person wearing clothes"
    descriptions = [word_nude, word_wear]
    
    # Setup InTAct protection (operates directly on diffusion_model)
    protection = setup_intact_protection(
        model, forget_dl, remain_dl, descriptions, device,
        targets=targets,
        lambda_interval=lambda_interval,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        reduced_dim=reduced_dim,
        infinity_scale=infinity_scale,
        use_actual_bounds=use_actual_bounds,
        normalize_protection=normalize_protection,
    )
    
    # Get reference to diffusion_model for InTAct operations
    diffusion_model = model.model.diffusion_model
    
    # Mark non-target parameters (doesn't freeze to avoid breaking checkpointing)
    protection.freeze_non_target_params(diffusion_model)
    
    # Get only trainable parameters for optimizer
    trainable_params = protection.get_trainable_params(diffusion_model)
    log.info(f"Training {len(trainable_params)} parameters out of {sum(1 for _ in model.model.diffusion_model.parameters())} total")
    if not trainable_params:
        raise ValueError(
            f"InTAct selected no trainable parameters for targets={targets}. "
            "Check intact.targets against diffusion_model parameter names."
        )
    
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    model.train()
    
    losses = []
    history_rows = []
    targets_str = "_".join(targets)
    name = f"compvis-intact-nsfw-targets_{targets_str}-lambda_{lambda_interval}-lr_{lr}"
    
    # Training loop
    for epoch in range(epochs):
        forget_iter = iter(forget_dl)
        remain_iter = iter(remain_dl)
        with tqdm(total=len(forget_dl), desc=f"Epoch {epoch}") as pbar:
            for i in range(len(forget_dl)):
                optimizer.zero_grad()
                
                forget_images, forget_iter = _next_or_restart(forget_iter, forget_dl)
                remain_images, remain_iter = _next_or_restart(remain_iter, remain_dl)
                forget_images = forget_images.to(device)
                remain_images = remain_images.to(device)
                
                # Compute NSFW loss
                base_loss, forget_loss_val, remain_loss_val = compute_nsfw_loss(
                    model, forget_images, remain_images, word_nude, word_wear,
                    alpha, criteria, device
                )
                
                # Compute InTAct protection loss
                intact_loss = protection.compute_protection_loss(diffusion_model, device)
                
                # Total loss
                total_loss = base_loss + intact_loss
                total_loss.backward()
                
                optimizer.step()
                
                base_loss_value = float(base_loss.detach().cpu())
                intact_loss_value = float(intact_loss.detach().cpu())
                total_loss_value = float(total_loss.detach().cpu())
                losses.append(total_loss_value)
                history_rows.append(
                    {
                        "step": len(history_rows),
                        "epoch": epoch,
                        "batch": i,
                        "total_loss": total_loss_value,
                        "base_loss": base_loss_value,
                        "intact_loss": intact_loss_value,
                    }
                )
                pbar.set_postfix({
                    "base": base_loss_value,
                    "intact": intact_loss_value,
                    "total": total_loss_value,
                })
                pbar.update(1)
    
    model.eval()
    save_model(model, name, None, config_path, diffusers_config_path,
               model_save_dir=model_save_dir, device=device)
    save_history(losses, name, "nsfw", history_rows=history_rows, logs_dir=logs_dir)
    
    return model


def intact_unlearn_esd(
    prompt,
    start_guidance,
    negative_guidance,
    iterations,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    devices,
    # InTAct parameters
    targets,
    lambda_interval=1.0,
    lower_percentile=0.05,
    upper_percentile=0.95,
    reduced_dim=32,
    infinity_scale=20.0,
    use_actual_bounds=False,
    normalize_protection=True,
    # SD parameters
    seperator=None,
    image_size=512,
    ddim_steps=50,
):
    """
    InTAct unlearning with ESD base method.
    Note: ESD doesn't use dataloaders in the same way, so InTAct setup is different.
    We setup protection but skip activation collection (no forget dataloader).
    """
    log.info(f"InTAct ESD Unlearning: prompt={prompt}, targets={targets}")
    
    # Prompt processing (from train-esd.py)
    word_print = prompt.replace(" ", "")
    if prompt == "allartist":
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == "i2p":
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"
    
    if seperator is not None:
        words = [word.strip() for word in prompt.split(seperator)]
    else:
        words = [prompt]
    
    ddim_eta = 0
    
    # Load models
    model_orig, sampler_orig, model, sampler = get_models_esd(config_path, ckpt_path, devices)
    
    # For ESD, we generate synthetic latents directly (no real images needed)
    log.info("Generating synthetic forget data for InTAct setup...")
    
    # ESD forward function - works with pre-generated latents
    def esd_forward_fn(diffusion_model, batch, device, **kwargs):
        """Forward for ESD synthetic data (latents + embeddings)."""
        z, c = batch
        z = z.to(device)
        c = c.to(device)
        n = z.size(0)
        t = torch.randint(0, model.num_timesteps, (n,), device=device).long()
        diffusion_model(z, t.float(), context=c)
    
    # Simple generator for synthetic ESD data
    def generate_esd_batches(n_samples=50):
        for i in range(n_samples):
            word = random.choice(words)
            emb = model.get_learned_conditioning([word])
            z = torch.randn((1, 4, image_size // 8, image_size // 8)).to(devices[0])
            yield z, emb
    
    synthetic_forget_dl = list(generate_esd_batches(50))
    
    # Create protection instance
    protection = UnlearnIntervalProtection(
        targets=targets,
        lambda_interval=lambda_interval,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        reduced_dim=reduced_dim,
        infinity_scale=infinity_scale,
        use_actual_bounds=use_actual_bounds,
        normalize_protection=normalize_protection,
    )
    
    diffusion_model = model.model.diffusion_model
    
    # Setup protection
    protection.setup_protection(
        diffusion_model,
        synthetic_forget_dl,
        devices[0],
        remain_dataloader=None,
        forward_fn=esd_forward_fn,
    )
    
    # Mark non-target parameters (doesn't freeze to avoid breaking checkpointing)
    protection.freeze_non_target_params(diffusion_model)
    
    # Get only trainable parameters for optimizer
    trainable_params = protection.get_trainable_params(diffusion_model)
    log.info(f"Training {len(trainable_params)} parameters out of {sum(1 for _ in model.model.diffusion_model.parameters())} total")
    if not trainable_params:
        raise ValueError(
            f"InTAct selected no trainable parameters for targets={targets}. "
            "Check intact.targets against diffusion_model parameter names."
        )
    
    model.train()
    
    losses = []
    history_rows = []
    opt = torch.optim.Adam(trainable_params, lr=lr)
    criteria = torch.nn.MSELoss()
    
    targets_str = "_".join(targets)
    name = f"compvis-intact-esd-prompt_{word_print}-targets_{targets_str}-lambda_{lambda_interval}-lr_{lr}"
    
    quick_sample_till_t = lambda x, s, code, t: sample_model(
        model, sampler, x, image_size, image_size, ddim_steps, s, ddim_eta,
        start_code=code, till_T=t, verbose=False
    )
    
    # Training loop
    pbar = tqdm(range(iterations))
    for i in pbar:
        word = random.sample(words, 1)[0]
        emb_0 = model.get_learned_conditioning([""])
        emb_p = model.get_learned_conditioning([word])
        emb_n = model.get_learned_conditioning([f"{word}"])
        
        opt.zero_grad()
        
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        og_num = round((int(t_enc) / ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        
        start_code = torch.randn((1, 4, image_size // 8, image_size // 8)).to(devices[0])
        
        # Compute ESD loss
        base_loss, base_loss_val, _ = compute_esd_loss(
            model, model_orig, sampler, word, emb_0, emb_p, emb_n,
            t_enc, t_enc_ddpm, start_code, criteria, devices,
            start_guidance, negative_guidance, image_size, ddim_steps, ddim_eta
        )
        
        # Compute InTAct protection loss
        intact_loss = protection.compute_protection_loss(diffusion_model, devices[0])
        
        # Total loss
        total_loss = base_loss + intact_loss
        total_loss.backward()
        
        opt.step()
        
        base_loss_value = float(base_loss.detach().cpu())
        intact_loss_value = float(intact_loss.detach().cpu())
        total_loss_value = float(total_loss.detach().cpu())
        losses.append(total_loss_value)
        history_rows.append(
            {
                "step": len(history_rows),
                "epoch": 0,
                "batch": i,
                "total_loss": total_loss_value,
                "base_loss": base_loss_value,
                "intact_loss": intact_loss_value,
            }
        )
        pbar.set_postfix({
            "base": base_loss_value,
            "intact": intact_loss_value,
            "total": total_loss_value,
        })
        
        # Save checkpoint periodically
        if (i + 1) % 500 == 0 and i + 1 != iterations:
            save_model(model, name, i, save_diffusers=False)
    
    model.eval()
    save_model(model, name, None, config_path, diffusers_config_path)
    save_history(losses, name, word_print, history_rows=history_rows)
    
    return model


# ============================================================================
# Utility Functions
# ============================================================================

def moving_average(a, n=3):
    values = np.asarray(a, dtype=float)
    if values.size == 0:
        return values
    n = max(1, min(int(n), int(values.size)))
    ret = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.figure()
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)
    plt.close()


def save_model(model, name, num, compvis_config_file=None, diffusers_config_file=None,
               device="cpu", save_compvis=True, save_diffusers=True, model_save_dir="models"):
    folder_path = f"{model_save_dir}/{name}"
    os.makedirs(folder_path, exist_ok=True)
    
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    
    if save_compvis:
        torch.save(model.state_dict(), path)
    
    if save_diffusers and diffusers_config_file is not None:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device, 
                          save_dir=model_save_dir, checkpoint_path=path)
        diffusers_path = f"{folder_path}/{name.replace('compvis','diffusers')}.pt"
        if not os.path.exists(diffusers_path):
            raise FileNotFoundError(f"Diffusers export failed or wrote no checkpoint: {diffusers_path}")


def save_history(losses, name, word_print, history_rows=None, logs_dir="models"):
    folder_path = f"{logs_dir}/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) + "\n" for i in losses])
    if history_rows is not None:
        fieldnames = list(dict.fromkeys(key for row in history_rows for key in row.keys()))
        with open(f"{folder_path}/training_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history_rows)
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="InTAct Unlearn",
        description="InTAct unlearning for Stable Diffusion with composable base methods"
    )
    
    # Base method selection
    parser.add_argument(
        "--base_method",
        help="Base unlearning method: ga, rl, nsfw, esd",
        type=str,
        required=False,
        default=None,
        choices=["ga", "rl", "nsfw", "esd"],
    )
    
    # Common parameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ckpt_path", type=str, 
                        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--config_path", type=str,
                        default="configs/stable-diffusion/v1-intact.yaml")
    parser.add_argument("--diffusers_config_path", type=str,
                        default="diffusers_unet_config.json")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--ddim_steps", type=int, default=50)
    
    # GA/RL specific
    parser.add_argument("--class_to_forget", type=str, default=None)
    parser.add_argument("--forget_classes", nargs="+", default=None)
    parser.add_argument("--forget_concepts", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    
    # ESD specific
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start_guidance", type=float, default=None)
    parser.add_argument("--negative_guidance", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--devices", type=str, default="0,0",
                        help="Two devices for ESD: training,frozen model")
    parser.add_argument("--seperator", type=str, default=None)
    
    # InTAct parameters
    parser.add_argument("--targets", type=str, nargs="+",
                        default=None,
                        help="Target layer patterns for protection (e.g., to_q to_k to_v for cross-attn QKV)")
    parser.add_argument("--lambda_interval", type=float, default=None,
                        help="Weight for InTAct protection loss")
    parser.add_argument("--lower_percentile", type=float, default=None)
    parser.add_argument("--upper_percentile", type=float, default=None)
    parser.add_argument("--reduced_dim", type=int, default=None)
    parser.add_argument("--infinity_scale", type=float, default=None)
    parser.add_argument("--use_actual_bounds", action="store_true",
                        help="Use actual min/max from remain+forget instead of scaled bounds")
    parser.add_argument("--normalize_protection", action="store_true", default=None)

    # DASH warm-start parameters for class/concept forgetting
    parser.add_argument("--dash_warm_start", action="store_true")
    parser.add_argument("--dash_target", type=str, default="unet")
    parser.add_argument(
        "--dash_signal_mode",
        type=str,
        default="preserve_complement",
        choices=["retain_only", "forget_perp_retain", "preserve_complement"],
    )
    parser.add_argument(
        "--plasticity_granularity",
        type=str,
        default="per_filter",
        choices=["global", "per_filter"],
    )
    parser.add_argument("--dash_grad_aggregation", type=str, default="mean", choices=["mean", "ema"])
    parser.add_argument("--dash_alpha", type=float, default=0.1)
    parser.add_argument("--dash_num_aug", type=int, default=10)
    parser.add_argument("--dash_aug_mode", type=str, default="none", choices=["none", "default"])
    parser.add_argument("--dash_min_shrink", type=float, default=0.004)
    parser.add_argument("--dash_svd_truncate_evr", type=float, default=0.95)
    parser.add_argument("--dash_preserve_forget_evr", type=float, default=0.95)
    parser.add_argument("--dash_include_bias", action="store_true")
    parser.add_argument("--dash_log_cosine_histograms", action="store_true")
    parser.add_argument("--dash_cosine_hist_bins", type=int, default=50)
    parser.add_argument("--dash_retain_batches", type=int, default=None)
    parser.add_argument("--dash_forget_batches", type=int, default=None)
    parser.add_argument("--bn_recalibrate", action="store_true")
    parser.add_argument("--bn_recalib_batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--full_retain_per_epoch", action="store_true")
    
    args = parser.parse_args()
    
    # Load training config from YAML file
    training_config = load_training_config(args.config_path)
    
    # Merge config with command-line args (args override config)
    def get_param(arg_val, config_key, default):
        """Get parameter from args (priority), then config, then default."""
        if arg_val is not None:
            return arg_val
        if training_config and hasattr(training_config, config_key):
            return getattr(training_config, config_key)
        return default
    
    # Extract parameters with fallbacks
    base_method = get_param(args.base_method, 'base_method', 'rl')
    lr = get_param(args.lr, 'lr', 1e-5)
    alpha = get_param(args.alpha, 'alpha', 0.1)
    batch_size = get_param(args.batch_size, 'batch_size', 8)
    epochs = get_param(args.epochs, 'epochs', 5)
    
    # ESD params
    start_guidance = get_param(args.start_guidance, 'start_guidance', 3.0)
    negative_guidance = get_param(args.negative_guidance, 'negative_guidance', 1.0)
    iterations = get_param(args.iterations, 'iterations', 1000)
    
    # InTAct params
    targets = get_param(args.targets, 'targets', ['to_q', 'to_k', 'to_v'])
    lambda_interval = get_param(args.lambda_interval, 'lambda_interval', 1.0)
    lower_percentile = get_param(args.lower_percentile, 'lower_percentile', 0.05)
    upper_percentile = get_param(args.upper_percentile, 'upper_percentile', 0.95)
    reduced_dim = get_param(args.reduced_dim, 'reduced_dim', 32)
    infinity_scale = get_param(args.infinity_scale, 'infinity_scale', 20.0)
    use_actual_bounds = args.use_actual_bounds if args.use_actual_bounds else get_param(None, 'use_actual_bounds', False)
    normalize_protection = get_param(args.normalize_protection, 'normalize_protection', True)
    
    # Forget config
    class_to_forget = args.class_to_forget
    if class_to_forget is None and training_config and hasattr(training_config, 'forget'):
        if hasattr(training_config.forget, 'class_to_forget'):
            class_to_forget = str(training_config.forget.class_to_forget)
    if class_to_forget is None:
        class_to_forget = '0'
    
    prompt = args.prompt
    if prompt is None and training_config and hasattr(training_config, 'forget'):
        if hasattr(training_config.forget, 'prompt'):
            prompt = training_config.forget.prompt
    
    log.info(f"Configuration loaded: base_method={base_method}, targets={targets}")
    log.info(f"  lambda_interval={lambda_interval}, lr={lr}, alpha={alpha}")
    log.info(f"  class_to_forget={class_to_forget}, prompt={prompt}")
    
    # Device setup
    device = f"cuda:{args.device}"
    
    # InTAct common params
    intact_params = {
        "targets": targets,
        "lambda_interval": lambda_interval,
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "reduced_dim": reduced_dim,
        "infinity_scale": infinity_scale,
        "use_actual_bounds": use_actual_bounds,
        "normalize_protection": normalize_protection,
    }
    dash_config = {
        "warm_start": args.dash_warm_start,
        "target": args.dash_target,
        "signal_mode": args.dash_signal_mode,
        "plasticity_granularity": args.plasticity_granularity,
        "attention_head_wise": args.dash_attention_head_wise,
        "grad_aggregation": args.dash_grad_aggregation,
        "alpha": args.dash_alpha,
        "num_aug": args.dash_num_aug,
        "aug_mode": args.dash_aug_mode,
        "min_shrink": args.dash_min_shrink,
        "svd_truncate_evr": args.dash_svd_truncate_evr,
        "preserve_forget_evr": args.dash_preserve_forget_evr,
        "include_bias": args.dash_include_bias,
        "log_cosine_histograms": args.dash_log_cosine_histograms,
        "cosine_hist_bins": args.dash_cosine_hist_bins,
        "retain_batches": args.dash_retain_batches,
        "forget_batches": args.dash_forget_batches,
        "bn_recalibrate": args.bn_recalibrate,
        "bn_recalib_batches": args.bn_recalib_batches,
    }
    
    # Run appropriate method
    if base_method in ["ga", "rl"]:
        intact_unlearn_class(
            class_to_forget=class_to_forget,
            base_method=base_method,
            alpha=alpha,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            config_path=args.config_path,
            ckpt_path=args.ckpt_path,
            diffusers_config_path=args.diffusers_config_path,
            device=device,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            dash_config=dash_config,
            seed=args.seed,
            forget_classes=args.forget_classes,
            forget_concepts=args.forget_concepts,
            full_retain_per_epoch=args.full_retain_per_epoch,
            **intact_params,
        )
    
    elif args.base_method == "nsfw":
        intact_unlearn_nsfw(
            alpha=alpha,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            config_path=args.config_path,
            ckpt_path=args.ckpt_path,
            diffusers_config_path=args.diffusers_config_path,
            device=device,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            **intact_params,
        )
    
    elif args.base_method == "esd":
        if prompt is None:
            raise ValueError("--prompt is required for ESD base method")
        
        devices = [f"cuda:{d.strip()}" for d in args.devices.split(",")]
        
        intact_unlearn_esd(
            prompt=prompt,
            start_guidance=start_guidance,
            negative_guidance=negative_guidance,
            iterations=iterations,
            lr=lr,
            config_path=args.config_path,
            ckpt_path=args.ckpt_path,
            diffusers_config_path=args.diffusers_config_path,
            devices=devices,
            seperator=args.seperator,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            **intact_params,
        )
