"""
InTAct Unlearning for Flux (Rectified Flow Transformers)

Implements InTAct (Interval-based Task Activation Consolidation) for Flux,
composable with multiple base unlearning methods:
- ESD (Erased Stable Diffusion) — noise-prediction fine-tuning with unconditional guidance
- RL  (Random Label / Negative prompt) — predicts noise conditioned on negative prompt
- EA  (EraseAnything) — ESD + cross-attention deactivation + InfoNCE contrastive loss

InTAct adds interval protection loss on top of any base method:
    total_loss = base_loss + lambda_interval * intact_loss

NOTE: No LoRA — full weight fine-tuning of target layers. InTAct requires a properly
formed activation space that LoRA's low-rank structure cannot provide.

Usage:
    python intact_train.py --config configs/intact/pipeline.yaml
"""

import argparse
import copy
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directories to path (must be before library imports for setup_cache)
sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root for InTAct + setup_cache
sys.path.insert(0, str(Path(__file__).parent))          # For Flux modules

import setup_cache  # noqa: E402  — must precede torch / HF imports

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

from InTAct.intact import UnlearnIntervalProtection

import transformers
import diffusers
from diffusers import (
    AutoencoderKL,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from safetensors.torch import save_file

from tools.prompt_process import encode_prompt
from tools.scheduler_process import CustomFlowMatchEulerDiscreteScheduler
from utils.esd_utils import latent_sample, predict_noise, flux_pack_latents, flux_unpack_latents, _prepare_latent_image_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# ============================================================================
# Model Loading
# ============================================================================

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, subfolder="text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(class_one, class_two, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def load_flux_components(args, device):
    """Load all Flux model components."""
    log.info(f"Loading Flux from {args.pretrained_model_name_or_path}")

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    noise_scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    text_encoder_one, text_encoder_two = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, args
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant,
    ).to(device)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(device, dtype=weight_dtype)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)

    # Freeze encoders and VAE
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    return {
        "transformer": transformer,
        "vae": vae,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "noise_scheduler": noise_scheduler,
        "weight_dtype": weight_dtype,
    }


# ============================================================================
# Text Embedding Helpers
# ============================================================================

def make_compute_text_embeddings(text_encoders, tokenizers, max_seq_len, device):
    """Create a closure for computing text embeddings."""
    def compute_text_embeddings(prompt):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, max_seq_len
            )
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            text_ids = text_ids.to(device)
        return prompt_embeds, pooled_prompt_embeds, text_ids
    return compute_text_embeddings


# ============================================================================
# Flux Forward Function for InTAct Activation Collection
# ============================================================================

def flux_forward_fn(model, batch, device,
                    compute_text_embeddings=None,
                    vae=None, noise_scheduler=None,
                    weight_dtype=torch.bfloat16,
                    prompts=None,
                    image_size=512,
                    **kwargs):
    """
    Forward function for Flux transformer activation collection.
    InTAct hooks collect activations from target layers during this forward pass.

    Args:
        model: FluxTransformer2DModel
        batch: dict with 'pixel_values' [B,C,H,W] and optionally 'prompts' list
        device: CUDA device
        compute_text_embeddings: closure for text encoding
        vae: AutoencoderKL
        noise_scheduler: CustomFlowMatchEulerDiscreteScheduler
        weight_dtype: dtype for mixed precision
        prompts: fallback prompts if batch doesn't contain them
    """
    if isinstance(batch, (tuple, list)):
        # Synthetic data: (latents, prompt_embeds, pooled_embeds, text_ids)
        z, prompt_embeds, pooled_embeds, text_ids = batch
        z = z.to(device, dtype=weight_dtype)
        prompt_embeds = prompt_embeds.to(device, dtype=weight_dtype)
        pooled_embeds = pooled_embeds.to(device, dtype=weight_dtype)
        text_ids = text_ids.to(device, dtype=weight_dtype)
        model_input = z
    elif isinstance(batch, dict) and "pixel_values" in batch:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=device)
        with torch.no_grad():
            model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        model_input = model_input.to(dtype=weight_dtype)

        batch_prompts = batch.get("prompts", prompts)
        if batch_prompts is None:
            batch_prompts = [""] * model_input.shape[0]

        prompt_embeds, pooled_embeds, text_ids = compute_text_embeddings(batch_prompts)
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    n = model_input.shape[0]

    # Prepare latent IDs
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        n, model_input.shape[2] // 2, model_input.shape[3] // 2,
        device, weight_dtype,
    )

    # Add noise
    noise = torch.randn_like(model_input)
    t = torch.randint(0, 1000, (n,), device=device)
    noisy_input = noise_scheduler.add_noise(model_input, noise, t)

    # Pack latents
    packed = FluxPipeline._pack_latents(
        noisy_input, n, model_input.shape[1],
        model_input.shape[2], model_input.shape[3],
    )

    guidance = torch.tensor([3.5], device=device, dtype=weight_dtype).expand(n)

    # Forward through transformer (triggers hooks)
    model(
        hidden_states=packed.to(dtype=weight_dtype, device=device),
        timestep=t.float() / 1000,
        guidance=guidance,
        pooled_projections=pooled_embeds.to(dtype=weight_dtype, device=device),
        encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype, device=device),
        txt_ids=text_ids.to(dtype=weight_dtype, device=device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=device),
        return_dict=False,
    )


# ============================================================================
# Base Method: ESD Loss
# ============================================================================

def compute_esd_loss(transformer, noise_scheduler, compute_text_embeddings,
                     vae, prompt, neg_prompt, criteria, device, weight_dtype,
                     negative_guidance=1.0, ddim_steps=28, image_size=512):
    """
    ESD (Erased Stable Diffusion) loss for Flux.

    Generates a latent with the concept, then steers the model's noise prediction
    away from concept-conditioned prediction toward the unconditional/negative prediction.

    L_ESD = ||e_n(z,c_forget) - [e_0(z,c_neg) - γ(e_p(z,c_forget) - e_0(z,c_neg))]||²
    """
    vae_scale_factor = 2 ** len(vae.config.block_out_channels)
    num_channels = transformer.config.in_channels

    # Encode prompts
    emb_0, pooled_0, tid_0 = compute_text_embeddings(neg_prompt)
    emb_p, pooled_p, tid_p = compute_text_embeddings(prompt)

    # Random timestep
    t_enc = torch.randint(ddim_steps, (1,), device=device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=device)

    start_guidance = torch.tensor([3.0], device=device)

    with torch.no_grad():
        # Generate latent with concept using current model
        z, latent_image_ids = latent_sample(
            transformer, noise_scheduler, 1, num_channels,
            image_size, image_size,
            emb_p.to(device), pooled_p.to(device), tid_p.to(device),
            start_guidance, int(ddim_steps), vae_scale_factor,
        )
        # Get scores with frozen detached parameters 
        e_0 = predict_noise(
            transformer, z, emb_0, pooled_0, tid_0, latent_image_ids,
            guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True,
        )
        e_p = predict_noise(
            transformer, z, emb_p, pooled_p, tid_p, latent_image_ids,
            guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True,
        )

    # Get conditional score from training model (with gradients)
    e_n = predict_noise(
        transformer, z, emb_p, pooled_p, tid_p, latent_image_ids,
        guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True,
    )
    e_0.requires_grad = False
    e_p.requires_grad = False

    # ESD objective
    target = e_0.to(device) - (negative_guidance * (e_p.to(device) - e_0.to(device)))
    loss = criteria(e_n.to(device), target)

    return loss, t_enc_ddpm


# ============================================================================
# Base Method: RL (Random/Negative Label) Loss
# ============================================================================

def compute_rl_loss(transformer, noise_scheduler, compute_text_embeddings,
                    vae, prompt, neg_prompt, criteria, device, weight_dtype,
                    ddim_steps=28, image_size=512):
    """
    Random Label loss for Flux.

    Trains the model to predict noise as if the image was conditioned on a
    negative/unrelated prompt instead of the concept prompt.

    L_RL = ||f_θ(z_t, t, c_forget) - f_θ_frozen(z_t, t, c_neg)||²
    """
    vae_scale_factor = 2 ** len(vae.config.block_out_channels)
    num_channels = transformer.config.in_channels

    emb_p, pooled_p, tid_p = compute_text_embeddings(prompt)
    emb_neg, pooled_neg, tid_neg = compute_text_embeddings(neg_prompt)

    # Random timestep
    t_enc = torch.randint(ddim_steps, (1,), device=device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=device)

    start_guidance = torch.tensor([3.0], device=device)

    with torch.no_grad():
        # Generate latent with concept
        z, latent_image_ids = latent_sample(
            transformer, noise_scheduler, 1, num_channels,
            image_size, image_size,
            emb_p.to(device), pooled_p.to(device), tid_p.to(device),
            start_guidance, int(ddim_steps), vae_scale_factor,
        )
        # Target: noise prediction with negative prompt (frozen)
        e_neg = predict_noise(
            transformer, z, emb_neg, pooled_neg, tid_neg, latent_image_ids,
            guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True,
        )

    # Current model prediction with concept prompt (with gradients)
    e_n = predict_noise(
        transformer, z, emb_p, pooled_p, tid_p, latent_image_ids,
        guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True,
    )
    e_neg.requires_grad = False

    loss = criteria(e_n.to(device), e_neg.to(device))

    return loss, t_enc_ddpm


# ============================================================================
# Base Method: EraseAnything (EA) Loss
# ============================================================================

def compute_ea_loss(transformer, noise_scheduler, compute_text_embeddings,
                    vae, prompt, neg_prompt, key_word, tokenizer_t5,
                    criteria, device, weight_dtype,
                    negative_guidance=1.0, ddim_steps=28, image_size=512,
                    lamb_esd=1.0, lamb_attn=0.001):
    """
    EraseAnything loss for Flux (ESD + attention deactivation).

    L_EA = λ_esd * L_ESD + λ_attn * L_attn

    L_attn minimizes the attention weights at token positions of the key_word,
    forcing the model to stop attending to the erased concept.

    Note: This is EA without LoRA reconstruction and InfoNCE — those components
    relied on LoRA which is not used with InTAct.
    """
    from utils.find_token import get_word_index

    vae_scale_factor = 2 ** len(vae.config.block_out_channels)
    num_channels = transformer.config.in_channels

    emb_0, pooled_0, tid_0 = compute_text_embeddings(neg_prompt)
    emb_p, pooled_p, tid_p = compute_text_embeddings(prompt)

    # Random timestep
    t_enc = torch.randint(ddim_steps, (1,), device=device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=device)

    start_guidance = torch.tensor([3.0], device=device)

    # ---- ESD component ----
    with torch.no_grad():
        z, latent_image_ids = latent_sample(
            transformer, noise_scheduler, 1, num_channels,
            image_size, image_size,
            emb_p.to(device), pooled_p.to(device), tid_p.to(device),
            start_guidance, int(ddim_steps), vae_scale_factor,
        )
        e_0 = predict_noise(
            transformer, z, emb_0, pooled_0, tid_0, latent_image_ids,
            guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True,
        )
        e_p = predict_noise(
            transformer, z, emb_p, pooled_p, tid_p, latent_image_ids,
            guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True,
        )

    e_n = predict_noise(
        transformer, z, emb_p, pooled_p, tid_p, latent_image_ids,
        guidance=start_guidance, timesteps=t_enc_ddpm, CPU_only=True,
    )
    e_0.requires_grad = False
    e_p.requires_grad = False

    esd_target = e_0.to(device) - (negative_guidance * (e_p.to(device) - e_0.to(device)))
    loss_esd = criteria(e_n.to(device), esd_target)

    # ---- Attention deactivation component ----
    # Get token indices of the keyword
    remove_indices = get_word_index(prompt if isinstance(prompt, str) else prompt[0],
                                     key_word, tokenizer_t5)

    if len(remove_indices) > 0:
        # Create a noisy input for attention extraction
        fake_input = torch.randn(1, num_channels, image_size // 8, image_size // 8,
                                  device=device, dtype=weight_dtype)
        noise = torch.randn_like(fake_input)
        noisy = noise_scheduler.add_noise(fake_input, noise, t_enc_ddpm)

        packed = FluxPipeline._pack_latents(
            noisy, 1, num_channels,
            image_size // 8, image_size // 8,
        )
        latent_ids = FluxPipeline._prepare_latent_image_ids(
            1, (image_size // 8) // 2, (image_size // 8) // 2,
            device, weight_dtype,
        )

        guidance = torch.tensor([3.5], device=device, dtype=weight_dtype)

        model_pred, attn_maps = transformer(
            hidden_states=packed.to(dtype=weight_dtype),
            timestep=t_enc_ddpm / 1000,
            guidance=guidance,
            pooled_projections=pooled_p.to(dtype=weight_dtype, device=device),
            encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=device),
            txt_ids=tid_p.to(dtype=weight_dtype, device=device),
            img_ids=latent_ids.to(dtype=weight_dtype, device=device),
            return_dict=False,
        )[0:2]

        # Mask: select only the target token positions
        attn_map_mask = torch.zeros_like(attn_maps).to(device)
        attn_map_mask[..., remove_indices] = 1.0

        loss_attn = torch.norm(attn_map_mask * attn_maps, dim=(0, 1)).sum()
    else:
        loss_attn = torch.tensor(0.0, device=device)

    total_loss = lamb_esd * loss_esd + lamb_attn * loss_attn

    return total_loss, t_enc_ddpm, loss_esd.item(), loss_attn.item()


# ============================================================================
# InTAct Setup for Flux
# ============================================================================

def setup_intact_protection(
    transformer,
    forget_dataloader,
    device,
    compute_text_embeddings,
    vae,
    noise_scheduler,
    weight_dtype,
    remain_dataloader=None,
    targets=None,
    lambda_interval=1.0,
    lower_percentile=0.05,
    upper_percentile=0.95,
    reduced_dim=32,
    infinity_scale=20.0,
    use_actual_bounds=False,
    normalize_protection=True,
    image_size=512,
):
    """
    Setup InTAct protection for Flux transformer.

    NOTE: remain_dataloader is ONLY used for computing InTAct activation boundaries.
    It is NOT used for any reconstruction or regularization loss.
    """
    if targets is None:
        targets = [
            f"transformer_blocks.{b}.{layer}"
            for b in [12, 14, 16, 18]
            for layer in ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"]
        ]

    log.info(f"Setting up InTAct protection with targets: {targets}")

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

    # Create forward function closure
    def forward_fn(model, batch, dev, **kwargs):
        return flux_forward_fn(
            model, batch, dev,
            compute_text_embeddings=compute_text_embeddings,
            vae=vae, noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            image_size=image_size,
            **kwargs,
        )

    protection.setup_protection(
        transformer,
        forget_dataloader,
        device,
        remain_dataloader=remain_dataloader,
        forward_fn=forward_fn,
    )

    return protection


# ============================================================================
# Synthetic Data Generation for ESD-style Forward
# ============================================================================

def generate_synthetic_forget_data(transformer, noise_scheduler, compute_text_embeddings,
                                   prompts, device, weight_dtype, n_samples=50,
                                   image_size=512):
    """
    Generate synthetic (latent, embedding) pairs for InTAct activation collection.
    Used when no real image dataset is available (ESD-style).
    """
    vae_scale_factor = 2 ** 3  # Flux VAE has 3 block_out_channels levels
    num_channels = transformer.config.in_channels
    data = []

    log.info(f"Generating {n_samples} synthetic forget samples for InTAct setup...")
    for i in range(n_samples):
        prompt = prompts if isinstance(prompts, str) else random.choice(prompts)
        emb_p, pooled_p, tid_p = compute_text_embeddings(prompt)
        z = torch.randn(1, num_channels, image_size // 8, image_size // 8,
                        device=device, dtype=weight_dtype)
        data.append((z.cpu(), emb_p.cpu(), pooled_p.cpu(), tid_p.cpu()))

    return data


# ============================================================================
# Saving
# ============================================================================

def save_model_weights(transformer, output_dir, name, weight_dtype):
    """Save fine-tuned transformer weights."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}.safetensors")

    transformer = transformer.to(weight_dtype)
    state_dict = transformer.state_dict()
    save_file(state_dict, save_path)
    log.info(f"Saved model weights to {save_path}")
    return save_path


def save_history(losses, output_dir, name):
    """Save loss history and plot."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}_loss.txt"), "w") as f:
        for l in losses:
            f.write(f"{l}\n")

    if len(losses) > 3:
        # Moving average
        arr = np.array(losses)
        n = min(10, len(arr))
        ret = np.cumsum(arr, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ma = ret[n - 1:] / n

        plt.figure(figsize=(10, 6))
        plt.plot(ma, label="loss (moving avg)")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss — {name}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{name}_loss.png"))
        plt.close()


# ============================================================================
# Main Training Function
# ============================================================================

def intact_unlearn(args):
    """Main InTAct unlearning function for Flux."""
    device = f"cuda:{args.device}" if not args.device.startswith("cuda") else args.device
    base_method = args.base_method
    prompt = args.instance_prompt
    neg_prompt = getattr(args, "neg_prompt", "")
    key_word = getattr(args, "key_word", None)

    log.info(f"=== InTAct Unlearning for Flux ===")
    log.info(f"  base_method: {base_method}")
    log.info(f"  prompt: {prompt}")
    log.info(f"  neg_prompt: {neg_prompt}")
    log.info(f"  key_word: {key_word}")

    # Load model components
    components = load_flux_components(args, device)
    transformer = components["transformer"]
    vae = components["vae"]
    noise_scheduler = components["noise_scheduler"]
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    weight_dtype = components["weight_dtype"]
    tokenizer_two = components["tokenizer_two"]

    text_encoders = [components["text_encoder_one"], components["text_encoder_two"]]
    tokenizers = [components["tokenizer_one"], components["tokenizer_two"]]

    compute_text_embeddings = make_compute_text_embeddings(
        text_encoders, tokenizers, args.max_sequence_length, device
    )

    # ---- Determine which layers to train ----
    # Expand (target_blocks × target_layers) into fully-qualified module patterns
    intact_blocks = getattr(args, "intact_target_blocks", None)
    intact_layers = getattr(args, "intact_target_layers", None)

    if intact_blocks is not None and intact_layers is not None:
        # Build explicit patterns: transformer_blocks.{i}.{layer}
        intact_targets = [
            f"transformer_blocks.{b}.{layer}"
            for b in intact_blocks
            for layer in intact_layers
        ]
        log.info(f"  Target blocks: {intact_blocks}")
        log.info(f"  Target layers: {intact_layers}")
    else:
        # Fallback: use pre-built patterns list
        intact_targets = getattr(args, "intact_targets", ["attn.add_q_proj", "attn.add_k_proj"])
        if isinstance(intact_targets, str):
            intact_targets = [t.strip() for t in intact_targets.split(",")]

    log.info(f"  InTAct targets ({len(intact_targets)}): {intact_targets[:6]}{'...' if len(intact_targets) > 6 else ''}")

    # Freeze everything first
    transformer.requires_grad_(False)

    # Unfreeze target layers
    unfrozen_count = 0
    for name, module in transformer.named_modules():
        should_unfreeze = False
        for pattern in intact_targets:
            if pattern.lower() in name.lower():
                should_unfreeze = True
                break
        if should_unfreeze:
            for p in module.parameters():
                p.requires_grad = True
                unfrozen_count += 1

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    log.info(f"  Trainable parameters: {len(trainable_params)} (unfrozen {unfrozen_count})")

    if len(trainable_params) == 0:
        raise ValueError(f"No trainable parameters found for targets {intact_targets}. "
                         f"Check target patterns against transformer module names.")

    # Make trainable params float32 for mixed precision stability
    for p in trainable_params:
        p.data = p.data.float()

    # ---- Setup forget/remaining data for InTAct ----
    forget_data = None
    remain_data = None

    # if NSFW dataset paths are provided, use real images for activation bounds
    if hasattr(args, "nsfw_data_path") and args.nsfw_data_path and \
       hasattr(args, "not_nsfw_data_path") and args.not_nsfw_data_path:
        from eval.dataset import setup_forget_nsfw_data
        batch_sz = args.batch_size or 8
        forget_dl, remain_dl = setup_forget_nsfw_data(
            batch_sz,
            args.resolution,
            nsfw_data_path=args.nsfw_data_path,
            not_nsfw_data_path=args.not_nsfw_data_path,
        )
        forget_data = forget_dl
        if args.intact_use_actual_bounds:
            remain_data = remain_dl
    else:
        # no dataset, fall back to synthetic prompts as before
        forget_data = generate_synthetic_forget_data(
            transformer, noise_scheduler_copy, compute_text_embeddings,
            prompt, device, weight_dtype, n_samples=args.intact_n_samples,
            image_size=args.resolution,
        )
        if args.intact_use_actual_bounds and hasattr(args, "remain_prompts") and args.remain_prompts:
            remain_prompts = args.remain_prompts
            if isinstance(remain_prompts, str):
                remain_prompts = [p.strip() for p in remain_prompts.split(";")]
            remain_data = generate_synthetic_forget_data(
                transformer, noise_scheduler_copy, compute_text_embeddings,
                remain_prompts, device, weight_dtype, n_samples=args.intact_n_samples,
                image_size=args.resolution,
            )

    # ---- Setup InTAct protection ----
    protection = setup_intact_protection(
        transformer, forget_data, device,
        compute_text_embeddings, vae, noise_scheduler_copy, weight_dtype,
        remain_dataloader=remain_data,
        targets=intact_targets,
        lambda_interval=args.intact_lambda,
        lower_percentile=args.intact_lower_pct,
        upper_percentile=args.intact_upper_pct,
        reduced_dim=args.intact_reduced_dim,
        infinity_scale=args.intact_infinity_scale,
        use_actual_bounds=args.intact_use_actual_bounds,
        normalize_protection=args.intact_normalize_protection,
        image_size=args.resolution,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))
    criteria = torch.nn.MSELoss()

    # ---- Training ----
    losses_history = []
    transformer.train()

    # Build model name for saving
    intact_blocks = getattr(args, "intact_target_blocks", None)
    intact_layers_cfg = getattr(args, "intact_target_layers", None)
    if intact_blocks is not None and intact_layers_cfg is not None:
        blocks_str = "-".join(str(b) for b in intact_blocks)
        layers_str = "_".join([l.split(".")[-1] for l in intact_layers_cfg])
        targets_str = f"blk{blocks_str}_{layers_str}"
    else:
        targets_str = "_".join([t.replace(".", "-") for t in intact_targets])
    name = (f"flux-intact-{base_method}-{key_word or 'concept'}"
            f"-targets_{targets_str}-lambda_{args.intact_lambda}"
            f"-steps_{args.max_train_steps}-lr_{args.learning_rate}")

    log.info(f"  Model name: {name}")
    log.info(f"  Starting training for {args.max_train_steps} steps...")

    pbar = tqdm(range(args.max_train_steps), desc="Training")
    for step in pbar:
        optimizer.zero_grad()

        # Compute base method loss
        if base_method == "esd":
            base_loss, t_enc_ddpm = compute_esd_loss(
                transformer, noise_scheduler_copy, compute_text_embeddings,
                vae, prompt, neg_prompt, criteria, device, weight_dtype,
                negative_guidance=args.negative_guidance,
                ddim_steps=args.ddim_steps, image_size=args.resolution,
            )
            log_dict = {"esd": base_loss.item()}

        elif base_method == "rl":
            base_loss, t_enc_ddpm = compute_rl_loss(
                transformer, noise_scheduler_copy, compute_text_embeddings,
                vae, prompt, neg_prompt, criteria, device, weight_dtype,
                ddim_steps=args.ddim_steps, image_size=args.resolution,
            )
            log_dict = {"rl": base_loss.item()}

        elif base_method == "ea":
            if key_word is None:
                raise ValueError("key_word is required for EraseAnything (ea) base method")
            base_loss, t_enc_ddpm, esd_val, attn_val = compute_ea_loss(
                transformer, noise_scheduler_copy, compute_text_embeddings,
                vae, prompt, neg_prompt, key_word, tokenizer_two,
                criteria, device, weight_dtype,
                negative_guidance=args.negative_guidance,
                ddim_steps=args.ddim_steps, image_size=args.resolution,
                lamb_esd=args.lamb_esd, lamb_attn=args.lamb_attn,
            )
            log_dict = {"esd": esd_val, "attn": attn_val}
        else:
            raise ValueError(f"Unknown base method: {base_method}")

        # Compute InTAct protection loss
        intact_loss = protection.compute_protection_loss(transformer, device)

        # Total loss
        total_loss = base_loss + intact_loss
        total_loss.backward()
        optimizer.step()

        log_dict["intact"] = intact_loss.item()
        log_dict["total"] = total_loss.item()
        losses_history.append(total_loss.item())

        pbar.set_postfix(**{k: f"{v:.4f}" for k, v in log_dict.items()})

        # Periodic checkpoint
        if (step + 1) % args.checkpointing_steps == 0 and (step + 1) < args.max_train_steps:
            ckpt_name = f"{name}-step_{step+1}"
            save_model_weights(transformer, args.output_dir, ckpt_name, weight_dtype)

    # Final save
    save_model_weights(transformer, args.output_dir, name, weight_dtype)
    logs_dir = getattr(args, "logs_dir", args.output_dir)
    save_history(losses_history, logs_dir, name)

    log.info("Training complete!")
    return name


# ============================================================================
# Entry Point
# ============================================================================

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InTAct Unlearning for Flux")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    cli_args = parser.parse_args()

    config = load_config(cli_args.config)

    # Flatten config into args namespace — with proper key mapping
    # The YAML has sections: unlearn, intact, paths, pipeline, etc.
    # intact_unlearn() expects: args.intact_targets, args.intact_lambda, etc.
    # So we prefix 'intact' section keys and remap 'paths' keys.
    args = argparse.Namespace()

    # Map from (section, key) -> attribute name
    INTACT_KEY_MAP = {
        "target_blocks": "intact_target_blocks",
        "target_layers": "intact_target_layers",
        "targets": "intact_targets",       # legacy fallback
        "lambda_interval": "intact_lambda",
        "lower_percentile": "intact_lower_pct",
        "upper_percentile": "intact_upper_pct",
        "reduced_dim": "intact_reduced_dim",
        "infinity_scale": "intact_infinity_scale",
        "use_actual_bounds": "intact_use_actual_bounds",
        "normalize_protection": "intact_normalize_protection",
        "n_samples": "intact_n_samples",
    }

    PATHS_KEY_MAP = {
        "model_save_dir": "output_dir",
        "logs_dir": "logs_dir",
    }

    for section_key, section_val in config.items():
        if isinstance(section_val, dict):
            for k, v in section_val.items():
                if section_key == "intact" and k in INTACT_KEY_MAP:
                    setattr(args, INTACT_KEY_MAP[k], v)
                elif section_key == "paths" and k in PATHS_KEY_MAP:
                    setattr(args, PATHS_KEY_MAP[k], v)
                else:
                    setattr(args, k, v)
        else:
            setattr(args, section_key, section_val)

    # Device: pipeline section sets 'device' as just device id like "0"
    if hasattr(args, "device"):
        args.device = str(args.device)

    # Defaults
    defaults = {
        "revision": None,
        "variant": None,
        "mixed_precision": "bf16",
        "max_sequence_length": 256,
        "resolution": 512,
        "ddim_steps": 28,
        "negative_guidance": 1.0,
        "neg_prompt": "",
        "key_word": None,
        "checkpointing_steps": 500,
        "output_dir": "/net/tscratch/people/plgphelm/unl/Flux/models",
        "logs_dir": "/net/tscratch/people/plgphelm/unl/Flux/logs",
        "lamb_esd": 1.0,
        "lamb_attn": 0.001,
        "intact_target_blocks": [12, 14, 16, 18],
        "intact_target_layers": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
        "intact_lambda": 1.0,
        "intact_lower_pct": 0.05,
        "intact_upper_pct": 0.95,
        "intact_reduced_dim": 32,
        "intact_infinity_scale": 20.0,
        "intact_use_actual_bounds": True,
        "intact_normalize_protection": True,
        "intact_n_samples": 50,
        "remain_prompts": None,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    name = intact_unlearn(args)
    print(f"Done. Model saved as: {name}")
