"""
Image generation for Flux evaluation pipeline.

Loads a Flux model (base + optional fine-tuned transformer weights) and generates
images from a CSV of prompts.

CSV format: case_number, prompt, evaluation_seed
Images saved as: <save_path>/<model_name>/<case_number>_<sample>.png

Supports batched generation (--batch_size / batch_size parameter) for efficient
large-scale evaluation (e.g. 30K MS-COCO prompts).
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import FluxPipeline
from safetensors.torch import load_file


def _load_flux_pipeline(base_model_path, model_name, model_dir, device):
    """Load FluxPipeline with optional fine-tuned weights (shared helper)."""
    print(f"Loading Flux pipeline from {base_model_path}")
    pipe = FluxPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)

    if model_name:
        weights_path = os.path.join(model_dir, f"{model_name}.safetensors")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_dir, model_name, f"{model_name}.safetensors")

        if os.path.exists(weights_path):
            print(f"Loading fine-tuned weights from {weights_path}")
            state_dict = load_file(weights_path)
            pipe.transformer.load_state_dict(state_dict)
            print("Successfully loaded fine-tuned transformer weights")
        else:
            print(f"WARNING: Fine-tuned weights not found at {weights_path}")
            print("Using base model weights")

    pipe = pipe.to(device)
    return pipe


def generate_images(
    model_name,
    prompts_path,
    save_path,
    device="cuda:0",
    guidance_scale=3.5,
    image_size=512,
    ddim_steps=28,
    num_samples=10,
    from_case=0,
    base_model_path="black-forest-labs/FLUX.1-dev",
    model_dir="models",
    max_prompts=None,
    batch_size=1,
    pipe=None,
):
    """
    Generate evaluation images using FluxPipeline.

    Args:
        model_name: Name of the model (used for directory naming and weight loading)
        prompts_path: Path to CSV with columns: case_number, prompt, evaluation_seed
        save_path: Root directory for saved images
        device: CUDA device string
        guidance_scale: Classifier-free guidance scale
        image_size: Output image size
        ddim_steps: Number of inference steps
        num_samples: Number of images per prompt
        from_case: Skip prompts with case_number < from_case
        base_model_path: Path or HF ID for base Flux model
        model_dir: Directory containing fine-tuned model weights
        max_prompts: Limit number of prompts (None = all)
        batch_size: Number of images to generate in parallel per forward pass.
                    For large-scale runs (e.g. COCO 30K), use 64 to maximise GPU util.
        pipe: Pre-loaded FluxPipeline (avoids reloading for multiple calls).
    """
    if pipe is None:
        pipe = _load_flux_pipeline(base_model_path, model_name, model_dir, device)

    # Read prompts
    df = pd.read_csv(prompts_path)
    if max_prompts is not None and len(df) > max_prompts:
        print(f"Limiting to first {max_prompts} prompts (out of {len(df)} total)")
        df = df.head(max_prompts)

    folder_path = os.path.join(save_path, model_name) if model_name else save_path
    os.makedirs(folder_path, exist_ok=True)

    print(f"Generating {num_samples} image(s) per prompt (batch_size={batch_size}), "
          f"saving to {folder_path}")

    # Build flat list of (case_number, prompt, seed, sample_idx) jobs
    jobs = []
    for _, row in df.iterrows():
        case_number = int(row.case_number)
        if case_number < from_case:
            continue
        prompt_text = str(row.prompt)
        seed = int(row.get("evaluation_seed", 42))
        for sample_idx in range(num_samples):
            img_path = os.path.join(folder_path, f"{case_number}_{sample_idx}.png")
            if os.path.exists(img_path):
                continue  # skip already-generated
            jobs.append((case_number, prompt_text, seed, sample_idx))

    if not jobs:
        print("All images already exist, skipping generation.")
        return folder_path

    # Process in batches
    for batch_start in tqdm(range(0, len(jobs), batch_size),
                            total=(len(jobs) + batch_size - 1) // batch_size,
                            desc="Generating (batched)"):
        batch_jobs = jobs[batch_start:batch_start + batch_size]
        prompts_batch = [j[1] for j in batch_jobs]
        generators = [
            torch.Generator(device="cpu").manual_seed(j[2] + j[3])
            for j in batch_jobs
        ]

        images = pipe(
            prompt=prompts_batch,
            height=image_size,
            width=image_size,
            num_inference_steps=ddim_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=256,
            generator=generators,
        ).images

        for img, (case_number, _, _, sample_idx) in zip(images, batch_jobs):
            img_path = os.path.join(folder_path, f"{case_number}_{sample_idx}.png")
            img.save(img_path)

    print(f"Generation complete. Images saved to {folder_path}")
    return folder_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with Flux")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--ddim_steps", type=int, default=28)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--from_case", type=int, default=0)
    parser.add_argument("--base_model_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images generated in parallel per forward pass")
    args = parser.parse_args()

    generate_images(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        from_case=args.from_case,
        base_model_path=args.base_model_path,
        model_dir=args.model_dir,
        max_prompts=args.max_prompts,
        batch_size=args.batch_size,
    )
