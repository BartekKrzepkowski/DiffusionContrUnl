import argparse
import os

import pandas as pd
import torch
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer


def generate_images(
    model_name,
    prompts_path,
    save_path,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=512,
    ddim_steps=100,
    num_samples=10,
    from_case=0,
    base_model_path="CompVis/stable-diffusion-v1-4",
    base_config_path=None,
    model_dir="models",
    max_prompts=None,
    n_outer=1,
):
    """
    Function to generate images from diffusers code

    The program requires the prompts to be in a csv format with headers
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)

    Parameters
    ----------
    model_name : str
        name of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.

    """

    # Load base model components
    print(f"Loading models from base: {base_model_path}")
    
    # Check if base_model_path is a checkpoint file
    if base_model_path.endswith('.ckpt'):
        print("Loading from checkpoint format...")
        if base_config_path is None:
            raise ValueError("base_config_path required when loading from .ckpt")
        
        # Need to convert checkpoint to diffusers format
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'train-scripts'))
        from convertModels import (
            create_unet_diffusers_config,
            create_vae_diffusers_config,
            convert_ldm_unet_checkpoint,
            convert_ldm_vae_checkpoint,
        )
        from omegaconf import OmegaConf
        
        checkpoint = torch.load(base_model_path, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        
        original_config = OmegaConf.load(base_config_path)
        
        # Create and load VAE
        vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
        vae = AutoencoderKL(**vae_config)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)
        vae.load_state_dict(converted_vae_checkpoint)
        
        # Create and load UNet
        unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
        unet_config["upcast_attention"] = False
        unet = UNet2DConditionModel(**unet_config)
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config)
        unet.load_state_dict(converted_unet_checkpoint)
        
        # Extract text encoder from checkpoint
        from convertModels import convert_ldm_clip_checkpoint
        print("Extracting CLIP text encoder from checkpoint...")
        text_encoder = convert_ldm_clip_checkpoint(checkpoint)
        
        # Use tokenizer from CompVis SD (should be cached or available locally)
        print("Loading tokenizer...")
        try:
            # Try from local cache first
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
        except:
            print("Tokenizer not in cache, downloading (may be slow)...")
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        print("Successfully loaded from checkpoint")
    else:
        # Load from diffusers format directory or HuggingFace
        vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder")
        unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    
    # Load fine-tuned UNet weights if model_name specified
    # Skip loading if model_name is exactly "SD" or empty
    if model_name:
        try:
            # Check if model_name is an absolute path to a .pt file
            if os.path.isabs(model_name) and os.path.exists(model_name):
                model_path = model_name
            elif os.path.exists(model_name):
                # Relative path that exists
                model_path = model_name
            else:
                # Try loading from model_dir directory (configurable, defaults to "models")
                model_path = f'{model_dir}/{model_name}/{model_name.replace("compvis","diffusers")}.pt'
            
            print(f"Loading fine-tuned UNet weights from: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Fine-tuned UNet not found at: {model_path}\n"
                    f"  model_name={model_name}, model_dir={model_dir}\n"
                    f"  Check that the unlearning step saved the model to the expected location."
                )
            unet.load_state_dict(torch.load(model_path, map_location="cpu"))
            print("Successfully loaded fine-tuned UNet")
        except FileNotFoundError:
            raise  # Don't swallow missing-model errors
        except Exception as e:
            print(f"Could not load fine-tuned UNet: {e}")
            print("Using base UNet instead")
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    df = pd.read_csv(prompts_path)
    
    # Limit to max_prompts if specified
    if max_prompts is not None and len(df) > max_prompts:
        print(f"Limiting to first {max_prompts} prompts (out of {len(df)} total)")
        df = df.head(max_prompts)

    folder_path = f"{save_path}/{model_name}"
    os.makedirs(folder_path, exist_ok=True)

    for _, row in df.iterrows():
        prompt = [str(row.prompt)] * num_samples
        seed = row.evaluation_seed
        case_number = row.case_number
        if case_number < from_case:
            continue

        height = image_size
        width = image_size
        num_inference_steps = ddim_steps
        guidance_scale = guidance_scale
        generator = torch.manual_seed(seed)
        batch_size = len(prompt)

        for i in range(n_outer):
            # Check if all images for this prompt already exist
            all_exist = True
            for num in range(batch_size):
                img_path = f"{folder_path}/{case_number}_{i * batch_size + num}.png"
                if not os.path.exists(img_path):
                    all_exist = False
                    break
            if all_exist:
                print(f"Skipping case_number {case_number} batch {i}: images already exist.")
                continue

            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(torch_device)
            scheduler.set_timesteps(num_inference_steps)
            latents = latents * scheduler.init_noise_sigma
            from tqdm.auto import tqdm
            scheduler.set_timesteps(num_inference_steps)
            for t in tqdm(scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, timestep=t
                )
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            for num, im in enumerate(pil_images):
                img_path = f"{folder_path}/{case_number}_{i * batch_size + num}.png"
                if not os.path.exists(img_path):
                    im.save(img_path)
                    print(f"Saved {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--model_name", help="name of model", type=str, required=True)
    parser.add_argument(
        "--prompts_path", help="path to csv file with prompts", type=str, required=True
    )
    parser.add_argument(
        "--save_path", help="folder where to save images", type=str, required=True
    )
    parser.add_argument(
        "--device",
        help="cuda device to run on",
        type=str,
        required=False,
        default="cuda:0",
    )
    parser.add_argument(
        "--guidance_scale",
        help="guidance to run eval",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--from_case",
        help="continue generating from case_number",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--num_samples",
        help="number of samples per prompt",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--base_model_path",
        help="path to base model checkpoint (.ckpt) or diffusers format directory",
        type=str,
        required=False,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--base_config_path",
        help="path to base model config (required if base_model_path is .ckpt)",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--model_dir",
        help="directory where model checkpoints are saved",
        type=str,
        required=False,
        default="models",
    )
    args = parser.parse_args()

    model_name = args.model_name
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples = args.num_samples
    from_case = args.from_case
    base_model_path = args.base_model_path
    base_config_path = args.base_config_path

    generate_images(
        model_name,
        prompts_path,
        save_path,
        device=device,
        guidance_scale=guidance_scale,
        image_size=image_size,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        from_case=from_case,
        base_model_path=base_model_path,
        base_config_path=base_config_path,
        model_dir=args.model_dir,
    )