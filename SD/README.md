# Stable Diffusion Unlearning

Class forgetting (Imagenette) and NSFW concept removal for Stable Diffusion v1.4. Based on the [ESD](https://github.com/rohitgandikota/erasing/tree/main) codebase.

## Installation
* Clone [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and overlay this repo's files
* Download weights from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt) → `SD/models/ldm/stable-diffusion-v1/`
* Download diffusers UNet config from [here](https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json) → `SD/diffusers_unet_config.json`

## Pipeline (Recommended)

Single command: unlearn → generate images → evaluate (FID + classification + NudeNet) → wandb:

```bash
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"
conda env create -f environment.yaml && conda activate ldm

# Class forgetting (e.g., forget "tench")
python pipeline.py --config configs/pipeline_class.yaml

# NSFW concept removal
python pipeline.py --config configs/pipeline_nsfw.yaml
```

Key config fields:
- `paths.sd_ckpt` – path to `sd-v1-4-full-ema.ckpt`
- `paths.sd_config` – model config yaml
- `wandb.entity` – your wandb team/user
- `unlearn.class_to_forget` – Imagenette class index (0-9)

### wandb Sweeps

Sweeps run the pipeline with different hyperparameter combinations automatically.
The sweep YAML specifies which parameters to vary; the pipeline config provides
defaults for everything else. Dotted keys (e.g. `unlearn.lr`) map to nested
fields in the pipeline YAML.

```bash
# Class forgetting sweep
./run_sweep.sh sweep_class

# NSFW removal sweep
./run_sweep.sh sweep_nsfw
```

To add a new sweep parameter, copy any dotted config key into the sweep's
`parameters:` block with `values:` (grid) or `min:`/`max:` (random).

On SLURM, launch each agent in its own job:
```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --mem=48G --time=48:00:00 --array=0-9
source activate ldm
cd /path/to/InTAct-Unl/SD
export PYTHONPATH="${PYTHONPATH}:/path/to/InTAct-Unl"
wandb agent <sweep-id>
```

### SLURM

```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --mem=48G --time=48:00:00
source activate ldm
cd /path/to/InTAct-Unl/SD
export PYTHONPATH="${PYTHONPATH}:/path/to/InTAct-Unl"
python pipeline.py --config configs/pipeline_class.yaml
```

---

<details>
<summary>Manual workflow (original)</summary>

## Unlearned Weights
The unlearned weights for NSFW and object forgetting are available [here](https://drive.google.com/drive/folders/1fOx-v_ru3NfB2rPe5LGxaQS-Q17QzKzp?usp=sharing).

# Forgetting Training with Saliency-Unlearning
1. First, we need to generate saliency map for unlearning.

   ```
    python train-scripts/generate_mask.py --ckpt_path 'models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt' --classes {label} --device '0'
   ```

   This will save saliency map in `SD/mask/{label}`.

2. Forgetting training with Saliency-Unlearning

   ```
    python train-scripts/random_label.py --train_method full --alpha 0.5 --lr 1e-5 --epochs 5  --class_to_forget {label} --mask_path 'mask/{label}/with_0.5.pt' --device '0'
   ```

   This should create another folder in `SD/model`. 

   You can experiment with forgetting different class labels using the `--class_to_forget` flag, but we will consider forgetting the 0 (tench) class here.

3. Forgetting training with ESD

    Edit `train-script/train-esd.py` and change the default argparser values according to your convenience (especially the config paths)
    To choose train_method, pick from following `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'` 
    ```
    python train-scripts/train-esd.py --prompt 'your prompt' --train_method 'your choice of training' --devices '0,1'
    ```

# Generating Images
  1. To use `eval-scripts/generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`)
  2. To generate multiple images per prompt use the argument `num_samples`. It is default to 10.
  3. The path to model can be customised in the script.
  4. It is to be noted that the current version requires the model to be in saved in `SD/model/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt`
        ```
        python eval-scripts/generate-images.py --prompts_path 'prompts/imagenette.csv' --save_path 'evaluation_folder/ --model_name {model} --device 'cuda:0'
        ``` 

# Evaluation
1. FID
   * First,we need to select some images from Imagenette as real images.
   * Then, we can compute FID between real images and generated images. 
        ```
        python eval-scripts/compute-fid.py --folder_path {images_path}
        ```

2. Accuracy
   ```
   python eval-scripts/imageclassify.py --prompts_path 'prompts/imagenette.csv' --folder_path {images_path}
   ```


# NSFW-concept removal with Saliency-Unlearning
1. To remove NSFW-concept, we initially utilize SD V1.4 to generate 800 images as Df with the prompt "a photo of a nude person" and store them in "SD/data/nsfw". Additionally, we generate another 800 images designated as Dr using the prompt "a photo of a person wearing clothes" and store them in "SD/data/not-nsfw".


2. Next, we need to generate saliency map for NSFW-concept.

   ```
   python train-scripts/generate_mask.py --ckpt_path 'models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt' --nsfw True --device '0'
   ```

   This will save saliency map in `SD/mask`.

3. Forgetting training with Saliency-Unlearning

   ```
   python train-scripts/nsfw_removal.py --train_method 'full' --mask_path 'mask/nude_0.5.pt' --device '0'
   ```

# InTAct Unlearning

InTAct (Interval Protection) unlearning adds protection loss to preserve model performance on retain data while unlearning forget concepts. It is composable with all existing base methods (GA, RL, NSFW, ESD).

## Usage

InTAct targets cross-attention Q/K/V projections (`to_q`, `to_k`, `to_v`) by default. You can customize targets via `--targets`.

1. **InTAct + Gradient Ascent (class forgetting)**:
   ```bash
   python train-scripts/intact_unlearn.py \
       --base_method ga \
       --class_to_forget 0 \
       --train_method xattn \
       --lambda_interval 1.0 \
       --epochs 5 \
       --device 0
   ```

2. **InTAct + Random Label (class forgetting)**:
   ```bash
   python train-scripts/intact_unlearn.py \
       --base_method rl \
       --class_to_forget 0 \
       --train_method xattn \
       --lambda_interval 1.0 \
       --epochs 5 \
       --device 0
   ```

3. **InTAct + NSFW removal**:
   ```bash
   python train-scripts/intact_unlearn.py \
       --base_method nsfw \
       --train_method xattn \
       --lambda_interval 1.0 \
       --device 0
   ```

4. **InTAct + ESD (prompt-based)**:
   ```bash
   python train-scripts/intact_unlearn.py \
       --base_method esd \
       --prompt "nudity" \
       --train_method xattn \
       --lambda_interval 1.0 \
       --iterations 1000 \
       --devices 0,0
   ```

## InTAct Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--targets` | `["to_q", "to_k", "to_v"]` | Target layer patterns for protection (cross-attention QKV) |
| `--lambda_interval` | `1.0` | Weight for InTAct protection loss |
| `--lower_percentile` | `0.05` | Lower bound for activation safe zone |
| `--upper_percentile` | `0.95` | Upper bound for activation safe zone |
| `--reduced_dim` | `32` | SVD dimension for efficiency |
| `--infinity_scale` | `20.0` | Scale for infinity bounds |
| `--use_actual_bounds` | `False` | Use actual min/max from remain+forget data |
| `--normalize_protection` | `True` | Normalize protection loss by number of layers |

## Config File

A sample config is provided at `configs/stable-diffusion/v1-intact.yaml`.

</details>