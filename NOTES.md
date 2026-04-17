## Prerequisites

### 1. Install Dependencies

```bash
# Classification
cd Classification
pip install -r requirements.txt

# DDPM
cd ../DDPM
conda create -n salun-ddpm python=3.8
conda activate salun-ddpm
pip install -r requirements.txt

# SD
cd ../SD
conda env create -f environment.yaml
conda activate ldm
```

### 2. Prepare Data & Models

**Classification (CIFAR-10):**
```bash
cd Classification

# Train a pretrained model first (or download one)
python main_train.py --arch resnet18 --dataset cifar10 --lr 0.1 --epochs 182 --save_dir ./checkpoints

# Update configs to point to the trained model
# Edit configs/pipeline_classwise.yaml and configs/pipeline_random.yaml:
#   paths.model: "./checkpoints/resnet18_cifar10.pth"
```

**DDPM (CIFAR-10):**
```bash
cd DDPM

# 1. Train base DDPM model
CUDA_VISIBLE_DEVICES="0" python train.py --config configs/cifar10_train.yml --mode train

# This saves checkpoint to results/cifar10/YYYY_MM_DD_HHMMSS/
# Note the checkpoint folder path

# 2. Create reference dataset (for FID)
python save_base_dataset.py --dataset cifar10 --label_to_forget 0

# This creates cifar10_without_label_0/ folder

# 3. Train classifier (for evaluation)
CUDA_VISIBLE_DEVICES="0" python train_classifier.py --dataset cifar10

# This saves cifar10_resnet34.pth

# 4. Update config with paths
# Edit configs/pipeline.yaml:
#   paths.pretrained_ckpt_folder: "results/cifar10/YYYY_MM_DD_HHMMSS"
#   paths.ref_dataset_dir: "cifar10_without_label_0"
#   paths.classifier_ckpt: "cifar10_resnet34.pth"
```

**SD (Stable Diffusion):**
```bash
cd SD

# 1. Download SD v1.4 weights
mkdir -p models/ldm/stable-diffusion-v1
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt \
  -O models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt

# 2. Download diffusers config
wget https://huggingface.co/CompVis/stable-diffusion-v1-4/raw/main/unet/config.json \
  -O diffusers_unet_config.json

# 3. For class forgetting: create reference Imagenette dataset
# This is auto-downloaded by HuggingFace datasets on first run
# Just ensure imagenette_without_label_6/ folder exists or will be created

# 4. For NSFW: prepare NSFW/not-NSFW images (see SD/README.md)
# Generate 800 images with SD v1.4 using:
#   - "a photo of a nude person" → data/nsfw/
#   - "a photo of a person wearing clothes" → data/not-nsfw/
#
# When using the Flux pipeline you can now point at the same two folders
# (or HF dataset ids) by setting `paths.nsfw_data` and
# `paths.not_nsfw_data` in `Flux/configs/intact/pipeline_nsfw.yaml` or
# `pipeline_concept.yaml`.  The base_method should be "rl" for concept
# erasure; real images are only used for InTAct boundary computation.

# Configs already have default paths, but verify:
# Edit configs/pipeline_class.yaml and configs/pipeline_nsfw.yaml:
#   paths.sd_ckpt: "models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"
```

### 3. Set wandb Entity

Edit ALL config files to set your wandb entity:

```bash
# Replace "entity: null" with "entity: your-wandb-username" in:
- Classification/configs/pipeline_classwise.yaml
- Classification/configs/pipeline_random.yaml
- DDPM/configs/pipeline.yaml
- SD/configs/pipeline_class.yaml
- SD/configs/pipeline_nsfw.yaml
```

Or set environment variable:
```bash
export WANDB_ENTITY="your-wandb-username"
```

## Testing Commands

### Test 1: Classification - Class-wise Forgetting

```bash
cd Classification
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"

# Quick test (reduce epochs for speed)
python pipeline.py --config configs/pipeline_classwise.yaml

# Check wandb for:
# - acc/forget (should be low)
# - acc/retain (should be high)
# - acc/test (should be high)
# - UA (Unlearning Accuracy = 100 - forget_acc)
# - MIA metrics
```

**Expected output:**
- Logs show unlearning progress + protection loss
- Final metrics logged to wandb
- Model checkpoint saved as artifact

### Test 2: Classification - Random Data Forgetting

```bash
cd Classification

python pipeline.py --config configs/pipeline_random.yaml

# Verify same metrics as Test 1
```

### Test 3: DDPM - Class Forgetting

```bash
cd DDPM
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"

# This will take longer (unlearning + sampling + FID)
python pipeline.py --config configs/pipeline.yaml

# Check wandb for:
# - fid (lower is better for remaining classes)
# - inception_score
# - sfid, precision, recall
# - classifier/entropy (higher = more confused about forgotten class)
# - Sample images of forgotten class
```

**Expected output:**
- Checkpoint folder in unlearn_output/
- Generated images in class_samples/ and fid_samples/
- FID computed via TensorFlow evaluator
- Classifier metrics on forgotten class samples

### Test 4: SD - Class Forgetting (Imagenette)

```bash
cd SD
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"
conda activate ldm

# This takes longest (unlearning + image generation + evaluation)
python pipeline.py --config configs/pipeline_class.yaml

# Check wandb for:
# - fid (for forgotten class)
# - classify/accuracy
# - Sample generated images
```

**Expected output:**
- Model saved in models/{model_name}/
- Generated images in evaluation/generated/{model_name}/
- FID score (torchmetrics)
- Sample images logged to wandb

### Test 5: SD - NSFW Removal

```bash
cd SD

python pipeline.py --config configs/pipeline_nsfw.yaml

# Check wandb for:
# - nudenet/nude_ratio (should be low)
# - nudenet/total_images
# - classify/accuracy (general quality maintained)
```