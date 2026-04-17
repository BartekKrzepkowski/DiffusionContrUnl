"""
Unified Pipeline for DDPM Unlearning Experiments.

Handles class-conditional DDPM on CIFAR-10 / STL-10.
Steps: unlearn → sample → evaluate (FID + classifier) → log to wandb.

Usage:
    cd DDPM
    python pipeline.py --config configs/pipeline.yaml

    # wandb sweep:
    wandb sweep configs/sweep.yaml
    wandb agent <sweep-id>
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import setup_cache  # noqa: E402  — must precede torch / HF imports

import argparse
import logging
import pathlib
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from functions import dict2namespace, create_class_labels
from runners.diffusion import Diffusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# Config helpers
# =============================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_wandb_config(cfg):
    import wandb
    for key, val in dict(wandb.config).items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def build_runner_config(cfg):
    """Build a Namespace config for the Diffusion runner from pipeline YAML."""
    model_cfg = cfg["model_config"]
    with open(model_cfg, "r") as f:
        runner_cfg = yaml.safe_load(f)

    # Override training params from pipeline config
    uc = cfg.get("unlearn", {})
    if "training" not in runner_cfg:
        runner_cfg["training"] = {}
    runner_cfg["training"]["batch_size"] = uc.get("batch_size", runner_cfg["training"].get("batch_size", 128))
    runner_cfg["training"]["n_iters"] = uc.get("n_iters", runner_cfg["training"].get("n_iters", 1500))

    # InTAct params
    ic = cfg.get("intact", {})
    if ic:
        for k, v in ic.items():
            runner_cfg["training"][k] = v

    # Optim
    if "optim" not in runner_cfg:
        runner_cfg["optim"] = {}
    runner_cfg["optim"]["lr"] = uc.get("lr", runner_cfg["optim"].get("lr", 1e-4))

    config = dict2namespace(runner_cfg)

    # Setup dirs under pipeline output
    output_dir = cfg["paths"].get("output_dir", "./results/pipeline")
    checkpoint_dir = cfg["paths"].get("checkpoint_dir", None)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join(output_dir, timestamp)
    config.log_dir = os.path.join(config.exp_root_dir, "logs")
    
    # Use checkpoint_dir if specified, otherwise default to exp_root_dir/ckpts
    if checkpoint_dir:
        config.ckpt_dir = os.path.join(checkpoint_dir, timestamp)
    else:
        config.ckpt_dir = os.path.join(config.exp_root_dir, "ckpts")
    
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.ckpt_dir, exist_ok=True)

    return config


def build_runner_args(cfg, runner_config):
    """Build an argparse.Namespace for the Diffusion runner."""
    uc = cfg.get("unlearn", {})
    args = argparse.Namespace()
    args.config = cfg["model_config"]
    args.ckpt_folder = cfg["paths"]["pretrained_ckpt_folder"]
    args.mode = uc.get("mode", "intact")
    args.label_to_forget = uc.get("label_to_forget", 0)
    args.seed = cfg["pipeline"].get("seed", 1234)
    args.sample_type = cfg.get("sample_type", "generalized")
    args.skip_type = cfg.get("skip_type", "uniform")
    args.timesteps = cfg.get("timesteps", 1000)
    args.eta = cfg.get("eta", 1.0)
    args.cond_scale = cfg.get("cond_scale", 2.0)
    args.sequence = False
    args.alpha = uc.get("alpha", 1.0)
    args.mask_path = None
    args.method = uc.get("method", "ga")
    args.uc = True
    args.negative_guidance = uc.get("negative_guidance", 7.5)
    args.mask_ratio = 0.5
    args.sparse = False

    # Sampling args
    args.n_samples_per_class = cfg.get("evaluate", {}).get("n_samples_per_class", 500)
    args.classes_to_generate = None  # set per-call

    return args


# =============================================================================
# FID evaluation – uses the same TF-based evaluator as the reference code
# =============================================================================

def compute_fid_reference(ref_dir, sample_dir):
    """
    Compute FID (and IS, sFID, Precision, Recall) using the exact same
    TensorFlow Inception V3 evaluator from evaluator.py.

    Returns a dict with keys: FID, IS, sFID, Precision, Recall.
    """
    import cv2
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator, read_images_folder

    ref_arr = read_images_folder(ref_dir)
    sample_arr = read_images_folder(sample_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    evaluator = Evaluator(sess)
    evaluator.warmup()

    ref_acts = evaluator.read_activations(ref_arr)
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_acts)

    sample_acts = evaluator.read_activations(sample_arr)
    sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_acts)

    inception_score = evaluator.compute_inception_score(sample_acts[0])
    fid = sample_stats.frechet_distance(ref_stats)
    sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

    sess.close()

    return {
        "fid": float(fid),
        "inception_score": float(inception_score),
        "sfid": float(sfid),
        "precision": float(prec),
        "recall": float(recall),
    }


# =============================================================================
# Classifier evaluation (reuses existing logic)
# =============================================================================

class ImagePathDataset(torch.utils.data.Dataset):
    IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

    def __init__(self, img_folder, transform=None, n=None):
        self.transform = transform
        path = pathlib.Path(img_folder)
        self.files = sorted([
            f for ext in self.IMAGE_EXTENSIONS for f in path.glob(f"*.{ext}")
        ])
        if n is not None:
            self.files = self.files[:n]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def classifier_eval(sample_path, dataset, label_of_forgotten_class, classifier_ckpt, device):
    """Evaluate generated samples of forgotten class with a ResNet-34 classifier.
    Returns dict with classifier/acc_forgotten (lower = better forgetting)."""
    model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = ImagePathDataset(sample_path, transform=transform)
    loader = DataLoader(ds, batch_size=64)

    n = len(ds)
    if n == 0:
        return {}

    entropy_sum = 0.0
    prob_sum = 0.0
    acc_sum = 0.0

    with torch.no_grad():
        for data in loader:
            logits = model(data.to(device))
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-30)
            ent = -torch.sum(probs * log_probs, dim=1)
            entropy_sum += ent.sum().item()
            prob_sum += probs[:, label_of_forgotten_class].sum().item()
            acc_sum += (torch.argmax(logits, dim=-1) == label_of_forgotten_class).sum().item()

    return {
        "classifier/entropy": entropy_sum / n,
        "classifier/prob_forgotten": prob_sum / n,
        "classifier/acc_forgotten": acc_sum / n,
    }


def classifier_eval_remaining(class_samples_dir, label_to_forget, n_classes, classifier_ckpt, device):
    """
    Evaluate generated samples of remaining classes with a ResNet-34 classifier.
    Returns TA = average per-class accuracy across remaining classes.
    """
    model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    per_class_acc = {}
    for cls_idx in range(n_classes):
        if cls_idx == label_to_forget:
            continue
        cls_dir = os.path.join(class_samples_dir, str(cls_idx))
        if not os.path.isdir(cls_dir):
            continue
        ds = ImagePathDataset(cls_dir, transform=transform)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=64)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                logits = model(data.to(device))
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == cls_idx).sum().item()
                total += data.size(0)

        per_class_acc[cls_idx] = correct / max(total, 1)

    if not per_class_acc:
        return None, {}

    ta = np.mean(list(per_class_acc.values()))
    detail = {f"classifier/acc_class_{k}": v for k, v in per_class_acc.items()}
    return ta, detail


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    import wandb

    parser = argparse.ArgumentParser(description="DDPM Unlearning Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline YAML config")
    cli = parser.parse_args()

    cfg = load_config(cli.config)

    # --- wandb ---
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        group=cfg["wandb"].get("group"),
        tags=cfg["wandb"].get("tags", []),
        config=cfg,
    )
    cfg = merge_wandb_config(cfg)

    seed = cfg["pipeline"].get("seed", 1234)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    runner_config = build_runner_config(cfg)
    runner_args = build_runner_args(cfg, runner_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_to_forget = cfg["unlearn"].get("label_to_forget", 0)
    dataset_name = cfg["data"]["dataset"].lower()
    eval_cfg = cfg.get("evaluate", {})
    metrics = {}

    # =========================================================================
    # Step 1: Unlearn
    # =========================================================================
    mode = cfg["unlearn"].get("mode", "intact")
    log.info(f"Step 1: Unlearning  mode={mode}  label_to_forget={label_to_forget}")

    runner = Diffusion(runner_args, runner_config)

    if mode == "intact":
        runner.intact_unlearn()
    elif mode == "saliency_unlearn":
        runner.saliency_unlearn()
    elif mode == "forget":
        runner.train_forget()
    elif mode == "retrain":
        runner.retrain()
    elif mode == "train_esd":
        runner.train_esd()
    else:
        raise ValueError(f"Unknown unlearn mode: {mode}")

    # The runner saves checkpoints to runner_config.ckpt_dir
    # Update ckpt_folder to point to the new run for sampling
    unlearn_output = runner_config.exp_root_dir
    runner_args.ckpt_folder = unlearn_output
    log.info(f"Unlearning complete. Output at {unlearn_output}")

    # =========================================================================
    # Step 2: Sample images
    # =========================================================================
    n_samples = eval_cfg.get("n_samples_per_class", 500)
    cond_scale = cfg.get("cond_scale", 2.0)
    n_classes = cfg["data"].get("n_classes", 10)

    # 2a: Sample ALL classes (for classifier eval on forgotten + remaining, and wandb images)
    if eval_cfg.get("classifier", {}).get("enabled", True):
        log.info(f"Step 2a: Sampling {n_samples} images per class (all {n_classes} classes)")
        all_class_labels = ",".join(str(i) for i in range(n_classes))
        runner_args.classes_to_generate = all_class_labels
        runner_args.n_samples_per_class = n_samples
        runner_args.mode = "sample_classes"
        sample_runner = Diffusion(runner_args, runner_config)
        sample_runner.sample()
        class_samples_dir = os.path.join(unlearn_output, "class_samples")
        forget_sample_dir = os.path.join(class_samples_dir, str(label_to_forget))

    # 2b: Sample remaining classes (for FID)
    if eval_cfg.get("fid", {}).get("enabled", True):
        fid_n_samples = eval_cfg.get("fid", {}).get("n_samples_per_class", 500)
        log.info(f"Step 2b: Sampling {fid_n_samples} images per remaining class for FID")
        runner_args.classes_to_generate = f"x{label_to_forget}"
        runner_args.n_samples_per_class = fid_n_samples
        runner_args.mode = "sample_fid"
        fid_runner = Diffusion(runner_args, runner_config)
        fid_runner.sample()
        fid_sample_dir = os.path.join(
            unlearn_output,
            f"fid_samples_guidance_{cond_scale}_excluded_class_{label_to_forget}",
        )
        # fallback name used by some versions
        if not os.path.exists(fid_sample_dir):
            fid_sample_dir = os.path.join(
                unlearn_output,
                f"fid_samples_without_label_{label_to_forget}_guidance_{cond_scale}",
            )

    # =========================================================================
    # Step 3: Evaluate
    # =========================================================================
    log.info("Step 3: Evaluation")

    clf_ckpt = cfg["paths"].get("classifier_ckpt", f"{dataset_name}_resnet34.pth")

    # 3a: FID (+ IS, sFID, Precision, Recall) — remaining classes only
    if eval_cfg.get("fid", {}).get("enabled", True):
        ref_dir = cfg["paths"].get("ref_dataset_dir")
        if ref_dir is None:
            ref_dir = f"{dataset_name}_without_label_{label_to_forget}"
        if os.path.exists(ref_dir) and os.path.exists(fid_sample_dir):
            log.info(f"Computing FID (TF evaluator): {ref_dir} vs {fid_sample_dir}")
            fid_metrics = compute_fid_reference(ref_dir, fid_sample_dir)
            metrics["FID"] = fid_metrics.pop("fid")
            metrics.update(fid_metrics)  # IS, sFID, precision, recall
            for k, v in {**{"FID": metrics["FID"]}, **fid_metrics}.items():
                log.info(f"  {k} = {v:.4f}")
        else:
            log.warning(f"Skipping FID: ref_dir={ref_dir} exists={os.path.exists(ref_dir)}"
                        f"  samples exist={os.path.exists(fid_sample_dir)}")
            log.warning("Run save_base_dataset.py first to create reference images.")

    # 3b: UA — Unlearning Accuracy (1 - acc on forgotten class)
    if eval_cfg.get("classifier", {}).get("enabled", True):
        if os.path.exists(clf_ckpt) and os.path.exists(forget_sample_dir):
            log.info(f"Classifier eval (UA) on forget class {label_to_forget}")
            clf_metrics = classifier_eval(
                forget_sample_dir, dataset_name, label_to_forget, clf_ckpt, device,
            )
            metrics.update(clf_metrics)
            ua = 1.0 - clf_metrics.get("classifier/acc_forgotten", 0.0)
            metrics["UA"] = ua
            log.info(f"  UA = {ua:.4f}")
            for k, v in clf_metrics.items():
                log.info(f"  {k} = {v:.4f}")
        else:
            log.warning(f"Skipping UA: ckpt={clf_ckpt} exist={os.path.exists(clf_ckpt)}")

    # 3c: TA — Testing Accuracy (avg accuracy on remaining classes)
    if eval_cfg.get("classifier", {}).get("enabled", True):
        if os.path.exists(clf_ckpt) and os.path.exists(class_samples_dir):
            log.info(f"Classifier eval (TA) on remaining classes")
            ta, ta_detail = classifier_eval_remaining(
                class_samples_dir, label_to_forget, n_classes, clf_ckpt, device,
            )
            if ta is not None:
                metrics["TA"] = ta
                metrics.update(ta_detail)
                log.info(f"  TA = {ta:.4f}")
        else:
            log.warning(f"Skipping TA: class_samples_dir missing")

    # =========================================================================
    # Step 4: Log to wandb
    # =========================================================================
    log.info("Step 4: Logging to wandb")
    wandb.log(metrics)
    wandb.summary.update(metrics)

    # Log sample images — one panel per class (including forgotten)
    n_sample_imgs = eval_cfg.get("n_sample_images_per_class", 8)
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
    class_names = cifar10_classes if n_classes == 10 else [str(i) for i in range(n_classes)]

    for cls_idx in range(n_classes):
        cls_dir = os.path.join(unlearn_output, "class_samples", str(cls_idx))
        if os.path.isdir(cls_dir):
            imgs = sorted(pathlib.Path(cls_dir).glob("*.png"))[:n_sample_imgs]
            if imgs:
                cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                label = f"(FORGET) {cls_name}" if cls_idx == label_to_forget else cls_name
                wandb.log({
                    f"samples/{cls_idx}_{cls_name}": [
                        wandb.Image(str(p), caption=label) for p in imgs
                    ]
                })

    # Log model checkpoint as artifact
    ckpt_file = os.path.join(runner_config.ckpt_dir, "ckpt.pth")
    if os.path.exists(ckpt_file):
        art = wandb.Artifact(
            name=f"ddpm-{dataset_name}-forget{label_to_forget}-{wandb.run.id}",
            type="model",
            metadata=metrics,
        )
        art.add_file(ckpt_file)
        wandb.log_artifact(art)
        log.info("Model checkpoint logged as wandb artifact")

    wandb.finish()
    log.info(f"Pipeline complete. Results: {metrics}")


if __name__ == "__main__":
    main()
