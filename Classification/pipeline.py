"""
Unified Pipeline for Classification Unlearning Experiments.

Handles both class-wise and random data forgetting.
Integrates with wandb for logging, artifacts, and sweep support.

Usage:
    python pipeline.py --config configs/pipeline_classwise.yaml
    python pipeline.py --config configs/pipeline_random.yaml

    # wandb sweep:
    wandb sweep configs/sweep_classwise.yaml
    wandb agent <sweep-id>
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import setup_cache  # noqa: E402  — must precede torch / HF imports

import argparse
import copy
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add project root to path for InTAct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import evaluation
import unlearn
import utils
from trainer import validate

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


def merge_wandb_config(config):
    """Apply wandb sweep overrides (dot-notation) into nested config dict."""
    import wandb

    for key, val in dict(wandb.config).items():
        parts = key.split(".")
        d = config
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return config


def build_args(cfg):
    """Build an argparse.Namespace compatible with existing Classification code."""
    args = argparse.Namespace()

    # Dataset
    args.data = cfg["paths"].get("data_dir", "../data")
    args.dataset = cfg["data"]["dataset"]
    args.input_size = cfg["data"].get("input_size", 32)
    args.data_dir = cfg["paths"].get("data_dir", "../data")
    args.num_workers = cfg["data"].get("num_workers", 4)
    args.num_classes = cfg["data"].get("num_classes", 10)

    # Architecture
    args.arch = cfg["model"]["arch"]
    args.imagenet_arch = cfg["model"].get("imagenet_arch", False)
    args.train_y_file = cfg.get("train_y_file", "./labels/train_ys.pth")
    args.val_y_file = cfg.get("val_y_file", "./labels/val_ys.pth")

    # General
    args.seed = cfg["pipeline"].get("seed", 2)
    args.train_seed = cfg["pipeline"].get("train_seed", 1)
    args.gpu = cfg["pipeline"].get("gpu", 0)
    args.workers = cfg["data"].get("num_workers", 4)
    args.resume = False
    args.checkpoint = None
    args.save_dir = cfg["paths"].get("checkpoint_dir", cfg["paths"].get("output_dir", "./results/pipeline"))
    args.model_path = cfg["paths"]["pretrained_ckpt"]

    # Training / Unlearning
    args.batch_size = cfg["unlearn"].get("batch_size", 256)
    args.lr = cfg["unlearn"].get("lr", 0.1)
    args.momentum = cfg["unlearn"].get("momentum", 0.9)
    args.weight_decay = cfg["unlearn"].get("weight_decay", 5e-4)
    args.epochs = cfg["unlearn"].get("retrain_epochs", 182)
    args.warmup = cfg["unlearn"].get("warmup", 0)
    args.print_freq = cfg.get("print_freq", 50)
    args.decreasing_lr = cfg["unlearn"].get("decreasing_lr", "91,136")
    args.no_aug = cfg.get("no_aug", False)
    args.no_l1_epochs = 0
    args.smoothing = 0.0
    args.schedule = cfg["unlearn"].get("schedule", "cosine")

    # Pruning (compat defaults – not used but needed by some code paths)
    args.prune = cfg.get("prune", "omp")
    args.pruning_times = 1
    args.rate = cfg.get("rate", 0.95)
    args.prune_type = cfg.get("prune_type", "rewind_lt")
    args.random_prune = False
    args.rewind_epoch = cfg.get("rewind_epoch", 0)
    args.rewind_pth = None

    # Unlearn specifics
    args.unlearn = cfg["unlearn"]["method"]
    args.unlearn_lr = cfg["unlearn"].get("unlearn_lr", 0.01)
    args.unlearn_epochs = cfg["unlearn"].get("unlearn_epochs", 10)
    args.alpha = cfg["unlearn"].get("alpha", 0.2)
    args.mask_path = None  # no saliency masks

    # Forget target
    setting = cfg["pipeline"]["setting"]
    if setting == "classifier_classwise":
        args.class_to_replace = cfg["unlearn"].get("class_to_forget", 0)
        args.num_indexes_to_replace = None
        args.indexes_to_replace = None
    else:  # classifier_random
        args.class_to_replace = None
        num_to_replace = cfg["unlearn"].get("num_indexes_to_replace", 4500)
        args.num_indexes_to_replace = num_to_replace
        # Generate random indexes for forgetting (45000 train samples after val split)
        rng = np.random.RandomState(args.seed)
        args.indexes_to_replace = rng.choice(45000, size=num_to_replace, replace=False)

    return args


# =============================================================================
# Data loaders
# =============================================================================

def build_data_loaders(args, marked_loader, val_loader, test_loader):
    """Split marked_loader into forget/retain and return all loaders."""
    seed = args.seed

    def replace_loader_dataset(dataset, batch_size=args.batch_size, shuffle=True):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0,
            pin_memory=True, shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    try:
        marked = forget_dataset.targets < 0
        forget_dataset.data = forget_dataset.data[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1
    except Exception:
        marked = forget_dataset.targets < 0
        forget_dataset.imgs = forget_dataset.imgs[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1

    forget_loader = replace_loader_dataset(forget_dataset, shuffle=True)

    retain_dataset = copy.deepcopy(marked_loader.dataset)
    try:
        marked = retain_dataset.targets >= 0
        retain_dataset.data = retain_dataset.data[marked]
        retain_dataset.targets = retain_dataset.targets[marked]
    except Exception:
        marked = retain_dataset.targets >= 0
        retain_dataset.imgs = retain_dataset.imgs[marked]
        retain_dataset.targets = retain_dataset.targets[marked]

    retain_loader = replace_loader_dataset(retain_dataset, shuffle=True)
    log.info(f"Forget: {len(forget_dataset)}  Retain: {len(retain_dataset)}")

    return OrderedDict(
        retain=retain_loader, forget=forget_loader,
        val=val_loader, test=test_loader,
    ), forget_dataset, retain_dataset


# =============================================================================
# InTAct unlearning
# =============================================================================

def run_intact_unlearn(cfg, model, data_loaders, criterion, device):
    """Run InTAct unlearning for classification models."""
    import wandb
    from InTAct.intact import UnlearnIntervalProtection, classification_forward_fn

    ic = cfg.get("intact", {})
    protection = UnlearnIntervalProtection(
        targets=ic.get("targets", ["fc"]),
        lambda_interval=ic.get("lambda_interval", 100.0),
        lower_percentile=ic.get("lower_percentile", 0.05),
        upper_percentile=ic.get("upper_percentile", 0.95),
        reduced_dim=ic.get("reduced_dim", 32),
        infinity_scale=ic.get("infinity_scale", 20.0),
        use_actual_bounds=ic.get("use_actual_bounds", False),
        normalize_protection=ic.get("normalize_protection", True),
    )

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders.get("retain")

    protection.setup_protection(
        model, forget_loader, device,
        remain_dataloader=retain_loader,
        forward_fn=classification_forward_fn,
    )
    protection.freeze_non_target_params(model)
    trainable = protection.get_trainable_params(model)

    lr = cfg["unlearn"].get("unlearn_lr", 0.01)
    optimizer = torch.optim.SGD(
        trainable, lr=float(lr),
        momentum=cfg["unlearn"].get("momentum", 0.9),
        weight_decay=cfg["unlearn"].get("weight_decay", 5e-4),
    )

    base_method = ic.get("base_method", "ga")
    n_epochs = cfg["unlearn"].get("unlearn_epochs", 10)
    n_classes = cfg["data"].get("num_classes", 10)

    model.train()
    step = 0
    for epoch in range(n_epochs):
        for images, targets in forget_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            output = model(images)
            if base_method == "ga":
                base_loss = -criterion(output, targets)
            elif base_method == "rl":
                rand_t = torch.randint(0, n_classes, targets.shape, device=device)
                base_loss = criterion(output, rand_t)
            else:
                base_loss = -criterion(output, targets)

            intact_loss = protection.compute_protection_loss(model, device)
            total_loss = base_loss + intact_loss
            total_loss.backward()
            optimizer.step()

            if step % 20 == 0:
                wandb.log({
                    "train/base_loss": base_loss.item(),
                    "train/intact_loss": intact_loss.item(),
                    "train/total_loss": total_loss.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                })
            step += 1

        log.info(f"Epoch {epoch}: base={base_loss.item():.4f}  intact={intact_loss.item():.4f}")


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_all(model, data_loaders, criterion, args, forget_dataset, retain_dataset, device):
    """Run accuracy + SVC-MIA evaluation; return flat metrics dict."""
    import wandb

    model.eval()
    metrics = {}

    # Per-split accuracy
    for name, loader in data_loaders.items():
        if loader is None:
            continue
        utils.dataset_convert_to_test(loader.dataset, args)
        acc = validate(loader, model, criterion, args)
        metrics[f"acc/{name}"] = acc
        log.info(f"  {name} accuracy: {acc:.2f}%")

    # Standard metric names
    # UA = Unlearning Accuracy = 100 - forget accuracy
    if "acc/forget" in metrics:
        metrics["UA"] = 100.0 - metrics["acc/forget"]
    # RA = Retain Accuracy
    if "acc/retain" in metrics:
        metrics["RA"] = metrics["acc/retain"]
    # TA = Test Accuracy
    if "acc/test" in metrics:
        metrics["TA"] = metrics["acc/test"]

    # SVC-MIA forget efficacy
    try:
        test_loader = data_loaders["test"]
        test_len = len(test_loader.dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False,
        )
        utils.dataset_convert_to_test(data_loaders["forget"].dataset, args)
        utils.dataset_convert_to_test(test_loader.dataset, args)

        mia = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=data_loaders["forget"],
            model=model,
        )
        for k, v in mia.items():
            metrics[f"mia/{k}"] = v
            log.info(f"  MIA {k}: {v:.4f}")
        # SalUn convention: MIA = SVC_MIA confidence × 100
        if "confidence" in mia:
            metrics["MIA"] = mia["confidence"] * 100
    except Exception as e:
        log.warning(f"SVC_MIA failed: {e}")

    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    import wandb

    parser = argparse.ArgumentParser(description="Classification Unlearning Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    cli = parser.parse_args()

    cfg = load_config(cli.config)

    # --- wandb init ---
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        group=cfg["wandb"].get("group"),
        tags=cfg["wandb"].get("tags", []),
        config=cfg,
    )
    cfg = merge_wandb_config(cfg)

    # --- setup ---
    gpu_id = cfg["pipeline"].get("gpu", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    utils.setup_seed(cfg["pipeline"].get("seed", 2))

    args = build_args(cfg)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- model + data ---
    model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
    model = model.to(device)

    # Load pretrained checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    log.info(f"Loaded pretrained model from {args.model_path}")

    # Split into forget / retain
    data_loaders, forget_dataset, retain_dataset = build_data_loaders(
        args, marked_loader, val_loader, test_loader,
    )

    criterion = nn.CrossEntropyLoss()

    # --- unlearn ---
    method = cfg["unlearn"]["method"]
    log.info(f"Unlearning method: {method}  |  setting: {cfg['pipeline']['setting']}")

    if method == "intact":
        run_intact_unlearn(cfg, model, data_loaders, criterion, device)
    elif method == "raw":
        log.info("raw: skipping unlearning, evaluating original model")
    else:
        if method != "retrain":
            # retrain doesn't need pretrained weights
            pass  # weights already loaded above
        unlearn_fn = unlearn.get_unlearn_method(method)
        unlearn_fn(data_loaders, model, criterion, args)

    # --- evaluate ---
    log.info("Running evaluation …")
    metrics = evaluate_all(
        model, data_loaders, criterion, args,
        forget_dataset, retain_dataset, device,
    )

    wandb.log(metrics)
    wandb.summary.update(metrics)
    log.info(f"Final metrics: {metrics}")

    # --- save model artifact ---
    ckpt_path = os.path.join(args.save_dir, f"{wandb.run.id}_model.pth")
    torch.save(model.state_dict(), ckpt_path)

    art = wandb.Artifact(
        name=f"classifier-{cfg['pipeline']['setting']}-{wandb.run.id}",
        type="model",
        metadata=metrics,
    )
    art.add_file(ckpt_path)
    wandb.log_artifact(art)
    log.info(f"Model saved to {ckpt_path} and logged as wandb artifact")

    wandb.finish()
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
