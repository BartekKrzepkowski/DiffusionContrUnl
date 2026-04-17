"""
Calculate the percentage of unfrozen (trainable) parameters in a Classification
model when InTAct is active with a given set of target layers.

Aligned with the pipeline semantics in Classification/pipeline.py:
  - `targets` = layers whose activations InTAct protects via interval constraints.
    These are ALSO the ONLY layers passed to the optimizer (i.e. the unfrozen ones).
  - All other layers are excluded from the optimizer (effectively frozen).

Matching replicates InTAct._find_target_layers():
  - Iterates model.named_modules() (not named_parameters()).
  - A module matches if type(module).__name__ == pattern  OR  pattern (case-insensitive)
    is a substring of the module's name.
  - Counted parameters: weight + bias of each matched module.

Usage:
    # Use targets from a config file:
    cd Classification
    python calculate_unfrozen_params.py --config configs/pipeline_classwise.yaml

    # Or pass targets explicitly:
    python calculate_unfrozen_params.py --targets conv1x1 fc

    # Specify a different architecture:
    python calculate_unfrozen_params.py --arch resnet50 --num_classes 10 --targets conv1x1 fc

    # List all module names (useful for crafting target patterns):
    python calculate_unfrozen_params.py --list-modules
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from models import model_dict


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_pipeline_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def targets_from_config(config_path):
    """Return the intact.targets list from a pipeline YAML config, or None if absent."""
    cfg = load_pipeline_config(config_path)
    return cfg.get("intact", {}).get("targets", None)


def model_info_from_config(config_path):
    """Return (arch, num_classes, imagenet_arch) from a pipeline YAML config."""
    cfg = load_pipeline_config(config_path)
    arch = cfg.get("model", {}).get("arch", "resnet18")
    num_classes = cfg.get("data", {}).get("num_classes", 10)
    imagenet_arch = cfg.get("model", {}).get("imagenet_arch", False)
    return arch, num_classes, imagenet_arch


# ---------------------------------------------------------------------------
# Matching logic — mirrors InTAct._find_target_layers()
# ---------------------------------------------------------------------------

def find_target_modules(model: nn.Module, targets):
    """
    Replicate InTAct._find_target_layers() logic exactly.

    A module is a target if:
      - type(module).__name__ == pattern  (exact class-name match), OR
      - pattern.lower() is a substring of module_name.lower()

    Returns:
        dict mapping module_name -> module for all matched modules.
    """
    matched = {}
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    for name, module in base_model.named_modules():
        for pattern in targets:
            if type(module).__name__ == pattern:
                matched[name] = module
                break
            if pattern.lower() in name.lower():
                matched[name] = module
                break
    return matched


def count_module_params(module: nn.Module):
    """Count weight + bias parameters of a single module (not recursive)."""
    count = 0
    if hasattr(module, 'weight') and module.weight is not None:
        count += module.weight.numel()
    if hasattr(module, 'bias') and module.bias is not None:
        count += module.bias.numel()
    return count


# ---------------------------------------------------------------------------
# Breakdown printer
# ---------------------------------------------------------------------------

def print_breakdown(target_modules, total_unfrozen):
    print("\n" + "=" * 70)
    print("UNFROZEN LAYERS (target modules passed to optimizer)")
    print("=" * 70)

    for mod_name, module in sorted(target_modules.items()):
        n = count_module_params(module)
        pct = (n / total_unfrozen * 100) if total_unfrozen > 0 else 0.0
        layer_type = type(module).__name__
        print(f"  {mod_name:55s}  [{layer_type:12s}]  {n:>10,}  ({pct:.2f}%)")


def print_all_modules(model):
    """Print all module names in the model (useful for debugging target patterns)."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    print("\n" + "=" * 70)
    print("ALL MODULES IN MODEL")
    print("=" * 70)
    for name, module in base.named_modules():
        n_params = count_module_params(module)
        layer_type = type(module).__name__
        if name:  # skip root module
            marker = f"  {n_params:>10,} params" if n_params > 0 else ""
            print(f"  {name:55s}  [{layer_type:12s}]{marker}")
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate unfrozen param %% for Classification when InTAct targets are set"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline config YAML (intact.targets, model.arch, etc. will be read from here "
             "unless overridden by CLI flags)"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=list(model_dict.keys()),
        help="Model architecture (default: read from config, or resnet18)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of output classes (default: read from config, or 10)"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        default=None,
        help="Use ImageNet-style stem (7x7 conv + maxpool)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=None,
        help="Override InTAct target patterns (default: read from config intact.targets)"
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="Print all module names in the model and exit (useful for crafting target patterns)"
    )
    args = parser.parse_args()

    # ---- Resolve model params: CLI > config > defaults ----
    cfg_arch, cfg_num_classes, cfg_imagenet = "resnet18", 10, False
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: config not found: {args.config}")
            sys.exit(1)
        cfg_arch, cfg_num_classes, cfg_imagenet = model_info_from_config(args.config)

    arch = args.arch if args.arch else cfg_arch
    num_classes = args.num_classes if args.num_classes is not None else cfg_num_classes
    imagenet_arch = args.imagenet_arch if args.imagenet_arch is not None else cfg_imagenet

    # ---- Resolve targets: CLI > config > defaults ----
    if args.targets:
        targets = args.targets
        targets_src = "CLI --targets"
    elif args.config:
        targets = targets_from_config(args.config)
        if targets:
            targets_src = f"config ({args.config})"
        else:
            targets = ["conv1x1", "fc"]
            targets_src = "hard-coded default (config had no intact.targets)"
    else:
        targets = ["conv1x1", "fc"]
        targets_src = "hard-coded default"

    # ---- Build model ----
    print("=" * 70)
    print("Classification InTAct — UNFROZEN PARAMETER CALCULATOR")
    print("=" * 70)
    print(f"Config       : {args.config or '(none)'}")
    print(f"Architecture : {arch}  (num_classes={num_classes}, imagenet={imagenet_arch})")
    print(f"Targets      : {targets}  (source: {targets_src})")
    print("=" * 70)

    print("\nInitialising model...")
    if imagenet_arch:
        model = model_dict[arch](num_classes=num_classes, imagenet=True)
    else:
        model = model_dict[arch](num_classes=num_classes)

    # ---- List modules mode ----
    if args.list_modules:
        print_all_modules(model)
        return

    # ---- Count parameters ----
    total_params = sum(p.numel() for p in model.parameters())

    # Find target modules using InTAct matching rules
    target_modules = find_target_modules(model, targets)

    # Collect unique target parameter tensors (weight + bias) to avoid double-counting
    target_param_ids = set()
    for module in target_modules.values():
        if hasattr(module, 'weight') and module.weight is not None:
            target_param_ids.add(id(module.weight))
        if hasattr(module, 'bias') and module.bias is not None:
            target_param_ids.add(id(module.bias))

    unfrozen_params = sum(
        p.numel() for p in model.parameters() if id(p) in target_param_ids
    )
    frozen_params = total_params - unfrozen_params
    unfrozen_pct = (unfrozen_params / total_params * 100) if total_params > 0 else 0.0
    frozen_pct = (frozen_params / total_params * 100) if total_params > 0 else 0.0

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total parameters         : {total_params:,}")
    print(f"Unfrozen (target/InTAct) : {unfrozen_params:,}  ({unfrozen_pct:.2f}%)")
    print(f"Frozen   (non-target)    : {frozen_params:,}  ({frozen_pct:.2f}%)")
    print(f"Target modules matched   : {len(target_modules)}")
    print("=" * 70)

    if target_modules:
        print_breakdown(target_modules, unfrozen_params)
    else:
        print("\nWARNING: No modules matched the target patterns!")
        print("Check your target strings against actual module names.")
        print("Run with --list-modules to see all module names.\n")
        print("First 40 module names:")
        base = model.module if isinstance(model, nn.DataParallel) else model
        for i, (name, _) in enumerate(base.named_modules()):
            if i >= 40:
                print(f"  ... (truncated, {sum(1 for _ in base.named_modules())} total)")
                break
            print(f"  {name}")

    print("\n" + "=" * 70)
    print(f"  Unfrozen: {unfrozen_pct:.2f}%   |   Frozen: {frozen_pct:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
