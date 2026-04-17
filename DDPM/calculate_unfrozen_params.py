"""
Calculate the percentage of unfrozen (trainable) parameters in the DDPM model
when InTAct is active with a given set of target layers.

Aligned with the pipeline semantics in DDPM/runners/diffusion.py:
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
    cd DDPM
    python calculate_unfrozen_params.py --config configs/cifar10_intact.yml

    # Or pass targets explicitly:
    python calculate_unfrozen_params.py --config configs/cifar10_train.yml \\
        --targets attn.0.q attn.0.k attn.0.v attn_1.q attn_1.k attn_1.v \\
                  attn.1.q attn.1.k attn.1.v cemb.dense.0 cemb.dense.1
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import yaml
from functions import dict2namespace
from models.diffusion import Conditional_Model


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict2namespace(config_dict)


def targets_from_config(config_path):
    """Return the training.targets list from a YAML config, or None if absent."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get("training", {}).get("targets", None)


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
    print("\n" + "="*70)
    print("UNFROZEN LAYERS (target modules passed to optimizer)")
    print("="*70)

    # Group by the pattern that caused the match – just print per module name
    for mod_name, module in sorted(target_modules.items()):
        n = count_module_params(module)
        pct = (n / total_unfrozen * 100) if total_unfrozen > 0 else 0.0
        layer_type = type(module).__name__
        print(f"  {mod_name:55s}  [{layer_type:12s}]  {n:>10,}  ({pct:.2f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate unfrozen param % for DDPM when InTAct targets are set"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cifar10_intact.yml",
        help="Path to model/training config YAML (training.targets will be read from here "
             "unless --targets is given)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=None,
        help="Override InTAct target patterns (default: read from config training.targets)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Resolve targets: CLI > config file > hard-coded fallback
    if args.targets:
        targets = args.targets
        targets_src = "CLI --targets"
    else:
        targets = targets_from_config(args.config)
        if targets:
            targets_src = f"config ({args.config})"
        else:
            targets = [
                'attn.0.q', 'attn.0.k', 'attn.0.v',
                'attn_1.q', 'attn_1.k', 'attn_1.v',
                'attn.1.q', 'attn.1.k', 'attn.1.v',
                'cemb.dense.0', 'cemb.dense.1',
            ]
            targets_src = "hard-coded default"

    print("="*70)
    print("DDPM InTAct — UNFROZEN PARAMETER CALCULATOR")
    print("="*70)
    print(f"Config  : {args.config}")
    print(f"Targets : {targets}  (source: {targets_src})")
    print("="*70)

    # Build model (CPU is fine — we only need the architecture)
    print("\nInitialising model...")
    model = Conditional_Model(config)

    # Total parameter count
    total_params = sum(p.numel() for p in model.parameters())

    # Find target modules using InTAct matching rules
    target_modules = find_target_modules(model, targets)

    # Collect the unique set of target parameter tensors (weight + bias)
    # to avoid double-counting when a module is matched by multiple patterns
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
    frozen_pct   = (frozen_params   / total_params * 100) if total_params > 0 else 0.0

    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total parameters         : {total_params:,}")
    print(f"Unfrozen (target/InTAct) : {unfrozen_params:,}  ({unfrozen_pct:.2f}%)")
    print(f"Frozen   (non-target)    : {frozen_params:,}  ({frozen_pct:.2f}%)")
    print(f"Target modules matched   : {len(target_modules)}")
    print("="*70)

    if target_modules:
        print_breakdown(target_modules, unfrozen_params)
    else:
        print("\nWARNING: No modules matched the target patterns!")
        print("Check your target strings against actual module names:")
        base = model.module if isinstance(model, nn.DataParallel) else model
        for i, (name, _) in enumerate(base.named_modules()):
            if i >= 40:
                print(f"  ... (truncated, {sum(1 for _ in base.named_modules())} total)")
                break
            print(f"  {name}")

    print("\n" + "="*70)
    print(f"  Unfrozen: {unfrozen_pct:.2f}%   |   Frozen: {frozen_pct:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
