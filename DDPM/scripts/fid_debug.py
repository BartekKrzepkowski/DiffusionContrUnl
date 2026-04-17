"""
FID debug helper:
  1. Subsample 500/class from a flat FID folder (5000/class × 9 classes)
     and compute FID vs the existing 500/class reference.
  2. Save 5000/class REAL CIFAR-10 images (excluding the forgotten class)
     as a new reference set, then compute FID vs the full generated folder.

Usage (called by the companion sbatch script):
    python scripts/fid_debug.py \
        --gen_dir  <path to fid_samples_guidance_..._excluded_class_0> \
        --ref_dir  <path to current 500/class reference> \
        --data_path ../data \
        --label_to_forget 0 \
        --new_ref_dir <output path for 5000/class reference>
"""

import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── helpers ─────────────────────────────────────────────────────────────────

def list_images(folder):
    """Return sorted list of image paths in *folder* (flat, no sub-dirs)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if os.path.splitext(f)[1].lower() in exts]
    )
    return paths


def subsample_flat_fid_folder(gen_dir, out_dir, n_classes, label_to_forget,
                              n_per_class_gen, n_per_class_keep, seed=42):
    """
    The sample_fid runner writes images sequentially:
      class 1 → 0.png .. 4999.png
      class 2 → 5000.png .. 9999.png   etc. (class 0 excluded)

    We subsample *n_per_class_keep* from each class-block and copy them
    into *out_dir* with flat numbering.
    """
    imgs = list_images(gen_dir)
    remaining_classes = [c for c in range(n_classes) if c != label_to_forget]
    expected = n_per_class_gen * len(remaining_classes)
    if len(imgs) != expected:
        print(f"WARNING: expected {expected} images but found {len(imgs)} "
              f"(n_per_class_gen might differ; falling back to uniform random)")
        # fallback: just randomly pick n_per_class_keep * len(remaining_classes)
        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(len(imgs)),
                                   min(n_per_class_keep * len(remaining_classes), len(imgs))))
    else:
        rng = random.Random(seed)
        chosen = []
        for ci, _ in enumerate(remaining_classes):
            block_start = ci * n_per_class_gen
            block_end = block_start + n_per_class_gen
            chosen.extend(sorted(rng.sample(range(block_start, block_end),
                                            n_per_class_keep)))
        chosen.sort()

    os.makedirs(out_dir, exist_ok=True)
    for new_id, idx in enumerate(chosen):
        src = imgs[idx]
        dst = os.path.join(out_dir, f"{new_id}.png")
        shutil.copy2(src, dst)

    print(f"Subsampled {len(chosen)} images → {out_dir}")
    return out_dir


def save_real_reference(data_path, label_to_forget, n_per_class, out_dir):
    """
    Save *n_per_class* REAL CIFAR-10 training images per remaining class.
    CIFAR-10 train set has 5000 images per class, so n_per_class <= 5000.
    Images are saved as flat PNGs in *out_dir*.
    """
    dataset = CIFAR10(data_path, train=True, download=True,
                      transform=transforms.ToTensor())

    # Group indices by class, excluding forgotten
    class_indices = {c: [] for c in range(10) if c != label_to_forget}
    for i, t in enumerate(dataset.targets):
        if t != label_to_forget:
            class_indices[t].append(i)

    os.makedirs(out_dir, exist_ok=True)
    img_id = 0
    for cls in sorted(class_indices.keys()):
        indices = class_indices[cls][:n_per_class]
        for idx in indices:
            img_tensor, _ = dataset[idx]
            save_image(img_tensor, os.path.join(out_dir, f"{img_id}.png"),
                       normalize=True)
            img_id += 1

    print(f"Saved {img_id} real images (label≠{label_to_forget}, "
          f"{n_per_class}/class) → {out_dir}")
    return out_dir


def compute_fid(ref_dir, sample_dir):
    """Compute FID using the TF-based evaluator."""
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator, read_images_folder

    ref_arr = read_images_folder(ref_dir)
    sample_arr = read_images_folder(sample_dir)
    print(f"  Reference images : {len(ref_arr)}")
    print(f"  Sample images    : {len(sample_arr)}")

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
        "FID": float(fid),
        "sFID": float(sfid),
        "IS": float(inception_score),
        "Precision": float(prec),
        "Recall": float(recall),
    }


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Flat folder with 5000/class generated FID images")
    parser.add_argument("--ref_dir", type=str, required=True,
                        help="Current reference dir (500/class real images)")
    parser.add_argument("--data_path", type=str, default="../data",
                        help="Path to CIFAR-10 root (for torch download)")
    parser.add_argument("--label_to_forget", type=int, default=0)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_per_class_gen", type=int, default=5000,
                        help="How many images per class in gen_dir")
    parser.add_argument("--new_ref_dir", type=str, default=None,
                        help="Where to save 5000/class real reference. "
                             "Defaults to <ref_dir>_5000")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.new_ref_dir is None:
        args.new_ref_dir = args.ref_dir.rstrip("/") + "_5000"

    n_remaining = args.n_classes - 1  # classes excl. forgotten

    # ── Task 1: subsample 500/class → FID vs. existing 500/class ref ──
    print("=" * 70)
    print("TASK 1: Subsample 500/class from generated → FID vs 500/class ref")
    print("=" * 70)
    sub_dir = args.gen_dir.rstrip("/") + "_sub500"
    subsample_flat_fid_folder(
        gen_dir=args.gen_dir,
        out_dir=sub_dir,
        n_classes=args.n_classes,
        label_to_forget=args.label_to_forget,
        n_per_class_gen=args.n_per_class_gen,
        n_per_class_keep=500,
        seed=args.seed,
    )

    print(f"\nComputing FID:  ref={args.ref_dir}  vs  samples={sub_dir}")
    metrics1 = compute_fid(args.ref_dir, sub_dir)
    print("\n--- Task 1 Results (500 vs 500 per class) ---")
    for k, v in metrics1.items():
        print(f"  {k:12s} = {v:.4f}")

    # ── Task 2: save 5000/class real ref → FID vs. full generated ──
    print("\n" + "=" * 70)
    print("TASK 2: Save 5000/class real reference → FID vs full generated")
    print("=" * 70)
    save_real_reference(
        data_path=args.data_path,
        label_to_forget=args.label_to_forget,
        n_per_class=5000,
        out_dir=args.new_ref_dir,
    )

    print(f"\nComputing FID:  ref={args.new_ref_dir}  vs  samples={args.gen_dir}")
    metrics2 = compute_fid(args.new_ref_dir, args.gen_dir)
    print("\n--- Task 2 Results (5000 vs 5000 per class) ---")
    for k, v in metrics2.items():
        print(f"  {k:12s} = {v:.4f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Task 1 (500 gen vs 500 real):   FID = {metrics1['FID']:.4f}")
    print(f"  Task 2 (5000 gen vs 5000 real):  FID = {metrics2['FID']:.4f}")


if __name__ == "__main__":
    main()
