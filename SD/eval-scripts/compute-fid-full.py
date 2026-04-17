"""
Standalone full FID evaluation for SD models.

Use this after a pipeline run to compute FID on all generated images
(no subsampling).  The pipeline computes a fast approximate FID; this
script gives the definitive number.

Usage:
    cd SD
    python eval-scripts/compute-fid-full.py \
        --folder_path /path/to/generated/images \
        --class_to_forget 0 \
        --image_size 512
"""

import argparse
import torch
from dataset import setup_fid_data
from torchmetrics.image.fid import FID


def compute_fid_full(class_to_forget, path, image_size):
    """Compute FID on the full set (no subsampling). Excludes forgotten class."""
    fid = FID(feature=64)
    real_set, fake_set = setup_fid_data(class_to_forget, path, image_size)
    print(f"Real images: {len(real_set)},  Fake images: {len(fake_set)}")

    # setup_fid_data applies Normalize([0.5],[0.5]) → [-1,1]; undo then scale to uint8
    real_images = ((torch.stack(real_set) * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu()
    fake_images = ((torch.stack(fake_set) * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu()

    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    score = fid.compute().item()
    print(f"FID (class_to_forget={class_to_forget}): {score:.4f}")
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute full FID for SD models")
    parser.add_argument("--folder_path", type=str, required=True,
                        help="Path to folder with generated images (named <classidx>_<sample>.png)")
    parser.add_argument("--class_to_forget", type=int, default=0,
                        help="Class index to exclude from FID (default: 0)")
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    compute_fid_full(args.class_to_forget, args.folder_path, args.image_size)
