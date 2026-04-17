"""
Standalone full FID evaluation for DDPM models.

Use this after a pipeline run to compute FID on the full set of generated
images (the pipeline may use a smaller n_samples_per_class for speed).

Requires the TF-based evaluator (evaluator.py) and a reference dataset
created by save_base_dataset.py.

Usage:
    cd DDPM
    python compute_fid_full.py \
        --ref_dir cifar10_without_label_0 \
        --sample_dir results/pipeline/<timestamp>/fid_samples_guidance_2.0_excluded_class_0
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Compute full FID for DDPM models")
    parser.add_argument("--ref_dir", type=str, required=True,
                        help="Directory with reference images (e.g. cifar10_without_label_0)")
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Directory with generated sample images")
    args = parser.parse_args()

    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator, read_images_folder

    print(f"Reference: {args.ref_dir}")
    print(f"Samples:   {args.sample_dir}")

    ref_arr = read_images_folder(args.ref_dir)
    sample_arr = read_images_folder(args.sample_dir)
    print(f"Reference images: {len(ref_arr)},  Sample images: {len(sample_arr)}")

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

    print(f"\nResults:")
    print(f"  FID:              {fid:.4f}")
    print(f"  sFID:             {sfid:.4f}")
    print(f"  Inception Score:  {inception_score:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {recall:.4f}")


if __name__ == "__main__":
    main()
