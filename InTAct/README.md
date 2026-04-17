# InTAct: Interval-based Task Activation Consolidation for Unlearning

Simplified implementation of InTAct unlearning method with negative space protection.

## Key Concept

InTAct computes activation intervals from the **forget set** and protects everything **outside** those intervals (negative space). This allows unlearning specific classes while preserving all other knowledge, without needing retain set data during training.

## Quick Start

```bash
# Run with default settings (unlearn classes 0 and 1)
python experiment.py

# Custom unlearning
python experiment.py \
    --unlearn_classes 0 1 2 \
    --pretrain_epochs 20 \
    --unlearn_epochs 15 \
    --lambda_interval 100.0 \
    --margin_percentile 0.2
```

## Key Parameters

- `--unlearn_classes`: Classes to forget (default: 0 1)
- `--pretrain_epochs`: Pretraining epochs (default: 10)
- `--unlearn_epochs`: Unlearning epochs (default: 10)
- `--lambda_interval`: Weight for protection loss (default: 100.0)
- `--margin_percentile`: Margin for interval bounds (default: 0.2)
- `--infinity_scale`: Scale for infinity bounds (default: 100.0)

## How It Works

1. **Pretrain**: Train model on all CIFAR-10 classes
2. **Setup Protection**: 
   - Collect activations from forget classes
   - Compute [forget_min, forget_max] intervals
   - Protect negative space: [-∞, forget_min] ∪ [forget_max, +∞]
3. **Unlearn**: Train with negative loss on forget set + protection loss
4. **Evaluate**: Test per-class accuracy
