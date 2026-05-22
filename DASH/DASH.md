# DASH (Direction-Aware Shrinkage Heuristic)

DASH in this repo is a stage-boundary warm-start. It does not run its own optimizer loop.
Instead, it estimates retain/forget gradient structure, builds a preserve-versus-complement signal,
and directly shrinks selected weights before the main unlearning training starts.

Implementation:
- `SEMU/Classification/unlearn/own/dash/dash_runtime.py`
- `SEMU/Classification/unlearn/own/dash/dash_utils.py`
- `SEMU/Classification/unlearn/own/stage_hooks.py`

## Where DASH runs

DASH is executed by stage hooks around the SEMU transform:
- `dash_target=semu`: pre-transform
- `dash_target=custom`: post-transform
- `dash_target=all`: post-transform

If `skip_svd=True` and `dash_target=custom`, runtime falls back to `dash_target=semu`.

Parser defaults are intentionally conservative for direct CLI use:
- `dash_target=semu`
- `dash_signal_mode=preserve_complement`
- `plasticity_granularity=per_filter`
- `dash_grad_mode=eval`
- `dash_grad_aggregation=ema`
- `dash_min_shrink=0.2`
- `dash_alpha=0.1`
- `dash_num_aug=10`
- `dash_preserve_forget_evr=0.95`
- `dash_svd_truncate_evr=None`

Some grids override these defaults explicitly. In particular,
`SEMU/Classification/cifar100_plasticity_methods_lr_grid_config.py` currently runs DASH with
`dash_target=semu`, `dash_grad_mode=eval`, `dash_grad_aggregation=mean`,
`dash_signal_mode=retain_only`, `plasticity_granularity=per_filter`,
`dash_min_shrink` taken from the sweep's plasticity scale, `dash_alpha=0.1`,
`dash_preserve_forget_evr=0.9`, `bn_recalibrate=true`, and
`bn_recalib_batches=200`.

## Current pipeline

### 1. Parameter selection

`--dash_target` controls which parameters are updated:
- `semu`: standard `Linear` / `Conv2d` weights before the SEMU transform
- `custom`: post-transform SEMU custom layers
- `all`: all model parameters after transform

Biases are skipped unless `--dash_include_bias` is enabled.

### 2. Gradient estimation

DASH computes gradients from retain data and, when needed by the selected signal mode,
also from forget data.

Current behavior:
- multiple stochastic views per batch via `--dash_num_aug`
- view-level gradients are averaged first
- then gradients across batches are aggregated by:
  - `ema`
  - or `mean`

Relevant flags:
- `--dash_grad_mode {eval,train_preserve_bn}`
- `--dash_grad_aggregation {ema,mean}`
- `--dash_alpha`
- `--dash_num_aug`
- `--dash_aug_mode {default,none}`

### 3. Signal mode

`--dash_signal_mode` is now the main control for how DASH decides what to preserve and what to shrink.

- `retain_only`
  legacy retain-centric shrinking
- `forget_perp_retain`
  uses `forget ⟂ retain`
- `preserve_complement`
  truncate forget with `--dash_preserve_forget_evr`, compute `preserve = retain ⟂ forget_trunc`, then shrink the complement more aggressively

Current default is `preserve_complement`.
The active CIFAR-100 plasticity LR grid currently overrides this to `retain_only`.

Legacy projection flags remain available for backward compatibility:
- `--dash_project_retain_to_forget`
- `--dash_project_forget_perp_retain`
- `--dash_project_retain_perp_forget`

### 4. Shared granularity

DASH now follows the same shared `--plasticity_granularity {global, per_filter}` flag as RS-FIRE.

`global`:
- one tensor-wide decision per layer
- tensor-wide projection, EVR truncation, masks, and shrink

`per_filter`:
- `Linear`: one unit = one output row
- `Conv2d`: one unit = one output filter using `[Cout, Cin * kh * kw]`
- projection, preserve/complement masks, shrink, and EVR truncation all operate at the same row/filter granularity

Current default is `per_filter`.

### 5. SVD / EVR truncation

DASH has two EVR controls:

- `--dash_svd_truncate_evr`
  optional truncation of the final DASH gradient before shrinking
- `--dash_preserve_forget_evr`
  truncates forget before computing `preserve = retain ⟂ forget_trunc` in `preserve_complement`

Both are now granularity-aware:
- `global`: tensor-wide
- `per_filter`: row/filter-local

### 6. Shrink rule

DASH does not call `optimizer.step()`. It scales parameters directly in place.

Core behavior:
- compute alignment between weight blocks and the chosen DASH signal
- turn that into a shrink factor
- clamp by `--dash_min_shrink`
  (`--dash_lambda` remains as a backward-compatible alias)
- apply the shrink to preserve/complement according to the selected signal mode

Practical interpretation:
- smaller `dash_min_shrink` means more aggressive shrinking
- larger `dash_min_shrink` keeps more of the original weights

### 7. BN recalibration

After DASH, runtime can recalibrate BN on retain data.

Current behavior:
- controlled by `--bn_recalibrate / --no_bn_recalibrate`
- enabled in the plasticity configs
- updates BN running statistics only

## CLI summary

Enablement:
- `--dash_warm_start`
- `--dash_target {semu,custom,all}`

Gradient estimation:
- `--dash_grad_mode {eval,train_preserve_bn}`
- `--dash_grad_aggregation {ema,mean}`
- `--dash_alpha`
- `--dash_num_aug`
- `--dash_aug_mode {default,none}`
- `--dash_include_bias`

Signal / granularity:
- `--dash_signal_mode {retain_only,forget_perp_retain,preserve_complement}`
- `--plasticity_granularity {global,per_filter}`
- legacy projection flags remain available

Shrink / truncation:
- `--dash_min_shrink`
- `--dash_lambda` (legacy alias)
- `--dash_svd_truncate_evr`
- `--dash_preserve_forget_evr`

Logging:
- `--dash_log_cosine_histograms`
- `--dash_cosine_hist_bins`

BN:
- `--bn_recalibrate`
- `--no_bn_recalibrate`
- `--bn_recalib_batches`

## Logging

DASH logs summaries under target-specific prefixes such as:
- `dash_semu_*`
- `dash_custom_*`
- `dash_all_*`

Logged values include:
- update counts
- cosine / alignment summaries
- projection or overlap summaries when those modes are used
- optional histogram payloads

## Practical notes

- `mean` aggregation gives the exact batch-average gradient; `ema` is smoother but not identical.
- `preserve_complement` is the main forget-targeted path in the current code.
- `per_filter` is the default and gives the most interpretable preserve/complement masks.
- `dash_target=custom` only makes sense when SEMU custom layers are actually created.
