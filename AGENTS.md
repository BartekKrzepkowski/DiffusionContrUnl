# AGENTS Instructions for unlearning-svd

## Scope

- Primary scope: `SEMU/Classification/`.
- `SEMU/DDPM/` and `SEMU/SD/` are legacy/low priority unless explicitly requested.
- Prefer minimal, surgical changes. Do not refactor unrelated code.
- Preserve existing style, naming, and abstractions.

## Environment

- This repo is currently used on the PLGrid cluster under:

```text
/net/home/plgrid/plgkrzepk/GitHub/unlearning-svd
```

- Shared project storage is:

```text
/net/storage/pr3/plgrid/plggdnnp
```

- Conda installations are cluster-provided in:

```text
/net/storage/pr3/plgrid/plggdnnp/apps/miniforge3
/net/storage/pr3/plgrid/plggdnnp/apps/miniforge3-gh200
```

- Shared Conda envs live in:

```text
/net/storage/pr3/plgrid/plggdnnp/conda_envs
```

- Default local CPU env for Codex and login-node testing on `x86_64` is:

```text
/net/storage/pr3/plgrid/plggdnnp/conda_envs/lapsum-local-cpu
```

- Use the `lapsum-gh200` environment only on GH200 / `aarch64` nodes:

```text
/net/storage/pr3/plgrid/plggdnnp/conda_envs/lapsum-gh200
```

- Preferred Python entrypoint on `x86_64` is the local wrapper from `~/software`, because it exports the env `lib/` path needed by Conda `numpy` / `torch` on this cluster:

```bash
bash ~/software/lapsum_python_local.sh ...
```

- If you use the repo wrapper `SEMU/Classification/lapsum_python.sh`, prefer passing `LAPSUM_PYTHON` explicitly so the interpreter choice is unambiguous across login-node and GH200 workflows.

- On `x86_64` login nodes, use:

```bash
LAPSUM_PYTHON=/net/storage/pr3/plgrid/plggdnnp/conda_envs/lapsum-local-cpu/bin/python \
  bash SEMU/Classification/lapsum_python.sh ...
```

- On GH200 / `aarch64` nodes, use:

```bash
LAPSUM_PYTHON=/net/storage/pr3/plgrid/plggdnnp/conda_envs/lapsum-gh200/bin/python \
  bash SEMU/Classification/lapsum_python.sh ...
```

- If running on a node with unknown architecture, verify first with `uname -m`, then confirm the env prefix with `conda env list`.
- Do not use legacy `/net/people/...`, `/net/tscratch/...`, or `/net/pr2/...` paths for new instructions or configs.
- Do not hardcode dataset/checkpoint paths.
- Read dataset paths from environment variables, e.g.:

```python
os.environ.get("CIFAR10_PATH", data_dir)
```

- Expected dataset environment variables on this cluster:

```text
CIFAR10_PATH=/net/storage/pr3/plgrid/plggdnnp/datasets/cifar10
CIFAR100_PATH=/net/storage/pr3/plgrid/plggdnnp/datasets/cifar100
TINYIMGNET_PATH=/net/storage/pr3/plgrid/plggdnnp/datasets/zh-plus_tiny-imagenet
TINYIMAGENET_PATH=/net/storage/pr3/plgrid/plggdnnp/datasets/zh-plus_tiny-imagenet
TINYIMGNET_HF_PATH=/net/storage/pr3/plgrid/plggdnnp/datasets/zh-plus_tiny-imagenet
```

- Do not commit host-specific absolute paths unless explicitly requested.

## Core Flow

```text
main_forget.py
  -> unlearn/__init__.py::get_unlearn_method()
     -> unlearn/own_SVD.py
        -> unlearn/own/impl.py
```

## Config-First Rules

- Prefer config files and CLI overrides over hardcoded behavior.
- When adding/changing a flag, update relevant wiring:
  1. `SEMU/Classification/arg_parser.py`
  2. relevant dataclass `from_args()` consumer(s)
  3. `SEMU/Classification/grid_config.py`, if sweep-related
- Register new unlearning methods in:

```text
SEMU/Classification/unlearn/__init__.py::get_unlearn_method()
```

## Critical Invariants

- Use existing `train_phase()` for selective training when applicable.
- Do not replace it with ad-hoc global `model.train()` / `model.eval()` toggles.
- Restore original model mode after temporary eval passes.
- Preserve forget/retain convention:
  - forget: `targets < 0`
  - retain: `targets >= 0`
- Representation geometry must preserve:
  - no hook leaks
  - deterministic eval behavior within tolerance
  - no mixing forget/retain paths
  - GAP/flatten shape contracts

## Change Classification

Treat a change as **non-trivial** if it touches:

- runtime logic
- parser / CLI / config / dataclass / grid wiring
- scheduler, optimizer, or LR behavior
- metrics or logging
- hooks, feature extraction, or representation collection
- checkpoint, eval, or training flow
- forget/retain data handling
- behavior-affecting bug fixes

**Trivial** changes are comments, docs-only edits, formatting, typo fixes, or behavior-preserving renames.

When unsure, treat the change as non-trivial.

## Testing Policy

- For every non-trivial change, run targeted local CPU tests:

```bash
bash ~/software/lapsum_python_local.sh -m pytest -q SEMU/Classification/tests/unit/test_<module>.py
```

- If the repo wrapper must be used during local CPU testing, pin the interpreter explicitly:

```bash
LAPSUM_PYTHON=/net/storage/pr3/plgrid/plggdnnp/conda_envs/lapsum-local-cpu/bin/python \
  bash SEMU/Classification/lapsum_python.sh -m pytest -q SEMU/Classification/tests/unit/test_<module>.py
```

- Any bug fix must include a regression test when practical.
- Parser/config changes should include parser/dataclass/grid wiring tests when applicable.
- The agent runs only targeted local CPU tests by default.
- The agent must not run these GPU gates unless explicitly requested:
  - `quick`
  - `full`
  - `all`
  - `chained`

- Report skipped GPU gates as:

```text
quick GPU gate: SKIPPED (user-owned gate)
full GPU gate: SKIPPED (user-owned gate)
```

- Do not claim a gate passed unless it was actually run and logs/output were reviewed.

## Safety and Side Effects

- Do not delete datasets, checkpoints, logs, experiment outputs, or cached artifacts unless explicitly requested.
- Do not launch long-running training jobs or large Slurm arrays unless explicitly requested.
- Do not modify shell startup files, env files, or cluster config unless explicitly requested.
- Do not create new `.md` files unless explicitly requested.
- Do not commit temporary files, backups, copied files, notebooks, debug dumps, local outputs, or experimental notes unless explicitly requested.

## Git / Pre-commit

- Keep commits minimal and related to the task.
- Do not mix feature code, experimental configs, local paths, and notes in one commit.
- When diagnosing staging/commit issues, inspect both:

```bash
git diff
git diff --cached
```

- If pre-commit hooks autofix files, review the diff, re-stage changed files, and retry when appropriate.
- Distinguish pre-commit autofix from real lint/test failures.

## Reporting

For small/trivial changes, concise reporting is enough.

For non-trivial changes, report:

1. summary of code changes
2. changed files
3. commands executed with key stdout/stderr
4. gate-by-gate test status
5. requirement traceability:

```text
requirement -> changed files -> validating tests
```

If something requested was not implemented, state it explicitly with the blocker.
