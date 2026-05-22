"""DASH warm-start runtime orchestration."""

from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from ..rs_fire import bn_recalibrate
from .dash_utils import apply_dash_warm_start


@dataclass
class DashWarmStartConfig:
    """Configuration holder for DASH warm-start."""

    enabled: bool = False
    target: str = "custom"
    grad_mode: str = "eval"
    grad_aggregation: str = "ema"
    signal_mode: str = "retain_only"
    alpha: float = 0.05
    lambda_min_shrink: float = 0.05
    include_bias: bool = False
    project_retain_to_forget: bool = False
    project_forget_perp_retain: bool = False
    project_retain_perp_forget: bool = False
    num_aug: int = 1
    aug_mode: str = "default"
    svd_truncate_evr: float | None = None
    preserve_forget_evr: float = 0.95
    plasticity_granularity: str = "per_filter"
    bn_recalibrate: bool = False
    bn_recalib_batches: int = 200
    log_cosine_histograms: bool = False
    cosine_hist_bins: int = 50
    wandb_logger: Optional[Any] = None

    @classmethod
    def from_args(cls, args: Any) -> "DashWarmStartConfig":
        enabled = bool(getattr(args, "dash_warm_start", False))
        num_aug = max(1, int(getattr(args, "dash_num_aug", 1)))
        aug_mode = str(getattr(args, "dash_aug_mode", "default")).lower()
        bn_recalibrate_arg = getattr(args, "bn_recalibrate", None)
        if bn_recalibrate_arg is None:
            bn_recalibrate = enabled
        else:
            bn_recalibrate = bool(bn_recalibrate_arg)
        return cls(
            enabled=enabled,
            target=str(getattr(args, "dash_target", "custom")),
            grad_mode=str(getattr(args, "dash_grad_mode", "eval")),
            grad_aggregation=str(getattr(args, "dash_grad_aggregation", "ema")),
            signal_mode=str(getattr(args, "dash_signal_mode", "retain_only")).lower(),
            alpha=float(getattr(args, "dash_alpha", 0.05)),
            lambda_min_shrink=float(getattr(args, "dash_lambda", 0.05)),
            include_bias=bool(getattr(args, "dash_include_bias", False)),
            project_retain_to_forget=bool(getattr(args, "dash_project_retain_to_forget", False)),
            project_forget_perp_retain=bool(getattr(args, "dash_project_forget_perp_retain", False)),
            project_retain_perp_forget=bool(getattr(args, "dash_project_retain_perp_forget", False)),
            num_aug=num_aug,
            aug_mode=aug_mode,
            svd_truncate_evr=getattr(args, "dash_svd_truncate_evr", None),
            preserve_forget_evr=float(getattr(args, "dash_preserve_forget_evr", 0.95)),
            plasticity_granularity=str(getattr(args, "plasticity_granularity", "per_filter")).lower(),
            bn_recalibrate=bn_recalibrate,
            bn_recalib_batches=int(getattr(args, "bn_recalib_batches", 200)),
            log_cosine_histograms=bool(getattr(args, "dash_log_cosine_histograms", False)),
            cosine_hist_bins=int(getattr(args, "dash_cosine_hist_bins", 50)),
            wandb_logger=getattr(args, "wandb_logger", None),
        )


def normalize_dash_target(config: DashWarmStartConfig, *, skip_svd: bool) -> DashWarmStartConfig:
    """
    Normalize DASH target for current run context.

    If SVD is skipped, "custom" layers do not exist, so we fall back to "semu".
    """
    if config.enabled and config.target == "custom" and skip_svd:
        print(
            "⚠️  DASH warm-start: skip_svd=True with dash_target=custom. "
            "Falling back to dash_target=semu (linear/conv2d).",
            flush=True,
        )
        config.target = "semu"
    return config


def run_dash_warm_start_phase(
    *,
    phase: str,
    model: nn.Module,
    criterion: nn.Module,
    config: DashWarmStartConfig,
    retain_loader: Optional[DataLoader],
    forget_loader: Optional[DataLoader],
    retain_train_loader: Optional[DataLoader] = None,
) -> None:
    """
    Run DASH warm-start for a given phase.

    Phases:
    - "pre_transform": runs only for target="semu"
    - "post_transform": runs only for target in {"custom", "all"}
    """
    if not config.enabled:
        return

    if phase not in ("pre_transform", "post_transform"):
        raise ValueError(f"Unknown DASH phase: {phase}")

    if phase == "pre_transform":
        if config.target != "semu":
            return
    else:
        if config.target not in ("custom", "all"):
            return

    if retain_loader is None:
        print("⚠️  DASH warm-start skipped: retain loader not available.", flush=True)
        return

    if config.target == "semu":
        title = "DASH WARM-START (SEMU layers, pre-SVD)"
        changed_layers_class = ["linear", "conv2d"]
        force_all_params = False
        log_prefix = "dash_semu"
    elif config.target == "custom":
        title = "DASH WARM-START (Custom SEMU layers, post-SVD)"
        changed_layers_class = [
            "customlinear",
            "customconv2dchannelwisescope",
            "customconv2dglobalscope",
        ]
        force_all_params = False
        log_prefix = "dash_custom"
    else:
        title = "DASH WARM-START (All model parameters, post-transform)"
        changed_layers_class = None
        force_all_params = True
        log_prefix = "dash_all"

    print("\n" + "=" * 70, flush=True)
    print(title, flush=True)
    print("=" * 70, flush=True)
    print(
        f"DASH gradient robustness: num_aug={config.num_aug}, aug_mode={config.aug_mode}, "
        f"signal_mode={config.signal_mode}, granularity={config.plasticity_granularity}",
        flush=True,
    )
    if config.svd_truncate_evr is not None:
        print(f"DASH SVD truncation: evr={config.svd_truncate_evr}", flush=True)
    if config.signal_mode == "preserve_complement":
        print(f"DASH preserve reference forget truncation: evr={config.preserve_forget_evr}", flush=True)

    apply_dash_warm_start(
        model=model,
        data_loader=retain_loader,
        forget_loader=forget_loader,
        project_retain_to_forget=config.project_retain_to_forget,
        project_forget_perp_retain=config.project_forget_perp_retain,
        project_retain_perp_forget=config.project_retain_perp_forget,
        criterion=criterion,
        device=next(model.parameters()).device,
        changed_layers_class=changed_layers_class,
        include_bias=config.include_bias,
        grad_mode=config.grad_mode,
        grad_aggregation=config.grad_aggregation,
        signal_mode=config.signal_mode,
        ema_alpha=config.alpha,
        min_shrink=config.lambda_min_shrink,
        force_all_params=force_all_params,
        wandb_logger=config.wandb_logger,
        log_cosine_histograms=config.log_cosine_histograms,
        cosine_hist_bins=config.cosine_hist_bins,
        log_prefix=log_prefix,
        log_step=-1,
        num_aug=config.num_aug,
        aug_mode=config.aug_mode,
        svd_truncate_evr=config.svd_truncate_evr,
        preserve_forget_evr=config.preserve_forget_evr,
        plasticity_granularity=config.plasticity_granularity,
    )

    if config.bn_recalibrate:
        recalib_loader = retain_train_loader if retain_train_loader is not None else retain_loader
        if recalib_loader is None:
            print("WARNING: DASH BN recalibration skipped: retain loader not available.", flush=True)
        else:
            bn_summary = bn_recalibrate(
                model=model,
                retain_loader=recalib_loader,
                device=next(model.parameters()).device,
                num_batches=config.bn_recalib_batches,
            )
            print(
                "DASH BN recalibration: "
                f"batches={int(bn_summary['processed_batches'])}, "
                f"delta_mean={bn_summary['bn_running_mean_delta_norm']:.4e}, "
                f"delta_var={bn_summary['bn_running_var_delta_norm']:.4e}",
                flush=True,
            )
