import argparse
import csv
import os
import time as time_module

import matplotlib.pyplot as plt
import numpy as np
import torch
from convertModels import savemodelDiffusers
from dash_sd_targets import select_unet_dash_params
from dataset import setup_class_forgetting_data, setup_forget_nsfw_data, setup_model
from dash_sd_runtime import run_dash_sd_warm_start
from run_naming import build_sd_unlearn_name
from training_eval import (
    compute_unet_change_stats,
    compute_unet_change_stats_from_baseline,
    run_training_eval,
    should_run_pre_epoch_train_eval,
    should_run_train_eval,
    snapshot_unet_change_baseline,
    write_metric_dict,
)
from tqdm import tqdm


def _next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _next_or_none(iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        return None, iterator


def _retain_label_pool(forget_indices, descriptions, device=None):
    forget_set = {int(idx) for idx in forget_indices}
    retain_labels = [idx for idx in range(len(descriptions)) if idx not in forget_set]
    if not retain_labels:
        raise ValueError("Cannot sample pseudo labels: every description is marked as forgotten.")
    return torch.tensor(retain_labels, dtype=torch.long, device=device)


def _sample_pseudo_labels(batch_size, forget_indices, descriptions, device=None):
    retain_labels = _retain_label_pool(forget_indices, descriptions, device=device)
    sample_indices = torch.randint(
        0,
        int(retain_labels.numel()),
        (int(batch_size),),
        device=retain_labels.device,
    )
    return retain_labels[sample_indices].tolist()


def _normalize_rl_loss_mode(rl_loss_mode):
    mode = str(rl_loss_mode or "output_matching").strip().lower().replace("-", "_")
    aliases = {
        "current": "output_matching",
        "output_match": "output_matching",
        "output_matching": "output_matching",
        "denoise": "denoise_pseudo",
        "pseudo_denoise": "denoise_pseudo",
        "denoise_pseudo": "denoise_pseudo",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported rl_loss_mode={rl_loss_mode!r}. "
            "Expected one of: output_matching, denoise_pseudo."
        )
    return aliases[mode]


def _log_wandb_scalars(payload, step=None):
    try:
        import wandb
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return
    if step is not None:
        payload = dict(payload)
        payload.setdefault("progress____global_step____train", float(step))
    wandb.log(payload)


def _sync_cuda_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now_seconds():
    _sync_cuda_if_needed()
    return time_module.perf_counter()


def _update_loss_accumulator(accumulator, values):
    for key, value in values.items():
        if not np.isfinite(value):
            continue
        entry = accumulator.setdefault(key, {"sum": 0.0, "count": 0})
        entry["sum"] += float(value)
        entry["count"] += 1


def _loss_mean_payload(accumulator, prefix):
    return {
        f"loss____train/{key}____{prefix}": entry["sum"] / max(entry["count"], 1)
        for key, entry in accumulator.items()
        if entry["count"] > 0
    }


def _set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    print(f"Seed set to {int(seed)}")


def _select_unlearning_parameters(model, train_method, dash_config=None):
    method = (train_method or "").lower()
    if method == "full":
        return list(model.model.diffusion_model.parameters())

    target = "unet_xattn" if method == "xattn" else method
    if target in {"unet", "unet_all", "all", "unet_xattn", "unet_attn", "unet_resnet", "unet_resblock"}:
        dash_targets = select_unet_dash_params(
            model,
            dash_target=target,
            include_bias=bool((dash_config or {}).get("include_bias", False)),
        )
        return [target.param for target in dash_targets]

    raise ValueError(
        f"Unsupported train_method={train_method!r}. "
        "Expected one of: xattn, full, unet, unet_xattn, unet_attn, unet_resnet."
    )


def _param_module_name(param_name):
    return param_name.rsplit(".", 1)[0] if "." in param_name else ""


def _module_or_ancestor_matches(module_name, module_by_name, tokens):
    names = [module_name]
    parts = module_name.split(".") if module_name else []
    names.extend(".".join(parts[:idx]) for idx in range(len(parts) - 1, 0, -1))
    for name in names:
        lowered_name = name.lower()
        module = module_by_name.get(name)
        lowered_type = module.__class__.__name__.lower() if module is not None else ""
        if any(token in lowered_name or token in lowered_type for token in tokens):
            return True
    return False


def _grad_group_key(param_name, module_by_name):
    lowered = param_name.lower()
    module_name = _param_module_name(param_name)
    if "attn2" in lowered:
        return "cross_attn"
    if "attn1" in lowered:
        return "self_attn"
    if _module_or_ancestor_matches(module_name, module_by_name, {"resblock", "resnet"}):
        return "resnet"
    if "attn" in lowered:
        return "attention_other"
    return "other"


def _resnet_stage_key(param_name):
    if param_name.startswith("input_blocks."):
        return "resnet_stage/input_blocks"
    if param_name.startswith("middle_block."):
        return "resnet_stage/middle_block"
    if param_name.startswith("output_blocks."):
        return "resnet_stage/output_blocks"
    return "resnet_stage/other"


def _add_grad_stats(norms, prefix, grad):
    if grad is None:
        return
    grad_float = grad.detach().float()
    entry = norms.setdefault(prefix, {"sum_sq": 0.0, "count": 0.0, "max_abs": 0.0})
    entry["sum_sq"] += float(grad_float.pow(2).sum().item())
    entry["count"] += float(grad_float.numel())
    entry["max_abs"] = max(entry["max_abs"], float(grad_float.abs().max().item()))


def _finalize_grad_norms(norms):
    payload = {}
    for key, values in norms.items():
        payload[f"grad_norm____unet/{key}____train"] = values["sum_sq"] ** 0.5
        payload[f"grad_peak_abs____unet/{key}____train"] = values["max_abs"]
    return payload


def _collect_unet_grad_norms(model, selected_param_ids):
    unet = model.model.diffusion_model
    module_by_name = dict(unet.named_modules())
    norms = {}
    for param_name, param in unet.named_parameters():
        if param.grad is None:
            continue
        group = _grad_group_key(param_name, module_by_name)
        if id(param) not in selected_param_ids:
            continue
        _add_grad_stats(norms, "total", param.grad)
        _add_grad_stats(norms, group, param.grad)
        if group == "resnet":
            stage = _resnet_stage_key(param_name)
            _add_grad_stats(norms, stage, param.grad)
    payload = _finalize_grad_norms(norms)
    payload["grad_norm____unet/log_event____train"] = 1.0
    return payload


def _grad_norm_log_points(epoch_batches, logs_per_epoch):
    epoch_batches = int(epoch_batches)
    logs_per_epoch = int(logs_per_epoch or 0)
    if logs_per_epoch <= 0 or epoch_batches <= 0:
        return set()
    last_safe_step = max(epoch_batches - 2, 0)
    if last_safe_step == 0:
        return {0}
    point_count = min(logs_per_epoch, last_safe_step + 1)
    raw_points = np.linspace(0, last_safe_step, num=point_count)
    return {int(np.floor(point + 0.5)) for point in raw_points}


def _should_log_grad_norm(epoch_step, epoch_batches, logs_per_epoch):
    return int(epoch_step) in _grad_norm_log_points(epoch_batches, logs_per_epoch)


def _unlearn_epoch_batch_count(retain_dl, forget_dl, full_retain_per_epoch=False, use_forget_in_unlearn=True):
    if not use_forget_in_unlearn:
        return len(retain_dl)
    if full_retain_per_epoch:
        return max(len(retain_dl), len(forget_dl))
    return len(forget_dl)


def _loader_data_stats(retain_dl, forget_dl, dash_config=None, full_retain_per_epoch=False, use_forget_in_unlearn=True):
    dash_config = dash_config or {}
    retain_batches = len(retain_dl)
    forget_batches = len(forget_dl)
    unlearn_batches = _unlearn_epoch_batch_count(
        retain_dl,
        forget_dl,
        full_retain_per_epoch=full_retain_per_epoch,
        use_forget_in_unlearn=use_forget_in_unlearn,
    )
    dash_retain_limit = dash_config.get("retain_batches")
    dash_forget_limit = dash_config.get("forget_batches")
    dash_uses_forget = bool(dash_config.get("warm_start", False)) and dash_config.get("signal_mode", "retain_only") != "retain_only"
    return {
        "retain_dataset_size": float(len(retain_dl.dataset)),
        "forget_dataset_size": float(len(forget_dl.dataset)),
        "retain_loader_batches": float(retain_batches),
        "forget_loader_batches": float(forget_batches),
        "dash_retain_batches_requested": float(dash_retain_limit) if dash_retain_limit is not None else float(retain_batches),
        "dash_forget_batches_requested": (
            float(dash_forget_limit) if dash_uses_forget and dash_forget_limit is not None
            else float(forget_batches) if dash_uses_forget
            else 0.0
        ),
        "dash_retain_batches_effective": float(min(retain_batches, int(dash_retain_limit))) if dash_retain_limit is not None else float(retain_batches),
        "dash_forget_batches_effective": (
            float(min(forget_batches, int(dash_forget_limit))) if dash_uses_forget and dash_forget_limit is not None
            else float(forget_batches) if dash_uses_forget
            else 0.0
        ),
        "unlearn_full_retain_per_epoch": 1.0 if full_retain_per_epoch else 0.0,
        "unlearn_uses_forget_set": 1.0 if use_forget_in_unlearn else 0.0,
        "unlearn_steps_per_epoch": float(unlearn_batches),
        "unlearn_forget_batches_per_epoch": (
            0.0 if not use_forget_in_unlearn else
            float(min(unlearn_batches, forget_batches)) if full_retain_per_epoch else float(unlearn_batches)
        ),
        "unlearn_retain_batches_per_epoch": (
            float(unlearn_batches) if not use_forget_in_unlearn else
            float(min(unlearn_batches, retain_batches)) if full_retain_per_epoch else float(unlearn_batches)
        ),
        "unlearn_forget_full_pass_per_epoch": 0.0 if not use_forget_in_unlearn else 1.0 if unlearn_batches >= forget_batches else 0.0,
        "unlearn_retain_full_pass_per_epoch": 1.0 if unlearn_batches >= retain_batches else 0.0,
        "unlearn_smaller_loader_cycles": 0.0 if (full_retain_per_epoch or not use_forget_in_unlearn) else 1.0,
    }



class _PromptedImageLoader:
    def __init__(self, loader, prompt):
        self.loader = loader
        self.prompt = str(prompt)
        self.dataset = getattr(loader, "dataset", None)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            images = _nsfw_images_from_batch(batch)
            yield {"jpg": images, "txt": [self.prompt] * int(images.shape[0])}


def _nsfw_images_from_batch(batch):
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        for key in ("jpg", "image", "images"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise ValueError(f"Unsupported NSFW batch type: {type(batch).__name__}")


def _nsfw_dash_settings(dash_config):
    dash_config = dash_config or {}
    nsfw_cfg = dash_config.get("nsfw", {}) or {}
    loss_mode = str(nsfw_cfg.get("loss_mode", "denoise")).strip().lower().replace("-", "_")
    if bool(dash_config.get("warm_start", False)) and loss_mode != "denoise":
        # TODO: implement loss_mode="nsfw_base" with base/pseudo target gradients once the denoise path is stable.
        raise NotImplementedError(
            f"Unsupported NSFW DASH loss_mode={loss_mode!r}. Only 'denoise' is implemented."
        )
    forget_prompt = str(nsfw_cfg.get("forget_prompt", "a photo of a nude person"))
    retain_prompt = str(nsfw_cfg.get("retain_prompt", "a photo of a person wearing clothes"))
    return loss_mode, forget_prompt, retain_prompt

def certain_label(
    class_to_forget,
    train_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
    model_save_dir="models",
    logs_dir="models",
    dash_config=None,
    train_eval_config=None,
    seed=None,
    full_retain_per_epoch=False,
    forget_classes=None,
    forget_concepts=None,
    use_forget_in_unlearn=True,
    method_name="rl",
    rl_loss_mode="output_matching",
    wandb_log_interval=1,
    log_grad_norms=True,
    grad_norm_log_interval=3,
    save_final_checkpoint=False,
):
    # MODEL TRAINING SETUP
    _set_seed(seed)
    rl_loss_mode = _normalize_rl_loss_mode(rl_loss_mode)
    wandb_log_interval = int(wandb_log_interval or 0)
    grad_norm_log_interval = int(grad_norm_log_interval or 0)
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    remain_dl, forget_dl, descriptions, forget_indices = setup_class_forgetting_data(
        class_to_forget=class_to_forget,
        batch_size=batch_size,
        image_size=image_size,
        forget_classes=forget_classes,
        forget_concepts=forget_concepts,
    )
    primary_forget_class = int(forget_indices[0])
    forget_name = "_".join(str(idx) for idx in forget_indices)
    data_stats = _loader_data_stats(
        remain_dl,
        forget_dl,
        dash_config=dash_config,
        full_retain_per_epoch=full_retain_per_epoch,
        use_forget_in_unlearn=use_forget_in_unlearn,
    )
    print(
        "Data usage: "
        f"retain={int(data_stats['retain_dataset_size'])} samples/{int(data_stats['retain_loader_batches'])} batches, "
        f"forget={int(data_stats['forget_dataset_size'])} samples/{int(data_stats['forget_loader_batches'])} batches, "
        f"DASH retain={int(data_stats['dash_retain_batches_effective'])} batches, "
        f"DASH forget={int(data_stats['dash_forget_batches_effective'])} batches, "
        f"unlearn steps={int(data_stats['unlearn_steps_per_epoch'])}/epoch, "
        f"full_retain_per_epoch={bool(full_retain_per_epoch)}, "
        f"use_forget_in_unlearn={bool(use_forget_in_unlearn)}"
    )

    uc_for_name = {
        "method": method_name,
        "class_to_forget": class_to_forget,
        "forget_classes": forget_classes,
        "forget_concepts": forget_concepts,
        "train_method": train_method,
        "alpha": alpha,
        "epochs": epochs,
        "lr": lr,
        "full_retain_per_epoch": full_retain_per_epoch,
    }
    if method_name == "rl":
        uc_for_name["rl_loss_mode"] = rl_loss_mode
    name = build_sd_unlearn_name(
        setting="sd",
        uc=uc_for_name,
        dash_cfg=dash_config,
        seed=seed,
        mask=bool(mask_path),
    )
    folder_path = f"{logs_dir}/{name}"
    os.makedirs(folder_path, exist_ok=True)

    if should_run_pre_epoch_train_eval(train_eval_config):
        run_training_eval(
            model=model,
            name=name,
            epoch=-1,
            class_to_forget=primary_forget_class,
            config_path=config_path,
            ckpt_path=ckpt_path,
            diffusers_config_path=diffusers_config_path,
            device=device,
            image_size=image_size,
            logs_dir=logs_dir,
            eval_config=train_eval_config,
        )

    dash_start = _now_seconds()
    dash_stats = run_dash_sd_warm_start(
        model=model,
        retain_loader=remain_dl,
        forget_loader=forget_dl,
        descriptions=descriptions,
        dash_config=dash_config,
    )
    dash_total_seconds = _now_seconds() - dash_start
    dash_stats["time____dash/total_seconds____warm_start"] = float(dash_total_seconds)
    _log_wandb_scalars({"time____dash/total_seconds____warm_start": float(dash_total_seconds)})

    # set model to train
    model.train()
    losses = []
    history_rows = []

    # choose parameters to train based on train_method
    parameters = _select_unlearning_parameters(model, train_method, dash_config=dash_config)
    if not parameters:
        raise ValueError(
            f"No trainable parameters selected for train_method={train_method!r}. "
            "Expected one of: xattn, full, unet, unet_xattn, unet_attn, unet_resnet."
        )

    optimizer = torch.optim.Adam(parameters, lr=lr)
    selected_param_ids = {id(param) for param in parameters}

    if mask_path:
        mask = torch.load(mask_path)

    dash_change_stats = compute_unet_change_stats(
        model,
        ckpt_path,
        dash_config=dash_config,
        prefix="after_dash_vs_base",
        target=(dash_config or {}).get("target", "unet"),
    )
    write_metric_dict(
        f"{folder_path}/dash_warm_start_stats.json",
        {"seed": seed if seed is not None else None, **data_stats, "rl_loss_mode": rl_loss_mode, **dash_stats, **dash_change_stats},
    )
    post_dash_train_baseline = snapshot_unet_change_baseline(
        model,
        dash_config=dash_config,
        train_method=train_method,
    )

    # TRAINING CODE
    last_epoch_seconds = None
    for epoch in range(epochs):
        epoch_start = _now_seconds()
        remain_iter = iter(remain_dl)
        forget_iter = iter(forget_dl)
        epoch_batches = _unlearn_epoch_batch_count(
            remain_dl,
            forget_dl,
            full_retain_per_epoch=full_retain_per_epoch,
            use_forget_in_unlearn=use_forget_in_unlearn,
        )
        epoch_loss_accumulator = {}
        with tqdm(total=epoch_batches) as time:

            for i in range(epoch_batches):
                optimizer.zero_grad()

                if not use_forget_in_unlearn:
                    forget_batch_data = None
                    remain_batch_data, remain_iter = _next_or_restart(remain_iter, remain_dl)
                elif full_retain_per_epoch:
                    forget_batch_data, forget_iter = _next_or_none(forget_iter)
                    remain_batch_data, remain_iter = _next_or_none(remain_iter)
                else:
                    forget_batch_data, forget_iter = _next_or_restart(forget_iter, forget_dl)
                    remain_batch_data, remain_iter = _next_or_restart(remain_iter, remain_dl)

                forget_loss = None
                remain_loss = None
                loss_terms = []

                if remain_batch_data is not None:
                    remain_images, remain_labels = remain_batch_data
                    remain_prompts = [descriptions[label] for label in remain_labels]
                    remain_batch = {
                        "jpg": remain_images.permute(0, 2, 3, 1),
                        "txt": remain_prompts,
                    }
                    remain_loss = model.shared_step(remain_batch)[0]
                    loss_terms.append(remain_loss)

                if forget_batch_data is not None:
                    forget_images, forget_labels = forget_batch_data
                    forget_prompts = [descriptions[label] for label in forget_labels]
                    pseudo_labels = _sample_pseudo_labels(
                        len(forget_labels),
                        forget_indices,
                        descriptions,
                        device=forget_labels.device,
                    )
                    pseudo_prompts = [descriptions[int(label)] for label in pseudo_labels]
                    forget_batch = {
                        "jpg": forget_images.permute(0, 2, 3, 1),
                        "txt": forget_prompts,
                    }
                    pseudo_batch = {
                        "jpg": forget_images.permute(0, 2, 3, 1),
                        "txt": pseudo_prompts,
                    }
                    forget_input, forget_emb = model.get_input(
                        forget_batch, model.first_stage_key
                    )
                    pseudo_input, pseudo_emb = model.get_input(
                        pseudo_batch, model.first_stage_key
                    )
                    t = torch.randint(
                        0,
                        model.num_timesteps,
                        (forget_input.shape[0],),
                        device=model.device,
                    ).long()
                    noise = torch.randn_like(forget_input, device=model.device)
                    forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                    if rl_loss_mode == "output_matching":
                        pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
                        forget_out = model.apply_model(forget_noisy, t, forget_emb)
                        pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()
                        forget_loss = criteria(forget_out, pseudo_out)
                    elif rl_loss_mode == "denoise_pseudo":
                        pseudo_out = model.apply_model(forget_noisy, t, pseudo_emb)
                        forget_loss = criteria(pseudo_out, noise)
                    else:
                        raise ValueError(f"Unsupported rl_loss_mode={rl_loss_mode!r}")
                    loss_terms.append(alpha * forget_loss)

                if not loss_terms:
                    raise ValueError("Internal error: both forget and retain batches are empty.")

                loss = sum(loss_terms) / len(loss_terms)
                loss.backward()
                if mask_path:
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                device
                            )

                total_loss = float(loss.detach().cpu())
                forget_loss_value = float(forget_loss.detach().cpu()) if forget_loss is not None else float("nan")
                weighted_forget_loss_value = (
                    float((alpha * forget_loss).detach().cpu()) if forget_loss is not None else float("nan")
                )
                remain_loss_value = float(remain_loss.detach().cpu()) if remain_loss is not None else float("nan")
                loss_values = {
                    "total": total_loss,
                    "forget": forget_loss_value,
                    "weighted_forget": weighted_forget_loss_value,
                    "remain": remain_loss_value,
                }
                _update_loss_accumulator(epoch_loss_accumulator, loss_values)
                global_step = len(history_rows)
                grad_payload = {}
                if log_grad_norms and _should_log_grad_norm(i, epoch_batches, grad_norm_log_interval):
                    grad_payload = _collect_unet_grad_norms(model, selected_param_ids)
                losses.append(total_loss)
                history_row = {
                    "step": global_step,
                    "epoch": epoch,
                    "batch": i,
                    "total_loss": total_loss,
                    "forget_loss": forget_loss_value,
                    "weighted_forget_loss": weighted_forget_loss_value,
                    "remain_loss": remain_loss_value,
                    "loss_term_count": len(loss_terms),
                    "has_forget_batch": 1 if forget_loss is not None else 0,
                    "has_retain_batch": 1 if remain_loss is not None else 0,
                }
                history_rows.append(history_row)
                if wandb_log_interval > 0 and global_step % wandb_log_interval == 0:
                    _log_wandb_scalars(
                        {
                            "loss____train/total____step": total_loss,
                            "loss____train/forget____step": forget_loss_value,
                            "loss____train/weighted_forget____step": weighted_forget_loss_value,
                            "loss____train/remain____step": remain_loss_value,
                            **_loss_mean_payload(epoch_loss_accumulator, "running"),
                            "meta____loss_term_count____train": float(len(loss_terms)),
                            "batch____has_forget____train": float(history_row["has_forget_batch"]),
                            "batch____has_retain____train": float(history_row["has_retain_batch"]),
                            "progress____epoch____train": float(epoch),
                            "progress____batch____train": float(i),
                            **grad_payload,
                        },
                        step=global_step,
                    )

                optimizer.step()
                time.set_description("Epoch %i" % epoch)
                time.set_postfix(loss=total_loss)
                time.update(1)

        last_epoch_seconds = _now_seconds() - epoch_start
        epoch_time_payload = {
            "time____train/epoch_seconds____epoch": float(last_epoch_seconds),
            "progress____epoch____train": float(epoch),
        }
        if epoch_batches > 0:
            epoch_time_payload["time____train/step_seconds____epoch_mean"] = float(last_epoch_seconds) / float(epoch_batches)
        if epoch == 0 and last_epoch_seconds > 0:
            epoch_time_payload["time_pct____dash/total_vs_epoch____warm_start"] = (
                100.0 * float(dash_total_seconds) / float(last_epoch_seconds)
            )
        _log_wandb_scalars(
            epoch_time_payload,
            step=len(history_rows) - 1 if history_rows else None,
        )

        if epoch_loss_accumulator:
            _log_wandb_scalars(
                {
                    **_loss_mean_payload(epoch_loss_accumulator, "epoch"),
                    "progress____epoch____train": float(epoch),
                },
                step=len(history_rows) - 1 if history_rows else None,
            )

        if should_run_train_eval(epoch, epochs, train_eval_config):
            eval_metrics = run_training_eval(
                model=model,
                name=name,
                epoch=epoch,
                class_to_forget=primary_forget_class,
                config_path=config_path,
                ckpt_path=ckpt_path,
                diffusers_config_path=diffusers_config_path,
                device=device,
                image_size=image_size,
                logs_dir=logs_dir,
                eval_config=train_eval_config,
            )
            if eval_metrics and last_epoch_seconds and last_epoch_seconds > 0:
                pct_payload = {}
                for key, pct_name in {
                    "time____eval_train/total_seconds____epoch": "time_pct____eval_train/total_vs_epoch____epoch",
                    "time____eval_train/generate_images_seconds____epoch": "time_pct____eval_train/generate_images_vs_epoch____epoch",
                    "time____eval_train/fid_seconds____epoch": "time_pct____eval_train/fid_vs_epoch____epoch",
                }.items():
                    value = eval_metrics.get(key)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        pct_payload[pct_name] = 100.0 * float(value) / float(last_epoch_seconds)
                if pct_payload:
                    pct_payload["progress____epoch____train_eval"] = float(epoch)
                    _log_wandb_scalars(pct_payload)

    model.eval()
    final_change_stats = compute_unet_change_stats(
        model,
        ckpt_path,
        dash_config=dash_config,
        prefix="after_unlearn_vs_base",
        train_method=train_method,
    )
    final_since_dash_stats = compute_unet_change_stats_from_baseline(
        model,
        post_dash_train_baseline,
        dash_config=dash_config,
        prefix="after_unlearn_vs_after_dash",
        train_method=train_method,
    )
    write_metric_dict(
        f"{folder_path}/final_unlearn_delta_stats.json",
        {
            "seed": seed if seed is not None else None,
            **data_stats,
            "rl_loss_mode": rl_loss_mode,
            **final_change_stats,
            **final_since_dash_stats,
        },
    )
    save_model(
        model,
        name,
        epoch,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
        model_save_dir=model_save_dir,
    )
    if not save_final_checkpoint:
        print("Final model checkpoint saved temporarily for final evaluation (save_final_checkpoint=false)")

    save_history(losses, name, f"class_{forget_name}", history_rows=history_rows, logs_dir=logs_dir)


def retain_only_finetune(*args, **kwargs):
    kwargs["use_forget_in_unlearn"] = False
    kwargs["method_name"] = "roft"
    return certain_label(*args, **kwargs)



def certain_label_nsfw(
    train_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
    model_save_dir="models",
    logs_dir="models",
    dash_config=None,
    train_eval_config=None,
    seed=None,
    full_retain_per_epoch=False,
    nsfw_data_path="data/nsfw",
    not_nsfw_data_path="data/not-nsfw",
    method_name="rl",
    use_forget_in_unlearn=True,
    rl_loss_mode="output_matching",
    wandb_log_interval=1,
    log_grad_norms=True,
    grad_norm_log_interval=3,
):
    _set_seed(seed)
    wandb_log_interval = int(wandb_log_interval or 0)
    grad_norm_log_interval = int(grad_norm_log_interval or 0)
    rl_loss_mode = _normalize_rl_loss_mode(rl_loss_mode)
    dash_loss_mode, forget_prompt, retain_prompt = _nsfw_dash_settings(dash_config)

    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    forget_dl, remain_dl = setup_forget_nsfw_data(
        batch_size,
        image_size,
        nsfw_data_path=nsfw_data_path,
        not_nsfw_data_path=not_nsfw_data_path,
    )
    data_stats = _loader_data_stats(
        remain_dl,
        forget_dl,
        dash_config=dash_config,
        full_retain_per_epoch=full_retain_per_epoch,
        use_forget_in_unlearn=use_forget_in_unlearn,
    )
    print(
        "NSFW data usage: "
        f"retain={int(data_stats['retain_dataset_size'])} samples/{int(data_stats['retain_loader_batches'])} batches, "
        f"forget={int(data_stats['forget_dataset_size'])} samples/{int(data_stats['forget_loader_batches'])} batches, "
        f"DASH retain={int(data_stats['dash_retain_batches_effective'])} batches, "
        f"DASH forget={int(data_stats['dash_forget_batches_effective'])} batches, "
        f"unlearn steps={int(data_stats['unlearn_steps_per_epoch'])}/epoch, "
        f"use_forget_in_unlearn={bool(use_forget_in_unlearn)}"
    )

    uc_for_name = {
        "method": method_name,
        "train_method": train_method,
        "alpha": alpha,
        "epochs": epochs,
        "lr": lr,
        "full_retain_per_epoch": full_retain_per_epoch,
    }
    if method_name == "rl":
        uc_for_name["rl_loss_mode"] = rl_loss_mode
    name = build_sd_unlearn_name(
        setting="sd_nsfw",
        uc=uc_for_name,
        dash_cfg=dash_config,
        seed=seed,
        mask=bool(mask_path),
    )
    folder_path = f"{logs_dir}/{name}"
    os.makedirs(folder_path, exist_ok=True)

    if should_run_pre_epoch_train_eval(train_eval_config):
        run_training_eval(
            model=model,
            name=name,
            epoch=-1,
            class_to_forget=None,
            config_path=config_path,
            ckpt_path=ckpt_path,
            diffusers_config_path=diffusers_config_path,
            device=device,
            image_size=image_size,
            logs_dir=logs_dir,
            eval_config=train_eval_config,
        )

    dash_start = _now_seconds()
    dash_stats = run_dash_sd_warm_start(
        model=model,
        retain_loader=_PromptedImageLoader(remain_dl, retain_prompt),
        forget_loader=_PromptedImageLoader(forget_dl, forget_prompt),
        descriptions=None,
        dash_config=dash_config,
    )
    dash_total_seconds = _now_seconds() - dash_start
    dash_stats["time____dash/total_seconds____warm_start"] = float(dash_total_seconds)
    _log_wandb_scalars({"time____dash/total_seconds____warm_start": float(dash_total_seconds)})

    # set model to train
    model.train()
    losses = []
    history_rows = []

    parameters = _select_unlearning_parameters(model, train_method, dash_config=dash_config)
    if not parameters:
        raise ValueError(
            f"No trainable parameters selected for train_method={train_method!r}. "
            "Expected one of: xattn, full, unet, unet_xattn, unet_attn, unet_resnet."
        )
    optimizer = torch.optim.Adam(parameters, lr=lr)
    selected_param_ids = {id(param) for param in parameters}

    if mask_path:
        mask = torch.load(mask_path)

    dash_change_stats = compute_unet_change_stats(
        model,
        ckpt_path,
        dash_config=dash_config,
        prefix="after_dash_vs_base",
        target=(dash_config or {}).get("target", "unet"),
    )
    write_metric_dict(
        f"{folder_path}/dash_warm_start_stats.json",
        {
            "seed": seed if seed is not None else None,
            **data_stats,
            "nsfw_dash_loss_mode": dash_loss_mode,
            **dash_stats,
            **dash_change_stats,
        },
    )
    post_dash_train_baseline = snapshot_unet_change_baseline(
        model,
        dash_config=dash_config,
        train_method=train_method,
    )

    last_epoch_seconds = None
    for epoch in range(epochs):
        epoch_start = _now_seconds()
        remain_iter = iter(remain_dl)
        forget_iter = iter(forget_dl)
        epoch_batches = _unlearn_epoch_batch_count(
            remain_dl,
            forget_dl,
            full_retain_per_epoch=full_retain_per_epoch,
            use_forget_in_unlearn=use_forget_in_unlearn,
        )
        epoch_loss_accumulator = {}
        with tqdm(total=epoch_batches) as time:
            for i in range(epoch_batches):
                optimizer.zero_grad()

                if not use_forget_in_unlearn:
                    forget_batch_data = None
                    remain_batch_data, remain_iter = _next_or_restart(remain_iter, remain_dl)
                elif full_retain_per_epoch:
                    forget_batch_data, forget_iter = _next_or_none(forget_iter)
                    remain_batch_data, remain_iter = _next_or_none(remain_iter)
                else:
                    forget_batch_data, forget_iter = _next_or_restart(forget_iter, forget_dl)
                    remain_batch_data, remain_iter = _next_or_restart(remain_iter, remain_dl)

                forget_loss = None
                remain_loss = None
                loss_terms = []

                if remain_batch_data is not None:
                    remain_images = _nsfw_images_from_batch(remain_batch_data)
                    remain_prompts = [retain_prompt] * int(remain_images.shape[0])
                    remain_batch = {
                        "jpg": remain_images.permute(0, 2, 3, 1),
                        "txt": remain_prompts,
                    }
                    remain_loss = model.shared_step(remain_batch)[0]
                    loss_terms.append(remain_loss)

                if forget_batch_data is not None:
                    forget_images = _nsfw_images_from_batch(forget_batch_data)
                    forget_prompts = [forget_prompt] * int(forget_images.shape[0])
                    pseudo_prompts = [retain_prompt] * int(forget_images.shape[0])
                    forget_batch = {
                        "jpg": forget_images.permute(0, 2, 3, 1),
                        "txt": forget_prompts,
                    }
                    pseudo_batch = {
                        "jpg": forget_images.permute(0, 2, 3, 1),
                        "txt": pseudo_prompts,
                    }
                    forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
                    pseudo_input, pseudo_emb = model.get_input(pseudo_batch, model.first_stage_key)
                    t = torch.randint(
                        0,
                        model.num_timesteps,
                        (forget_input.shape[0],),
                        device=model.device,
                    ).long()
                    noise = torch.randn_like(forget_input, device=model.device)
                    forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                    pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
                    forget_out = model.apply_model(forget_noisy, t, forget_emb)
                    pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()
                    forget_loss = criteria(forget_out, pseudo_out)
                    loss_terms.append(alpha * forget_loss)

                if not loss_terms:
                    raise ValueError("Internal error: both forget and retain batches are empty.")

                loss = sum(loss_terms) / len(loss_terms)
                loss.backward()
                if mask_path:
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(device)

                total_loss = float(loss.detach().cpu())
                forget_loss_value = float(forget_loss.detach().cpu()) if forget_loss is not None else float("nan")
                weighted_forget_loss_value = (
                    float((alpha * forget_loss).detach().cpu()) if forget_loss is not None else float("nan")
                )
                remain_loss_value = float(remain_loss.detach().cpu()) if remain_loss is not None else float("nan")
                loss_values = {
                    "total": total_loss,
                    "forget": forget_loss_value,
                    "weighted_forget": weighted_forget_loss_value,
                    "remain": remain_loss_value,
                }
                _update_loss_accumulator(epoch_loss_accumulator, loss_values)
                global_step = len(history_rows)
                grad_payload = {}
                if log_grad_norms and _should_log_grad_norm(i, epoch_batches, grad_norm_log_interval):
                    grad_payload = _collect_unet_grad_norms(model, selected_param_ids)
                losses.append(total_loss)
                history_row = {
                    "step": global_step,
                    "epoch": epoch,
                    "batch": i,
                    "total_loss": total_loss,
                    "forget_loss": forget_loss_value,
                    "weighted_forget_loss": weighted_forget_loss_value,
                    "remain_loss": remain_loss_value,
                    "loss_term_count": len(loss_terms),
                    "has_forget_batch": 1 if forget_loss is not None else 0,
                    "has_retain_batch": 1 if remain_loss is not None else 0,
                }
                history_rows.append(history_row)
                if wandb_log_interval > 0 and global_step % wandb_log_interval == 0:
                    _log_wandb_scalars(
                        {
                            "loss____train/total____step": total_loss,
                            "loss____train/forget____step": forget_loss_value,
                            "loss____train/weighted_forget____step": weighted_forget_loss_value,
                            "loss____train/remain____step": remain_loss_value,
                            **_loss_mean_payload(epoch_loss_accumulator, "running"),
                            "meta____loss_term_count____train": float(len(loss_terms)),
                            "batch____has_forget____train": float(history_row["has_forget_batch"]),
                            "batch____has_retain____train": float(history_row["has_retain_batch"]),
                            "progress____epoch____train": float(epoch),
                            "progress____batch____train": float(i),
                            **grad_payload,
                        },
                        step=global_step,
                    )

                optimizer.step()
                time.set_description("Epoch %i" % epoch)
                time.set_postfix(loss=total_loss)
                time.update(1)

        last_epoch_seconds = _now_seconds() - epoch_start
        epoch_time_payload = {
            "time____train/epoch_seconds____epoch": float(last_epoch_seconds),
            "progress____epoch____train": float(epoch),
        }
        if epoch_batches > 0:
            epoch_time_payload["time____train/step_seconds____epoch_mean"] = float(last_epoch_seconds) / float(epoch_batches)
        if epoch == 0 and last_epoch_seconds > 0:
            epoch_time_payload["time_pct____dash/total_vs_epoch____warm_start"] = (
                100.0 * float(dash_total_seconds) / float(last_epoch_seconds)
            )
        _log_wandb_scalars(epoch_time_payload, step=len(history_rows) - 1 if history_rows else None)
        if epoch_loss_accumulator:
            _log_wandb_scalars(
                {
                    **_loss_mean_payload(epoch_loss_accumulator, "epoch"),
                    "progress____epoch____train": float(epoch),
                },
                step=len(history_rows) - 1 if history_rows else None,
            )

        if should_run_train_eval(epoch, epochs, train_eval_config):
            eval_metrics = run_training_eval(
                model=model,
                name=name,
                epoch=epoch,
                class_to_forget=None,
                config_path=config_path,
                ckpt_path=ckpt_path,
                diffusers_config_path=diffusers_config_path,
                device=device,
                image_size=image_size,
                logs_dir=logs_dir,
                eval_config=train_eval_config,
            )
            if eval_metrics and last_epoch_seconds and last_epoch_seconds > 0:
                pct_payload = {}
                for key, pct_name in {
                    "time____eval_train/total_seconds____epoch": "time_pct____eval_train/total_vs_epoch____epoch",
                    "time____eval_train/generate_images_seconds____epoch": "time_pct____eval_train/generate_images_vs_epoch____epoch",
                    "time____eval_train/fid_seconds____epoch": "time_pct____eval_train/fid_vs_epoch____epoch",
                }.items():
                    value = eval_metrics.get(key)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        pct_payload[pct_name] = 100.0 * float(value) / float(last_epoch_seconds)
                if pct_payload:
                    pct_payload["progress____epoch____train_eval"] = float(epoch)
                    _log_wandb_scalars(pct_payload)

    model.eval()
    final_change_stats = compute_unet_change_stats(
        model,
        ckpt_path,
        dash_config=dash_config,
        prefix="after_unlearn_vs_base",
        train_method=train_method,
    )
    final_since_dash_stats = compute_unet_change_stats_from_baseline(
        model,
        post_dash_train_baseline,
        dash_config=dash_config,
        prefix="after_unlearn_vs_after_dash",
        train_method=train_method,
    )
    write_metric_dict(
        f"{folder_path}/final_unlearn_delta_stats.json",
        {
            "seed": seed if seed is not None else None,
            **data_stats,
            "nsfw_dash_loss_mode": dash_loss_mode,
            **final_change_stats,
            **final_since_dash_stats,
        },
    )
    save_model(
        model,
        name,
        epoch,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
        model_save_dir=model_save_dir,
    )
    save_history(losses, name, "nsfw", history_rows=history_rows, logs_dir=logs_dir)


def retain_only_finetune_nsfw(*args, **kwargs):
    kwargs["use_forget_in_unlearn"] = False
    kwargs["method_name"] = "roft"
    return certain_label_nsfw(*args, **kwargs)

def moving_average(a, n=3):
    values = np.asarray(a, dtype=float)
    if values.size == 0:
        return values
    n = max(1, min(int(n), int(values.size)))
    ret = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.figure()
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)
    plt.close()


def plot_training_history(history_rows, folder_path):
    if not history_rows:
        return

    steps = [row["step"] for row in history_rows]
    series = [
        ("total_loss", "total_loss.png"),
        ("forget_loss", "forget_loss.png"),
        ("weighted_forget_loss", "weighted_forget_loss.png"),
        ("remain_loss", "retain_loss.png"),
    ]

    for key, filename in series:
        plt.figure(figsize=(11, 5.5))
        plt.plot(steps, [row[key] for row in history_rows], linewidth=1.2, label=key)
        plt.legend(loc="upper left")
        plt.title(key.replace("_", " "), fontsize=18)
        plt.xlabel("Step", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{folder_path}/{filename}", dpi=160)
        plt.close()

    plt.figure(figsize=(11, 5.5))
    for key, _ in series:
        plt.plot(steps, [row[key] for row in history_rows], linewidth=1.2, label=key)
    plt.legend(loc="upper left")
    plt.title("Training losses", fontsize=18)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/training_losses.png", dpi=160)
    plt.close()


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
    model_save_dir="models",
):
    # SAVE MODEL
    folder_path = f"{model_save_dir}/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device,
            save_dir=model_save_dir, checkpoint_path=path
        )
        diffusers_path = f"{folder_path}/{name.replace('compvis','diffusers')}.pt"
        if not os.path.exists(diffusers_path):
            raise FileNotFoundError(f"Diffusers export failed or wrote no checkpoint: {diffusers_path}")


def save_history(losses, name, word_print, history_rows=None, logs_dir="models"):
    folder_path = f"{logs_dir}/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([f"{i}\n" for i in losses])
    if history_rows is not None:
        with open(f"{folder_path}/training_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "epoch",
                    "batch",
                    "total_loss",
                    "forget_loss",
                    "weighted_forget_loss",
                    "remain_loss",
                    "loss_term_count",
                    "has_forget_batch",
                    "has_retain_batch",
                ],
            )
            writer.writeheader()
            writer.writerows(history_rows)
        plot_training_history(history_rows, folder_path)
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=False,
        default="0",
    )
    parser.add_argument("--forget_classes", nargs="+", default=None)
    parser.add_argument("--forget_concepts", nargs="+", default=None)
    parser.add_argument(
        "--train_method",
        help="parameters to train: xattn, full, unet, unet_xattn, unet_attn, unet_resnet",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--alpha",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=5
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--rl_loss_mode",
        choices=["output_matching", "denoise_pseudo"],
        default="output_matching",
        help="RL forget loss: output_matching keeps the current implementation; denoise_pseudo uses pseudo-prompt denoising.",
    )
    parser.add_argument(
        "--wandb_log_interval",
        type=int,
        default=1,
        help="Log training losses to an active wandb run every N steps; 0 disables per-step wandb logging.",
    )
    parser.add_argument(
        "--log_grad_norms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log grouped U-Net gradient norms to an active wandb run.",
    )
    parser.add_argument(
        "--grad_norm_log_interval",
        type=int,
        default=3,
        help="Log grouped gradient norms N times per epoch at evenly spaced steps; 0 disables gradient norm logging.",
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="4",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument("--dash_warm_start", action="store_true")
    parser.add_argument("--dash_target", type=str, default="unet")
    parser.add_argument("--dash_signal_mode", type=str, default="preserve_complement")
    parser.add_argument("--plasticity_granularity", type=str, default="per_filter")
    parser.add_argument("--dash_attention_head_wise", action="store_true")
    parser.add_argument("--dash_grad_aggregation", type=str, default="mean")
    parser.add_argument("--dash_alpha", type=float, default=0.1)
    parser.add_argument("--dash_num_aug", type=int, default=10)
    parser.add_argument("--dash_aug_mode", type=str, default="none")
    parser.add_argument("--dash_min_shrink", type=float, default=0.004)
    parser.add_argument("--dash_svd_truncate_evr", type=float, default=0.95)
    parser.add_argument("--dash_preserve_forget_evr", type=float, default=0.95)
    parser.add_argument("--dash_include_bias", action="store_true")
    parser.add_argument("--dash_retain_batches", type=int, default=None)
    parser.add_argument("--dash_forget_batches", type=int, default=None)
    parser.add_argument("--bn_recalibrate", action="store_true")
    parser.add_argument("--bn_recalib_batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--full_retain_per_epoch", action="store_true")
    args = parser.parse_args()

    # classes = [int(d) for d in args.classes.split(',')]
    classes = args.class_to_forget
    train_method = args.train_method
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    config_path = args.config_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    dash_config = {
        "warm_start": args.dash_warm_start,
        "target": args.dash_target,
        "signal_mode": args.dash_signal_mode,
        "plasticity_granularity": args.plasticity_granularity,
        "attention_head_wise": args.dash_attention_head_wise,
        "grad_aggregation": args.dash_grad_aggregation,
        "alpha": args.dash_alpha,
        "num_aug": args.dash_num_aug,
        "aug_mode": args.dash_aug_mode,
        "min_shrink": args.dash_min_shrink,
        "svd_truncate_evr": args.dash_svd_truncate_evr,
        "preserve_forget_evr": args.dash_preserve_forget_evr,
        "include_bias": args.dash_include_bias,
        "retain_batches": args.dash_retain_batches,
        "forget_batches": args.dash_forget_batches,
        "bn_recalibrate": args.bn_recalibrate,
        "bn_recalib_batches": args.bn_recalib_batches,
    }

    certain_label(
        classes,
        train_method,
        alpha,
        batch_size,
        epochs,
        lr,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
        dash_config=dash_config,
        seed=args.seed,
        full_retain_per_epoch=args.full_retain_per_epoch,
        forget_classes=args.forget_classes,
        forget_concepts=args.forget_concepts,
        rl_loss_mode=args.rl_loss_mode,
        wandb_log_interval=args.wandb_log_interval,
        log_grad_norms=args.log_grad_norms,
        grad_norm_log_interval=args.grad_norm_log_interval,
        save_final_checkpoint=True,
    )
