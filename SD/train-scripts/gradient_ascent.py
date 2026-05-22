import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from convertModels import savemodelDiffusers
from dash_sd_targets import select_unet_dash_params
from dataset import setup_class_forgetting_data, setup_model
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


def _unlearn_epoch_batch_count(retain_dl, forget_dl, full_retain_per_epoch=False):
    if full_retain_per_epoch:
        return max(len(retain_dl), len(forget_dl))
    return len(forget_dl)


def _loader_data_stats(retain_dl, forget_dl, dash_config=None, full_retain_per_epoch=False):
    dash_config = dash_config or {}
    retain_batches = len(retain_dl)
    forget_batches = len(forget_dl)
    unlearn_batches = _unlearn_epoch_batch_count(
        retain_dl,
        forget_dl,
        full_retain_per_epoch=full_retain_per_epoch,
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
        "unlearn_steps_per_epoch": float(unlearn_batches),
        "unlearn_forget_batches_per_epoch": (
            float(min(unlearn_batches, forget_batches)) if full_retain_per_epoch else float(unlearn_batches)
        ),
        "unlearn_retain_batches_per_epoch": (
            float(min(unlearn_batches, retain_batches)) if full_retain_per_epoch else float(unlearn_batches)
        ),
        "unlearn_forget_full_pass_per_epoch": 1.0 if unlearn_batches >= forget_batches else 0.0,
        "unlearn_retain_full_pass_per_epoch": 1.0 if unlearn_batches >= retain_batches else 0.0,
        "unlearn_smaller_loader_cycles": 0.0 if full_retain_per_epoch else 1.0,
    }


def gradient_ascent(
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
):
    # MODEL TRAINING SETUP
    _set_seed(seed)
    model = setup_model(config_path, ckpt_path, device)
    # criteria = torch.nn.MSELoss()
    remain_dl, forget_dl, descriptions, forget_indices = setup_class_forgetting_data(
        class_to_forget=class_to_forget,
        batch_size=batch_size,
        image_size=image_size,
        forget_classes=forget_classes,
        forget_concepts=forget_concepts,
    )
    primary_forget_class = int(forget_indices[0])
    data_stats = _loader_data_stats(
        remain_dl,
        forget_dl,
        dash_config=dash_config,
        full_retain_per_epoch=full_retain_per_epoch,
    )
    print(
        "Data usage: "
        f"retain={int(data_stats['retain_dataset_size'])} samples/{int(data_stats['retain_loader_batches'])} batches, "
        f"forget={int(data_stats['forget_dataset_size'])} samples/{int(data_stats['forget_loader_batches'])} batches, "
        f"DASH retain={int(data_stats['dash_retain_batches_effective'])} batches, "
        f"DASH forget={int(data_stats['dash_forget_batches_effective'])} batches, "
        f"unlearn steps={int(data_stats['unlearn_steps_per_epoch'])}/epoch, "
        f"full_retain_per_epoch={bool(full_retain_per_epoch)}"
    )

    uc_for_name = {
        "method": "ga",
        "class_to_forget": class_to_forget,
        "forget_classes": forget_classes,
        "forget_concepts": forget_concepts,
        "train_method": train_method,
        "alpha": alpha,
        "epochs": epochs,
        "lr": lr,
        "full_retain_per_epoch": full_retain_per_epoch,
    }
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

    dash_stats = run_dash_sd_warm_start(
        model=model,
        retain_loader=remain_dl,
        forget_loader=forget_dl,
        descriptions=descriptions,
        dash_config=dash_config,
    )

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
        {"seed": seed if seed is not None else None, **data_stats, **dash_stats, **dash_change_stats},
    )
    post_dash_train_baseline = snapshot_unet_change_baseline(
        model,
        dash_config=dash_config,
        train_method=train_method,
    )

    # TRAINING CODE
    for epoch in range(epochs):
        remain_iter = iter(remain_dl)
        forget_iter = iter(forget_dl)
        epoch_batches = _unlearn_epoch_batch_count(
            remain_dl,
            forget_dl,
            full_retain_per_epoch=full_retain_per_epoch,
        )
        with tqdm(total=epoch_batches) as t:
            for i in range(epoch_batches):
                optimizer.zero_grad()

                if full_retain_per_epoch:
                    forget_batch_data, forget_iter = _next_or_none(forget_iter)
                    remain_batch_data, remain_iter = _next_or_none(remain_iter)
                else:
                    forget_batch_data, forget_iter = _next_or_restart(forget_iter, forget_dl)
                    remain_batch_data, remain_iter = _next_or_restart(remain_iter, remain_dl)

                forget_loss = None
                remain_loss = None
                loss_terms = []

                if forget_batch_data is not None:
                    forget_images, forget_labels = forget_batch_data
                    forget_prompts = [descriptions[label] for label in forget_labels]
                    forget_batch = {
                        "jpg": forget_images.permute(0, 2, 3, 1),
                        "txt": forget_prompts,
                    }
                    forget_loss = -model.shared_step(forget_batch)[0]
                    loss_terms.append(alpha * forget_loss)

                if remain_batch_data is not None:
                    remain_images, remain_labels = remain_batch_data
                    remain_prompts = [descriptions[label] for label in remain_labels]
                    remain_batch = {
                        "jpg": remain_images.permute(0, 2, 3, 1),
                        "txt": remain_prompts,
                    }
                    remain_loss = model.shared_step(remain_batch)[0]
                    loss_terms.append(remain_loss)

                if not loss_terms:
                    raise ValueError("Internal error: both forget and retain batches are empty.")

                loss = sum(loss_terms) / len(loss_terms)
                loss.backward()
                total_loss = float(loss.detach().cpu())
                forget_loss_value = float(forget_loss.detach().cpu()) if forget_loss is not None else float("nan")
                remain_loss_value = float(remain_loss.detach().cpu()) if remain_loss is not None else float("nan")
                losses.append(total_loss)
                history_rows.append(
                    {
                        "step": len(history_rows),
                        "epoch": epoch,
                        "batch": i,
                        "total_loss": total_loss,
                        "forget_loss": forget_loss_value,
                        "remain_loss": remain_loss_value,
                    }
                )

                if mask_path:
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                device
                            )

                optimizer.step()
                t.set_description("Epoch %i" % epoch)
                t.set_postfix(loss=total_loss)
                t.update(1)

        if should_run_train_eval(epoch, epochs, train_eval_config):
            run_training_eval(
                model=model,
                name=name,
                epoch=epoch,
                class_to_forget=forget_indices[0],
                config_path=config_path,
                ckpt_path=ckpt_path,
                diffusers_config_path=diffusers_config_path,
                device=device,
                image_size=image_size,
                logs_dir=logs_dir,
                eval_config=train_eval_config,
            )

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
            **final_change_stats,
            **final_since_dash_stats,
        },
    )
    save_model(
        model,
        name,
        None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
        model_save_dir=model_save_dir,
    )
    save_history(losses, name, f"class_{forget_indices}", history_rows=history_rows, logs_dir=logs_dir)


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
                fieldnames=["step", "epoch", "batch", "total_loss", "forget_loss", "remain_loss"],
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
        "--epochs", help="epochs used to train", type=int, required=False, default=10
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
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

    gradient_ascent(
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
    )
