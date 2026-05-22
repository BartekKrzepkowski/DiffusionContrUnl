"""Shared Stable Diffusion run-name helpers."""

from __future__ import annotations


IMAGENETTE_NAMES = [
    "tench",
    "english_springer",
    "cassette_player",
    "chain_saw",
    "church",
    "french_horn",
    "garbage_truck",
    "gas_pump",
    "golf_ball",
    "parachute",
]


def _class_token(uc: dict) -> str:
    name_to_idx = {name.replace("_", " "): idx for idx, name in enumerate(IMAGENETTE_NAMES)}
    raw = uc.get("forget_classes") or uc.get("forget_concepts") or uc.get("class_to_forget", 0)
    values = raw if isinstance(raw, list) else [raw]
    indices = []
    for value in values:
        text = str(value).strip()
        normalized = text.lower().replace("_", " ")
        if text.lstrip("-").isdigit():
            indices.append(str(int(text)))
        elif normalized in name_to_idx:
            indices.append(str(name_to_idx[normalized]))
        else:
            indices.append(text.replace(" ", "_"))
    return "_".join(indices)


def _token_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace(" ", "_")


def _dash_token(dash_cfg: dict | None) -> str:
    dash_cfg = dash_cfg or {}
    if not bool(dash_cfg.get("warm_start", False)):
        return "dash_off"
    parts = [
        "dash_on",
        f"dt_{_token_value(dash_cfg.get('target', 'unet'))}",
        f"sig_{_token_value(dash_cfg.get('signal_mode', 'preserve_complement'))}",
        f"gran_{_token_value(dash_cfg.get('plasticity_granularity', 'per_filter'))}",
        f"agg_{_token_value(dash_cfg.get('grad_aggregation', 'mean'))}",
        f"naug_{_token_value(dash_cfg.get('num_aug', 10))}",
        f"ms_{_token_value(dash_cfg.get('min_shrink', 0.004))}",
        f"svd_{_token_value(dash_cfg.get('svd_truncate_evr', 0.95))}",
    ]
    if dash_cfg.get("signal_mode", "preserve_complement") == "preserve_complement":
        parts.append(f"pfevr_{_token_value(dash_cfg.get('preserve_forget_evr', 0.95))}")
    parts.append(f"rb_{_token_value(dash_cfg.get('retain_batches'))}")
    parts.append(f"fb_{_token_value(dash_cfg.get('forget_batches'))}")
    return "-".join(parts)


def build_sd_unlearn_name(
    *,
    setting: str,
    uc: dict,
    ic: dict | None = None,
    dash_cfg: dict | None = None,
    seed: int | None = None,
    mask: bool = False,
) -> str:
    ic = ic or {}
    method = uc["method"]
    seed_token = f"seed_{seed}" if seed is not None else "seed_none"
    full_retain_token = f"fullret_{_token_value(bool(uc.get('full_retain_per_epoch', False)))}"
    dash_token = _dash_token(dash_cfg)

    if method == "intact":
        base = ic.get("base_method", "ga")
        targets_str = "_".join(ic.get("targets", ["to_q", "to_k", "to_v"]))
        lam = uc.get("lambda_interval", ic.get("lambda_interval", 1.0))
        lr = uc.get("lr", 1e-5)
        epochs = uc.get("epochs", 5)
        if setting == "sd_nsfw":
            base_name = f"compvis-intact-nsfw-targets_{targets_str}-lambda_{lam}-lr_{lr}"
        else:
            cls = _class_token(uc)
            base_name = f"compvis-intact-{base}-class_{cls}-targets_{targets_str}-lambda_{lam}-epochs_{epochs}-lr_{lr}"
    elif method == "ga":
        tm = uc.get("train_method", "xattn")
        alpha = uc.get("alpha", 0.1)
        epochs = uc.get("epochs", 5)
        lr = uc.get("lr", 1e-5)
        mask_token = "-mask" if mask else ""
        base_name = f"compvis-ga{mask_token}-method_{tm}-alpha_{alpha}-epoch_{epochs}-lr_{lr}"
    elif method in {"rl", "roft"}:
        tm = uc.get("train_method", "xattn")
        alpha = uc.get("alpha", 0.1)
        epochs = uc.get("epochs", 5)
        lr = uc.get("lr", 1e-5)
        mask_token = "-mask" if mask else ""
        rl_mode = uc.get("rl_loss_mode")
        rl_mode_token = f"-rlmode_{_token_value(rl_mode)}" if method == "rl" and rl_mode else ""
        if setting == "sd_nsfw":
            prefix = "compvis-nsfw-roft" if method == "roft" else "compvis-nsfw-rl"
            base_name = f"{prefix}{mask_token}-method_{tm}-alpha_{alpha}-epoch_{epochs}-lr_{lr}{rl_mode_token}"
        else:
            cls = _class_token(uc)
            prefix = "compvis-roft" if method == "roft" else "compvis-cl"
            base_name = f"{prefix}{mask_token}-class_{cls}-method_{tm}-alpha_{alpha}-epoch_{epochs}-lr_{lr}{rl_mode_token}"
    elif method == "nsfw":
        tm = uc.get("train_method", "xattn")
        lr = uc.get("lr", 1e-5)
        base_name = f"compvis-nsfw-method_{tm}-lr_{lr}"
    else:
        base_name = f"compvis-{method}"

    return f"{base_name}-{seed_token}-{full_retain_token}-{dash_token}"
