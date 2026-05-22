import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPTS = REPO_ROOT / "SD" / "train-scripts"
for path in (REPO_ROOT, TRAIN_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dash_sd_runtime
from dash_sd_runtime import run_dash_sd_warm_start
from dash_sd_targets import select_unet_dash_params

convert_models = types.ModuleType("convertModels")
convert_models.savemodelDiffusers = lambda *args, **kwargs: None
sys.modules.setdefault("convertModels", convert_models)
from training_eval import (
    compute_named_parameter_change_stats,
    compute_unet_change_stats,
    compute_unet_change_stats_from_baseline,
    snapshot_named_parameter_baseline,
    snapshot_unet_change_baseline,
)


class FakeResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv1(x)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_blocks = nn.ModuleDict({"0": nn.Conv2d(4, 4, kernel_size=1, bias=True)})
        self.attn2 = nn.Linear(4, 4, bias=False)
        self.resblock = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.nested_block = FakeResBlock()

    def forward(self, x, timesteps, context=None):
        h = self.input_blocks["0"](x)
        h = self.resblock(h)
        h = self.nested_block(h)
        b, c, height, width = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(-1, c)
        h_flat = self.attn2(h_flat)
        return h_flat.reshape(b, height, width, c).permute(0, 3, 1, 2)


class TinySD(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.diffusion_model = TinyUNet()
        self.first_stage_model = nn.Conv2d(4, 4, kernel_size=1)
        self.cond_stage_model = nn.Linear(4, 4)
        self.first_stage_key = "jpg"
        self.num_timesteps = 10
        self.parameterization = "eps"
        self.device = torch.device("cpu")

    def get_input(self, batch, key):
        image = batch[key]
        if image.shape[-1] in {1, 3, 4}:
            image = image.permute(0, 3, 1, 2)
        cond = torch.zeros(image.shape[0], 1, 4, device=image.device)
        return image.float(), cond

    def q_sample(self, x_start, t, noise=None):
        return x_start + 0.1 * noise

    def apply_model(self, x_noisy, t, cond):
        return self.model.diffusion_model(x_noisy, t, context=cond)

    def get_loss(self, pred, target, mean=True):
        loss = (pred - target).pow(2)
        return loss.mean() if mean else loss



class FakeCrossAttention(nn.Module):
    def __init__(self, heads=2, dim=4):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(0.0))

    def forward(self, x):
        return self.to_out(self.to_q(x) + self.to_k(x) + self.to_v(x))


class FakeUnknownAttention(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.to_q(x)


class TinyAttentionUNet(nn.Module):
    def __init__(self, unknown_heads=False):
        super().__init__()
        self.input_blocks = nn.ModuleDict({"0": nn.Conv2d(4, 4, kernel_size=1, bias=False)})
        self.block = nn.Module()
        self.block.attn2 = FakeUnknownAttention() if unknown_heads else FakeCrossAttention(heads=2, dim=4)

    def forward(self, x, timesteps, context=None):
        h = self.input_blocks["0"](x)
        b, c, height, width = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(-1, c)
        h_flat = self.block.attn2(h_flat)
        return h_flat.reshape(b, height, width, c).permute(0, 3, 1, 2)


class TinyAttentionSD(TinySD):
    def __init__(self, unknown_heads=False):
        nn.Module.__init__(self)
        self.model = nn.Module()
        self.model.diffusion_model = TinyAttentionUNet(unknown_heads=unknown_heads)
        self.first_stage_model = nn.Conv2d(4, 4, kernel_size=1)
        self.cond_stage_model = nn.Linear(4, 4)
        self.first_stage_key = "jpg"
        self.num_timesteps = 10
        self.parameterization = "eps"
        self.device = torch.device("cpu")

def _loaders():
    torch.manual_seed(1)
    retain_images = torch.randn(4, 4, 4, 4)
    forget_images = torch.randn(4, 4, 4, 4) + 0.5
    retain = DataLoader(TensorDataset(retain_images, torch.ones(4, dtype=torch.long)), batch_size=2)
    forget = DataLoader(TensorDataset(forget_images, torch.zeros(4, dtype=torch.long)), batch_size=2)
    return retain, forget, ["forget concept", "retain concept"]


def _config(**overrides):
    cfg = {
        "warm_start": True,
        "target": "unet",
        "signal_mode": "retain_only",
        "plasticity_granularity": "per_filter",
        "grad_aggregation": "mean",
        "alpha": 0.1,
        "num_aug": 1,
        "aug_mode": "none",
        "min_shrink": 0.2,
        "svd_truncate_evr": None,
        "preserve_forget_evr": 0.95,
        "include_bias": False,
        "retain_batches": 1,
        "forget_batches": 1,
    }
    cfg.update(overrides)
    return cfg


def _params_changed(before, model):
    after = {target.name: target.param.detach().clone() for target in select_unet_dash_params(model)}
    return any(not torch.allclose(before[name], after[name]) for name in before)


def test_unet_target_selection_skips_bias_vae_and_text_encoder_by_default():
    model = TinySD()
    targets = select_unet_dash_params(model, dash_target="unet", include_bias=False)
    names = {target.name for target in targets}

    assert "input_blocks.0.weight" in names
    assert "attn2.weight" in names
    assert "resblock.weight" in names
    assert "nested_block.conv1.weight" in names
    assert "input_blocks.0.bias" not in names
    assert all(target.param is not model.first_stage_model.weight for target in targets)
    assert all(target.param is not model.cond_stage_model.weight for target in targets)


def test_dash_disabled_changes_no_unet_parameters():
    model = TinySD()
    retain, forget, descriptions = _loaders()
    before = {target.name: target.param.detach().clone() for target in select_unet_dash_params(model)}

    stats = run_dash_sd_warm_start(
        model=model,
        retain_loader=retain,
        forget_loader=forget,
        descriptions=descriptions,
        dash_config={"warm_start": False},
    )

    assert stats["dash_sd_enabled"] == 0.0
    assert not _params_changed(before, model)


def test_dash_wandb_payload_groups_legacy_stat_keys():
    payload = dash_sd_runtime._wandb_dash_sd_payload(
        {
            "dash_sd_enabled": 1.0,
            "dash_sd_alignment_mean": 0.2,
            "dash_sd_shrink_min": 0.5,
            "dash_sd_grad_norm_retain": 3.0,
            "dash_sd_gradient_overlap_cos_mean": -0.1,
            "dash_sd_projection/forget_perp_retain/projected_fraction": 0.7,
        }
    )

    assert payload["dash_sd____status/enabled____warm_start"] == 1.0
    assert payload["dash_sd____alignment/mean____warm_start"] == 0.2
    assert payload["dash_sd____shrink/min____warm_start"] == 0.5
    assert payload["dash_sd____grad_norm/retain____warm_start"] == 3.0
    assert payload["dash_sd____overlap/cos_mean____warm_start"] == -0.1
    assert payload["dash_sd____projection/forget_perp_retain/projected_fraction____warm_start"] == 0.7


def test_dash_wandb_histogram_payload_logs_cosine_histograms(monkeypatch):
    class FakeHistogram:
        def __init__(self, np_histogram):
            self.np_histogram = np_histogram

    class FakeTable:
        def __init__(self, data, columns):
            self.data = data
            self.columns = columns

    class FakePlot:
        @staticmethod
        def line(table, x, y, title=None):
            return {"table": table, "x": x, "y": y, "title": title}

    fake_wandb = types.SimpleNamespace(Histogram=FakeHistogram, Table=FakeTable, plot=FakePlot)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = dash_sd_runtime.DashSDConfig(log_cosine_histograms=True, cosine_hist_bins=4)

    payload = dash_sd_runtime._wandb_dash_sd_histogram_payload(
        config,
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        {
            "input_blocks.0": torch.tensor([0.0, 1.0, 1.0, 0.0]),
            "attn2.to_q": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        },
        cosine_medians_by_module={"input_blocks.0": -0.123},
    )

    assert "dash_sd____alignment/cosine_histogram____warm_start" in payload
    counts, edges = payload["dash_sd____alignment/cosine_histogram____warm_start"].np_histogram
    assert counts.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert edges.tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert payload["dash_sd____alignment_hist/bin_00____warm_start"] == 0.1
    assert payload["dash_sd____alignment_hist/bin_03____warm_start"] == 0.4
    assert "dash_sd____alignment_cdf/bin_00____warm_start" not in payload
    cdf_plot = payload["dash_sd____alignment_cdf/plot____warm_start"]
    assert cdf_plot["x"] == "alignment"
    assert cdf_plot["y"] == "cdf"
    assert cdf_plot["table"].columns == ["alignment", "cdf"]
    assert cdf_plot["table"].data[0][0] == -0.75
    assert abs(cdf_plot["table"].data[0][1] - 0.1) < 1e-6
    assert cdf_plot["table"].data[-1] == [0.75, 1.0]
    assert payload["dash_sd____alignment_hist/negative_fraction____warm_start"] == 0.3
    assert payload["dash_sd____alignment_hist/near_zero_fraction____warm_start"] == 0.0
    assert payload["dash_sd____alignment_hist/positive_fraction____warm_start"] == 0.7
    assert "dash_sd____alignment_module/input_blocks.0/cosine_histogram____warm_start" in payload
    assert "dash_sd____alignment_module/input_blocks.0/cdf/bin_01____warm_start" not in payload
    assert "dash_sd____alignment_module/input_blocks.0/cdf/plot____warm_start" in payload
    assert abs(payload["dash_sd____alignment_module/input_blocks.0/median____warm_start"] - -0.123) < 1e-6
    assert "dash_sd____alignment_module/attn2.to_q/cosine_histogram____warm_start" in payload


def test_dash_enabled_changes_selected_unet_parameters():
    torch.manual_seed(2)
    model = TinySD()
    retain, forget, descriptions = _loaders()
    before = {target.name: target.param.detach().clone() for target in select_unet_dash_params(model)}

    stats = run_dash_sd_warm_start(
        model=model,
        retain_loader=retain,
        forget_loader=forget,
        descriptions=descriptions,
        dash_config=_config(signal_mode="retain_only"),
    )

    assert stats["dash_sd_enabled"] == 1.0
    assert stats["dash_sd_retain_batches"] == 1.0
    assert _params_changed(before, model)


def test_dash_signal_modes_run_without_shape_errors():
    for mode in ["retain_only", "forget_perp_retain", "preserve_complement"]:
        torch.manual_seed(3)
        model = TinySD()
        retain, forget, descriptions = _loaders()
        stats = run_dash_sd_warm_start(
            model=model,
            retain_loader=retain,
            forget_loader=forget,
            descriptions=descriptions,
            dash_config=_config(signal_mode=mode),
        )
        assert stats["dash_sd_updated_tensor_count"] > 0


def test_dash_granularities_run_for_linear_and_conv2d():
    for granularity in ["global", "per_filter"]:
        torch.manual_seed(4)
        model = TinySD()
        retain, forget, descriptions = _loaders()
        stats = run_dash_sd_warm_start(
            model=model,
            retain_loader=retain,
            forget_loader=forget,
            descriptions=descriptions,
            dash_config=_config(plasticity_granularity=granularity, signal_mode="retain_only"),
        )
        assert stats["dash_sd_updated_tensor_count"] == 4.0


def test_unet_resnet_selects_convs_inside_resblock_containers():
    model = TinySD()
    targets = select_unet_dash_params(model, dash_target="unet_resnet", include_bias=False)
    names = {target.name for target in targets}

    assert "nested_block.conv1.weight" in names
    assert "resblock.weight" in names
    assert "input_blocks.0.weight" not in names
    assert "attn2.weight" not in names


def test_unet_attn_and_resnet_are_named_subsets_not_full_unet_partition():
    model = TinySD()
    unet_names = {target.name for target in select_unet_dash_params(model, dash_target="unet", include_bias=False)}
    xattn_names = {target.name for target in select_unet_dash_params(model, dash_target="unet_xattn", include_bias=False)}
    attn_names = {target.name for target in select_unet_dash_params(model, dash_target="unet_attn", include_bias=False)}
    resnet_names = {target.name for target in select_unet_dash_params(model, dash_target="unet_resnet", include_bias=False)}

    assert xattn_names == {"attn2.weight"}
    assert attn_names == {"attn2.weight"}
    assert resnet_names == {"resblock.weight", "nested_block.conv1.weight"}
    assert attn_names.isdisjoint(resnet_names)
    assert attn_names | resnet_names < unet_names
    assert "input_blocks.0.weight" in unet_names - (attn_names | resnet_names)


def test_unet_change_stats_can_follow_train_method_instead_of_dash_target(tmp_path):
    model = TinySD()
    checkpoint = {
        "state_dict": {
            f"model.diffusion_model.{name}": param.detach().clone()
            for name, param in model.model.diffusion_model.named_parameters()
        }
    }
    ckpt_path = tmp_path / "base.ckpt"
    torch.save(checkpoint, ckpt_path)

    with torch.no_grad():
        model.model.diffusion_model.attn2.weight.add_(1.0)

    dash_target_stats = compute_unet_change_stats(
        model,
        ckpt_path,
        dash_config={"target": "unet_resnet"},
        prefix="dash_target",
    )
    train_target_stats = compute_unet_change_stats(
        model,
        ckpt_path,
        dash_config={"target": "unet_resnet"},
        prefix="train_target",
        train_method="xattn",
    )

    assert dash_target_stats["dash_target_target_selector"] == "unet_resnet"
    assert dash_target_stats["dash_target_delta_norm"] == 0.0
    assert train_target_stats["train_target_target_selector"] == "unet_xattn"
    assert train_target_stats["train_target_delta_norm"] > 0.0


def test_unet_change_stats_can_measure_since_post_dash_baseline():
    model = TinySD()
    with torch.no_grad():
        model.model.diffusion_model.resblock.weight.mul_(0.5)
    post_dash = snapshot_unet_change_baseline(
        model,
        dash_config={"target": "unet_resnet"},
    )

    no_unlearn_stats = compute_unet_change_stats_from_baseline(
        model,
        post_dash,
        dash_config={"target": "unet_resnet"},
        prefix="after_unlearn_vs_after_dash",
    )
    with torch.no_grad():
        model.model.diffusion_model.resblock.weight.add_(1.0)
    unlearn_stats = compute_unet_change_stats_from_baseline(
        model,
        post_dash,
        dash_config={"target": "unet_resnet"},
        prefix="after_unlearn_vs_after_dash",
    )

    assert no_unlearn_stats["after_unlearn_vs_after_dash_delta_norm"] == 0.0
    assert unlearn_stats["after_unlearn_vs_after_dash_delta_norm"] > 0.0


def test_named_parameter_change_stats_measure_actual_trainable_params(tmp_path):
    model = TinySD()
    named_params = [("attn2.weight", model.model.diffusion_model.attn2.weight)]
    checkpoint = {
        "state_dict": {
            f"model.diffusion_model.{name}": param.detach().clone()
            for name, param in model.model.diffusion_model.named_parameters()
        }
    }
    ckpt_path = tmp_path / "base.ckpt"
    torch.save(checkpoint, ckpt_path)
    post_dash = snapshot_named_parameter_baseline(named_params)

    with torch.no_grad():
        model.model.diffusion_model.attn2.weight.add_(1.0)

    base_stats = compute_named_parameter_change_stats(
        named_params,
        ckpt_path=ckpt_path,
        prefix="after_unlearn_vs_base",
        selector="intact_trainable",
    )
    dash_stats = compute_named_parameter_change_stats(
        named_params,
        baseline=post_dash,
        prefix="after_unlearn_vs_after_dash",
        selector="intact_trainable",
    )

    assert base_stats["after_unlearn_vs_base_target_selector"] == "intact_trainable"
    assert base_stats["after_unlearn_vs_base_delta_norm"] > 0.0
    assert dash_stats["after_unlearn_vs_after_dash_delta_norm"] > 0.0


def test_dash_projection_precedes_final_svd_truncation(monkeypatch):
    events = []
    original_project = dash_sd_runtime._project_perp_dict
    original_truncate_dict = dash_sd_runtime._truncate_gradient_dict

    def wrapped_project(*args, **kwargs):
        events.append("project")
        return original_project(*args, **kwargs)

    def wrapped_truncate_dict(*args, **kwargs):
        events.append("truncate_dict")
        return original_truncate_dict(*args, **kwargs)

    monkeypatch.setattr(dash_sd_runtime, "_project_perp_dict", wrapped_project)
    monkeypatch.setattr(dash_sd_runtime, "_truncate_gradient_dict", wrapped_truncate_dict)

    torch.manual_seed(5)
    model = TinySD()
    retain, forget, descriptions = _loaders()

    run_dash_sd_warm_start(
        model=model,
        retain_loader=retain,
        forget_loader=forget,
        descriptions=descriptions,
        dash_config=_config(
            signal_mode="preserve_complement",
            plasticity_granularity="global",
            svd_truncate_evr=0.95,
        ),
    )

    assert events.index("project") < events.index("truncate_dict")


def test_attention_head_wise_false_preserves_per_filter_layout():
    model = TinyAttentionSD()
    targets = select_unet_dash_params(model, dash_target="unet_xattn", include_bias=False)
    cfg = dash_sd_runtime.DashSDConfig(warm_start=True, plasticity_granularity="per_filter", attention_head_wise=False)
    layouts, stats = dash_sd_runtime._build_unit_layouts(model, targets, cfg)

    assert stats["attention_headwise_tensor_count"] == 0.0
    assert layouts["block.attn2.to_q.weight"].mode == "per_filter"
    assert layouts["block.attn2.to_out.0.weight"].mode == "per_filter"


def test_attention_head_wise_qkv_use_row_grouped_heads_and_to_out_uses_columns():
    model = TinyAttentionSD()
    targets = select_unet_dash_params(model, dash_target="unet_xattn", include_bias=False)
    cfg = dash_sd_runtime.DashSDConfig(warm_start=True, plasticity_granularity="global", attention_head_wise=True)
    layouts, stats = dash_sd_runtime._build_unit_layouts(model, targets, cfg)

    assert stats["attention_headwise_tensor_count"] == 4.0
    q_weight = model.model.diffusion_model.block.attn2.to_q.weight.detach()
    q_layout = layouts["block.attn2.to_q.weight"]
    q_matrix = dash_sd_runtime._as_layout_matrix(q_weight, q_layout)
    assert q_layout.mode == "head_rows"
    assert torch.equal(q_matrix[0], q_weight[:2, :].reshape(-1))
    assert torch.equal(q_matrix[1], q_weight[2:4, :].reshape(-1))

    out_weight = model.model.diffusion_model.block.attn2.to_out[0].weight.detach()
    out_layout = layouts["block.attn2.to_out.0.weight"]
    out_matrix = dash_sd_runtime._as_layout_matrix(out_weight, out_layout)
    assert out_layout.mode == "head_cols"
    assert torch.equal(out_matrix[0], out_weight[:, :2].reshape(-1))
    assert torch.equal(out_matrix[1], out_weight[:, 2:4].reshape(-1))


def test_attention_head_wise_runs_for_unet_xattn_without_shape_errors():
    torch.manual_seed(7)
    model = TinyAttentionSD()
    retain, forget, descriptions = _loaders()

    stats = run_dash_sd_warm_start(
        model=model,
        retain_loader=retain,
        forget_loader=forget,
        descriptions=descriptions,
        dash_config=_config(
            target="unet_xattn",
            signal_mode="retain_only",
            plasticity_granularity="global",
            attention_head_wise=True,
        ),
    )

    assert stats["dash_sd_enabled"] == 1.0
    assert stats["dash_sd_attention_headwise_tensor_count"] == 4.0
    assert stats["dash_sd_attention_headwise_fallback_tensor_count"] == 0.0


def test_attention_head_wise_unknown_heads_falls_back_safely():
    torch.manual_seed(8)
    model = TinyAttentionSD(unknown_heads=True)
    retain, forget, descriptions = _loaders()

    stats = run_dash_sd_warm_start(
        model=model,
        retain_loader=retain,
        forget_loader=forget,
        descriptions=descriptions,
        dash_config=_config(
            target="unet_xattn",
            signal_mode="retain_only",
            plasticity_granularity="per_filter",
            attention_head_wise=True,
        ),
    )

    assert stats["dash_sd_enabled"] == 1.0
    assert stats["dash_sd_attention_headwise_tensor_count"] == 0.0
    assert stats["dash_sd_attention_headwise_fallback_tensor_count"] == 1.0
    assert stats["dash_sd_attention_headwise_fallback_missing_num_heads"] == 1.0


def test_include_bias_controls_bias_selection():
    model = TinySD()
    no_bias = {target.name for target in select_unet_dash_params(model, dash_target="unet", include_bias=False)}
    with_bias = {target.name for target in select_unet_dash_params(model, dash_target="unet", include_bias=True)}

    assert "input_blocks.0.bias" not in no_bias
    assert "input_blocks.0.bias" in with_bias
