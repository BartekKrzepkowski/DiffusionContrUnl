import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_LABEL_PATH = REPO_ROOT / "SD" / "train-scripts" / "random_label.py"
TRAIN_SCRIPTS_PATH = REPO_ROOT / "SD" / "train-scripts"


def _install_random_label_stubs(monkeypatch):
    convert_models = types.ModuleType("convertModels")
    convert_models.savemodelDiffusers = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "convertModels", convert_models)

    dataset = types.ModuleType("dataset")
    dataset.setup_class_forgetting_data = lambda *args, **kwargs: None
    dataset.setup_model = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dataset", dataset)

    dash_sd_runtime = types.ModuleType("dash_sd_runtime")
    dash_sd_runtime.run_dash_sd_warm_start = lambda *args, **kwargs: {"dash_sd_enabled": 0.0}
    monkeypatch.setitem(sys.modules, "dash_sd_runtime", dash_sd_runtime)

    training_eval = types.ModuleType("training_eval")
    training_eval.compute_unet_change_stats = lambda *args, **kwargs: {}
    training_eval.compute_unet_change_stats_from_baseline = lambda *args, **kwargs: {}
    training_eval.run_training_eval = lambda *args, **kwargs: None
    training_eval.should_run_train_eval = lambda *args, **kwargs: False
    training_eval.snapshot_unet_change_baseline = lambda *args, **kwargs: {}
    training_eval.write_metric_dict = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "training_eval", training_eval)

    diffusers = types.ModuleType("diffusers")

    class LMSDiscreteScheduler:
        def __init__(self, *args, **kwargs):
            pass

    diffusers.LMSDiscreteScheduler = LMSDiscreteScheduler
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)


def _import_random_label(monkeypatch):
    _install_random_label_stubs(monkeypatch)
    monkeypatch.syspath_prepend(str(TRAIN_SCRIPTS_PATH))
    spec = importlib.util.spec_from_file_location("random_label_under_test", RANDOM_LABEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_random_label_module_imports_without_sd_runtime_dependencies(monkeypatch):
    module = _import_random_label(monkeypatch)

    assert hasattr(module, "certain_label")
    torch.manual_seed(12)
    pseudo_labels = module._sample_pseudo_labels(20, [0, 3], ["zero", "one", "two", "three"])
    assert set(pseudo_labels) <= {1, 2}
    assert len(pseudo_labels) == 20


def test_random_label_pseudo_labels_are_seeded_and_exclude_forget_classes(monkeypatch):
    module = _import_random_label(monkeypatch)
    descriptions = [f"class_{idx}" for idx in range(10)]
    forget_indices = [0, 3, 7]

    torch.manual_seed(123)
    first = module._sample_pseudo_labels(64, forget_indices, descriptions)
    torch.manual_seed(123)
    second = module._sample_pseudo_labels(64, forget_indices, descriptions)
    torch.manual_seed(124)
    third = module._sample_pseudo_labels(64, forget_indices, descriptions)

    assert first == second
    assert first != third
    assert set(first).isdisjoint(set(forget_indices))
    assert set(first) <= {1, 2, 4, 5, 6, 8, 9}


def test_random_label_rl_loss_mode_aliases_and_validation(monkeypatch):
    module = _import_random_label(monkeypatch)

    assert module._normalize_rl_loss_mode(None) == "output_matching"
    assert module._normalize_rl_loss_mode("current") == "output_matching"
    assert module._normalize_rl_loss_mode("denoise-pseudo") == "denoise_pseudo"

    try:
        module._normalize_rl_loss_mode("not_a_mode")
    except ValueError as exc:
        assert "output_matching" in str(exc)
        assert "denoise_pseudo" in str(exc)
    else:
        raise AssertionError("invalid rl_loss_mode should raise ValueError")


def test_random_label_collects_grouped_unet_gradient_norms(monkeypatch):
    module = _import_random_label(monkeypatch)

    class ResBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 1)

    class ToyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_blocks = torch.nn.ModuleList([ResBlock()])
            self.middle_block = ResBlock()
            self.output_blocks = torch.nn.ModuleList([ResBlock()])
            self.attn1 = torch.nn.Linear(2, 2)
            self.attn2 = torch.nn.Linear(2, 2)

    class Wrapper:
        pass

    model = Wrapper()
    model.model = Wrapper()
    model.model.diffusion_model = ToyUNet()
    selected_param_ids = set()
    for name, param in model.model.diffusion_model.named_parameters():
        param.grad = torch.ones_like(param)
        if name.startswith("input_blocks") or name.startswith("attn2"):
            selected_param_ids.add(id(param))

    payload = module._collect_unet_grad_norms(model, selected_param_ids)

    assert "grad_norm____unet/total____train" in payload
    assert "grad_norm____unet/resnet____train" in payload
    assert "grad_norm____unet/cross_attn____train" in payload
    assert "grad_norm____unet/resnet_stage/input_blocks____train" in payload
    assert "grad_peak_abs____unet/resnet____train" in payload
    assert "grad_peak_abs____unet/cross_attn____train" in payload
    assert "grad_norm____unet/self_attn____train" not in payload
    assert "grad_norm____unet/resnet_stage/middle_block____train" not in payload
    assert "grad_norm____unet/resnet_stage/output_blocks____train" not in payload
    assert payload["grad_norm____unet/log_event____train"] == 1.0


def test_random_label_logs_gradient_norms_evenly_within_epoch(monkeypatch):
    module = _import_random_label(monkeypatch)

    log_steps = [step for step in range(10) if module._should_log_grad_norm(step, epoch_batches=10, logs_per_epoch=3)]

    assert log_steps == [0, 4, 8]
    assert not module._should_log_grad_norm(0, epoch_batches=10, logs_per_epoch=0)
    assert module._grad_norm_log_points(epoch_batches=100, logs_per_epoch=3) == {0, 49, 98}


def test_random_label_retain_iterator_cycles(monkeypatch):
    module = _import_random_label(monkeypatch)
    loader = DataLoader(TensorDataset(torch.tensor([10, 11]), torch.tensor([0, 1])), batch_size=1)

    iterator = iter(loader)
    (first_x, _), iterator = module._next_or_restart(iterator, loader)
    (second_x, _), iterator = module._next_or_restart(iterator, loader)
    (third_x, _), iterator = module._next_or_restart(iterator, loader)

    assert first_x.item() == 10
    assert second_x.item() == 11
    assert third_x.item() == 10


def test_random_label_optional_iterator_does_not_cycle(monkeypatch):
    module = _import_random_label(monkeypatch)
    loader = DataLoader(TensorDataset(torch.tensor([10, 11]), torch.tensor([0, 1])), batch_size=1)

    iterator = iter(loader)
    (first_x, _), iterator = module._next_or_none(iterator)
    (second_x, _), iterator = module._next_or_none(iterator)
    third, iterator = module._next_or_none(iterator)

    assert first_x.item() == 10
    assert second_x.item() == 11
    assert third is None


def test_random_label_full_retain_epoch_count_and_stats(monkeypatch):
    module = _import_random_label(monkeypatch)
    retain_loader = DataLoader(
        TensorDataset(torch.arange(5), torch.arange(5)),
        batch_size=1,
    )
    forget_loader = DataLoader(
        TensorDataset(torch.arange(2), torch.arange(2)),
        batch_size=1,
    )

    default_stats = module._loader_data_stats(
        retain_loader,
        forget_loader,
        dash_config={"warm_start": False},
        full_retain_per_epoch=False,
    )
    full_retain_stats = module._loader_data_stats(
        retain_loader,
        forget_loader,
        dash_config={"warm_start": False},
        full_retain_per_epoch=True,
    )

    assert module._unlearn_epoch_batch_count(retain_loader, forget_loader) == 2
    assert module._unlearn_epoch_batch_count(
        retain_loader,
        forget_loader,
        full_retain_per_epoch=True,
    ) == 5
    assert default_stats["unlearn_retain_full_pass_per_epoch"] == 0.0
    assert full_retain_stats["unlearn_retain_full_pass_per_epoch"] == 1.0
    assert full_retain_stats["unlearn_steps_per_epoch"] == 5.0
    assert full_retain_stats["unlearn_forget_batches_per_epoch"] == 2.0
    assert full_retain_stats["unlearn_retain_batches_per_epoch"] == 5.0
    assert full_retain_stats["unlearn_smaller_loader_cycles"] == 0.0


def test_random_label_run_name_includes_seed_and_dash_params(monkeypatch):
    _import_random_label(monkeypatch)
    from run_naming import build_sd_unlearn_name

    name = build_sd_unlearn_name(
        setting="sd",
        uc={
            "method": "rl",
            "class_to_forget": 0,
            "train_method": "xattn",
            "alpha": 1.0,
            "epochs": 10,
            "lr": 0.05,
            "full_retain_per_epoch": True,
        },
        dash_cfg={
            "warm_start": True,
            "target": "unet_attn",
            "signal_mode": "retain_only",
            "plasticity_granularity": "global",
            "grad_aggregation": "mean",
            "num_aug": 2,
            "min_shrink": 0.02,
            "svd_truncate_evr": 0.95,
            "retain_batches": 8,
            "forget_batches": 0,
        },
        seed=123,
    )

    assert "seed_123" in name
    assert "fullret_true" in name
    assert "dt_unet_attn" in name
    assert "sig_retain_only" in name
    assert "ms_0.02" in name


def test_random_label_run_name_can_include_rl_loss_mode(monkeypatch):
    _import_random_label(monkeypatch)
    from run_naming import build_sd_unlearn_name

    name = build_sd_unlearn_name(
        setting="sd",
        uc={
            "method": "rl",
            "class_to_forget": 0,
            "train_method": "xattn",
            "alpha": 1.0,
            "epochs": 1,
            "lr": 0.05,
            "rl_loss_mode": "denoise_pseudo",
        },
        dash_cfg={"warm_start": False},
        seed=123,
    )

    assert "rlmode_denoise_pseudo" in name


def test_random_label_run_name_marks_no_dash(monkeypatch):
    _import_random_label(monkeypatch)
    from run_naming import build_sd_unlearn_name

    name = build_sd_unlearn_name(
        setting="sd",
        uc={
            "method": "rl",
            "class_to_forget": 0,
            "train_method": "xattn",
            "alpha": 1.0,
            "epochs": 1,
            "lr": 0.05,
        },
        dash_cfg={"warm_start": False},
        seed=123,
    )

    assert "dash_off" in name
    assert "dash_on" not in name
