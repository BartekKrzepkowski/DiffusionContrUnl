from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_intact_and_nsfw_loops_do_not_reset_dataloader_iterators():
    intact_source = _read("SD/train-scripts/intact_unlearn.py")
    nsfw_source = _read("SD/train-scripts/nsfw_removal.py")
    proximal_source = _read("SD/train-scripts/proximal_gradient.py")

    assert "next(iter(forget_dl))" not in intact_source
    assert "next(iter(remain_dl))" not in intact_source
    assert "next(iter(forget_dl))" not in nsfw_source
    assert "next(iter(remain_dl))" not in nsfw_source
    assert "next(iter(forget_dl))" not in proximal_source
    assert "next(iter(remain_dl))" not in proximal_source


def test_generate_images_fails_instead_of_evaluating_base_unet_on_load_error():
    source = _read("SD/eval-scripts/generate-images.py")
    pipeline_source = _read("SD/pipeline.py")

    assert "Using base UNet instead" not in source
    assert "refusing to evaluate the base UNet" in source
    assert "def require_diffusers_checkpoint" in pipeline_source
    assert "Expected unlearned diffusers checkpoint is missing" in pipeline_source


def test_unlearning_scripts_validate_empty_trainable_parameter_selection():
    for relative_path in [
        "SD/train-scripts/random_label.py",
        "SD/train-scripts/gradient_ascent.py",
        "SD/train-scripts/nsfw_removal.py",
        "SD/train-scripts/proximal_gradient.py",
    ]:
        source = _read(relative_path)
        assert "No trainable parameters selected" in source


def test_alpha_weights_forget_loss_in_ga_and_rl():
    random_label_source = _read("SD/train-scripts/random_label.py")
    gradient_ascent_source = _read("SD/train-scripts/gradient_ascent.py")
    intact_source = _read("SD/train-scripts/intact_unlearn.py")

    assert "loss_terms.append(remain_loss)" in random_label_source
    assert "loss_terms.append(alpha * forget_loss)" in random_label_source
    assert "loss_terms.append(remain_loss)" in gradient_ascent_source
    assert "loss_terms.append(alpha * forget_loss)" in gradient_ascent_source
    assert "loss_terms.append(remain_loss)" in intact_source
    assert "loss_terms.append(alpha * forget_loss)" in intact_source


def test_random_label_supports_switchable_rl_forget_loss_and_logs_terms():
    random_label_source = _read("SD/train-scripts/random_label.py")
    pipeline_source = _read("SD/pipeline.py")
    config_source = _read("SD/configs/pipeline_class.yaml")
    slurm_source = _read("SD/scripts/slurm_sd_dash_rl.sh")

    assert "def _normalize_rl_loss_mode" in random_label_source
    assert 'rl_loss_mode == "output_matching"' in random_label_source
    assert 'rl_loss_mode == "denoise_pseudo"' in random_label_source
    assert "forget_loss = criteria(pseudo_out, noise)" in random_label_source
    assert '"loss____train/weighted_forget____step"' in random_label_source
    assert '"weighted_forget_loss"' in random_label_source
    assert 'rl_loss_mode=uc.get("rl_loss_mode", "output_matching")' in pipeline_source
    assert "rl_loss_mode:" in config_source
    assert "--rl-loss-mode" in slurm_source


def test_sd_wandb_config_and_grouped_gradient_norm_logging_are_wired():
    random_label_source = _read("SD/train-scripts/random_label.py")
    pipeline_source = _read("SD/pipeline.py")
    config_source = _read("SD/configs/pipeline_class.yaml")
    slurm_source = _read("SD/scripts/slurm_sd_dash_rl.sh")

    assert "def resolve_wandb_settings" in pipeline_source
    assert "def build_wandb_tags" in pipeline_source
    assert '"method:{unlearn_cfg.get' in pipeline_source
    assert '"dash:{' in pipeline_source
    assert '"dash_min_shrink:{dash_cfg.get' in pipeline_source
    assert '"mode": wandb_settings["mode"]' in pipeline_source
    assert "use_wandb: true" in config_source
    assert 'project: "stable_diff_dash"' in config_source
    assert 'mode: "online"' in config_source
    assert "log_grad_norms: true" in config_source
    assert "grad_norm_log_interval: 3" in config_source
    assert "def _collect_unet_grad_norms" in random_label_source
    assert '"grad_norm____unet/{key}____train"' in random_label_source
    assert '"grad_peak_abs____unet/{key}____train"' in random_label_source
    assert '"grad_norm____unet/log_event____train"' in random_label_source
    assert '"loss____train/total____step"' in random_label_source
    assert '"loss____train/{key}____{prefix}"' in random_label_source
    assert '"resnet_stage/input_blocks"' in random_label_source
    dash_runtime_source = _read("SD/train-scripts/dash_sd_runtime.py")
    assert "def _wandb_dash_sd_payload" in dash_runtime_source
    assert "def _wandb_dash_sd_histogram_payload" in dash_runtime_source
    assert '"dash_sd____alignment/cosine_histogram____warm_start"' in dash_runtime_source
    assert '"dash_sd____alignment_cdf"' in dash_runtime_source
    assert '"dash_sd____alignment_module/{module_key}"' in dash_runtime_source
    assert "/cosine_histogram____warm_start" in dash_runtime_source
    assert '"{module_prefix}/cdf"' in dash_runtime_source
    assert '"{module_prefix}/median____warm_start"' in dash_runtime_source
    assert 'USE_WANDB="${USE_WANDB:-config}"' in slurm_source
    assert 'PIPELINE_ARGS+=(--no-wandb)' in slurm_source


def test_roft_method_skips_forget_set_during_main_unlearning_but_keeps_dash_loader():
    pipeline_source = _read("SD/pipeline.py")
    random_label_source = _read("SD/train-scripts/random_label.py")
    naming_source = _read("SD/train-scripts/run_naming.py")

    assert 'method in {"rl", "roft"}' in pipeline_source
    assert "def retain_only_finetune" in random_label_source
    assert "kwargs[\"use_forget_in_unlearn\"] = False" in random_label_source
    assert "run_dash_sd_warm_start" in random_label_source
    assert "unlearn_uses_forget_set" in random_label_source
    assert 'method in {"rl", "roft"}' in naming_source


def test_class_forgetting_forget_loader_shuffle_is_explicit_and_logged():
    source = _read("SD/train-scripts/dataset.py")

    assert "forget_shuffle=True" in source
    assert "retain_shuffle=%s forget_shuffle=%s" in source
    assert "shuffle=bool(forget_shuffle)" in source


def test_train_eval_has_generation_cost_guard_and_epoch_scoped_outputs():
    source = _read("SD/train-scripts/training_eval.py")
    config_source = _read("SD/configs/pipeline_class.yaml")
    intact_source = _read("SD/train-scripts/intact_unlearn.py")
    random_label_source = _read("SD/train-scripts/random_label.py")
    gradient_ascent_source = _read("SD/train-scripts/gradient_ascent.py")
    pipeline_source = _read("SD/pipeline.py")

    assert "return epoch % interval == 0" in source
    assert "def should_run_pre_epoch_train_eval" in source
    assert "max_generated_images" in source
    assert "train_eval would generate too many images" in source
    assert "eval_images_dir" in source
    assert "eval_checkpoint" in source
    assert "base_model_path=ckpt_path" in source
    assert "base_config_path=config_path" in source
    assert "Training eval diffusers export failed or wrote no checkpoint" in source
    assert "keep_checkpoints" in source
    assert "shutil.rmtree(checkpoint_path.parent" in source
    assert "def compute_unlearning_accuracy" in source
    assert "def compute_fid_score" in source
    assert "def _log_epoch_metrics_to_wandb" in source
    assert '"eval_epoch____FID____train"' in source
    assert '"eval_epoch____UA____train"' in source
    assert "run_before_first_epoch:" in config_source
    assert "interval_epochs: 3" in config_source
    assert "ua: true" in config_source
    assert "fid:" in config_source
    assert "run_training_eval(" in intact_source
    assert "should_run_pre_epoch_train_eval(train_eval_config)" in intact_source
    assert intact_source.index("should_run_pre_epoch_train_eval(train_eval_config)") < intact_source.index("run_dash_sd_warm_start(")
    assert "should_run_pre_epoch_train_eval(train_eval_config)" in random_label_source
    assert random_label_source.index("should_run_pre_epoch_train_eval(train_eval_config)") < random_label_source.index("run_dash_sd_warm_start(")
    assert "should_run_pre_epoch_train_eval(train_eval_config)" in gradient_ascent_source
    assert gradient_ascent_source.index("should_run_pre_epoch_train_eval(train_eval_config)") < gradient_ascent_source.index("run_dash_sd_warm_start(")
    assert "should_run_train_eval(epoch, epochs, train_eval_config)" in intact_source
    assert "_log_wandb_scalars(" in intact_source
    assert "_collect_unet_grad_norms(model, selected_param_ids)" in intact_source
    assert "_loss_mean_payload(epoch_loss_accumulator, \"running\")" in intact_source
    assert "rl_loss_mode=rl_loss_mode" in intact_source
    assert "train_eval_config=train_eval_cfg" in pipeline_source
    assert 'wandb_log_interval=uc.get("wandb_log_interval", 1)' in pipeline_source
    assert 'rl_loss_mode=uc.get("rl_loss_mode", "output_matching")' in pipeline_source
    assert 'log_grad_norms=uc.get("log_grad_norms", True)' in pipeline_source
    assert "def _sanitize_wandb_tag" in pipeline_source
    assert "max_len=64" in pipeline_source
    assert "_sanitize_wandb_tag(tag)" in pipeline_source


def test_pipeline_eval_uses_configured_local_sd_base_model():
    source = _read("SD/pipeline.py")

    assert "base_model_path=cfg[\"paths\"].get(\"sd_ckpt\"" in source
    assert "base_config_path=cfg[\"paths\"].get(\"sd_config\")" in source


def test_sd_checkpoint_load_mismatches_are_logged():
    dataset_source = _read("SD/train-scripts/dataset.py")
    intact_source = _read("SD/train-scripts/intact_unlearn.py")
    generate_source = _read("SD/eval-scripts/generate-images.py")

    for source in [dataset_source, intact_source]:
        assert "Missing keys while loading SD checkpoint" in source
        assert "Unexpected keys while loading SD checkpoint" in source
    assert "weights_only=False" in generate_source


def test_full_retain_per_epoch_uses_classification_style_noncycling_smaller_loader():
    random_label_source = _read("SD/train-scripts/random_label.py")
    gradient_ascent_source = _read("SD/train-scripts/gradient_ascent.py")
    intact_source = _read("SD/train-scripts/intact_unlearn.py")
    pipeline_source = _read("SD/pipeline.py")
    config_source = _read("SD/configs/pipeline_class.yaml")

    for source in [random_label_source, gradient_ascent_source, intact_source]:
        assert "def _next_or_none" in source
        assert "if full_retain_per_epoch:" in source
        assert "forget_batch_data, forget_iter = _next_or_none(forget_iter)" in source
    for source in [random_label_source, gradient_ascent_source]:
        assert "unlearn_smaller_loader_cycles" in source
    assert 'full_retain_per_epoch=uc.get("full_retain_per_epoch", False)' in pipeline_source
    assert "without cycling exhausted loaders" in config_source


def test_sd_dataset_uses_index_subsets_for_class_splits():
    source = _read("SD/train-scripts/dataset.py")

    assert "[data for data" not in source
    assert "Subset(train_set" in source
    assert "def _label_indices" in source


def test_sd_pipeline_blocks_multi_class_eval_until_metrics_are_extended():
    source = _read("SD/pipeline.py")

    assert "def validate_sd_single_class_eval" in source
    assert "currently supports exactly one forget class/concept" in source
    assert "compute_ua_class/compute_fid_sd/log_sample_images_per_class" in source


def test_final_unlearn_delta_stats_follow_train_method_not_dash_target():
    training_eval_source = _read("SD/train-scripts/training_eval.py")
    random_label_source = _read("SD/train-scripts/random_label.py")
    gradient_ascent_source = _read("SD/train-scripts/gradient_ascent.py")

    assert "def _change_stats_target_from_train_method" in training_eval_source
    assert 'train_method == "xattn"' in training_eval_source
    assert 'return "unet_xattn"' in training_eval_source
    assert '"unet_resnet"' in training_eval_source
    assert "select_unet_dash_params" in random_label_source
    assert "select_unet_dash_params" in gradient_ascent_source
    assert "train_method=train_method" in random_label_source
    assert "train_method=train_method" in gradient_ascent_source
    assert "snapshot_unet_change_baseline" in random_label_source
    assert "snapshot_unet_change_baseline" in gradient_ascent_source
    assert "after_unlearn_vs_after_dash" in random_label_source
    assert "after_unlearn_vs_after_dash" in gradient_ascent_source


def test_sd_pipeline_defaults_train_method_to_dash_target_when_dash_is_enabled():
    pipeline_source = _read("SD/pipeline.py")
    config_source = _read("SD/configs/pipeline_class.yaml")

    assert "def resolve_sd_train_method" in pipeline_source
    assert 'if dash_cfg.get("warm_start", False):' in pipeline_source
    assert 'return dash_cfg.get("target", "unet")' in pipeline_source
    assert 'train_method=uc.get("train_method", "xattn")' not in pipeline_source
    assert "null" in config_source
    assert "train dash.target" in config_source


def test_sd_dash_slurm_scripts_switch_between_class_and_nsfw_headwise_runs():
    launcher_source = _read("SD/scripts/slurm_sd_dash_rl.sh")
    sweep_source = _read("SD/scripts/slurm_sd_dash_lr_minshrink_sweep.sh")

    assert "--setting NAME" in launcher_source
    assert "SD/configs/pipeline_nsfw_dash.yaml" in launcher_source
    assert "--dash-attention-head-wise" in launcher_source
    assert "--no-dash-attention-head-wise" in launcher_source
    assert 'cfg["pipeline"]["setting"] = "sd_nsfw"' in launcher_source
    assert 'cfg["pipeline"]["setting"] = "sd"' in launcher_source
    assert '("dash", "attention_head_wise")' in launcher_source
    assert 'experiment_name = str(experiment_slug).split("/", 1)[0]' in launcher_source
    assert 'f"experiment_{tag_value(experiment_name)}"' in launcher_source
    assert 'f"experiment_{tag_value(experiment_slug)}"' not in launcher_source
    assert 'f"method_{method_tag}"' in launcher_source
    assert 'f"dash_{dash_enabled_tag}"' in launcher_source
    assert 'f"dash_target_' in launcher_source
    assert 'f"attention_headwise_{headwise_tag}"' in launcher_source
    assert 'f"min_shrink_' in launcher_source

    assert "DASH_ATTENTION_HEAD_WISE_VALUES" in sweep_source
    assert "--dash-attention-head-wise" in sweep_source
    assert "--no-dash-attention-head-wise" in sweep_source
    assert "headwise_on" in sweep_source
    assert "headwise_off" in sweep_source


def test_intact_uses_shared_pipeline_run_naming():
    intact_source = _read("SD/train-scripts/intact_unlearn.py")

    assert "from run_naming import build_sd_unlearn_name" in intact_source
    assert "name = build_sd_unlearn_name(" in intact_source
    assert "compvis-intact-{base_method}-class_" not in intact_source


def test_intact_logs_dash_and_final_trainable_delta_stats():
    intact_source = _read("SD/train-scripts/intact_unlearn.py")

    assert "dash_warm_start_stats.json" in intact_source
    assert "final_unlearn_delta_stats.json" in intact_source
    assert "snapshot_named_parameter_baseline" in intact_source
    assert "compute_named_parameter_change_stats" in intact_source
    assert "selector=\"intact_trainable\"" in intact_source


def test_final_sd_pipeline_logs_clip_and_retain_accuracy():
    pipeline_source = _read("SD/pipeline.py")

    assert "compute_sd_clip_score" in pipeline_source
    assert "compute_per_class_retain_accuracy" in pipeline_source
    assert 'metrics["CLIP"]' in pipeline_source
    assert "retain_acc/mean" in pipeline_source
    assert "run_metadata.json" in pipeline_source
    assert "diffusers_checkpoint_exists" in pipeline_source


def test_intact_esd_rejects_empty_trainable_parameter_selection():
    intact_source = _read("SD/train-scripts/intact_unlearn.py")

    esd_start = intact_source.index("def intact_unlearn_esd")
    esd_source = intact_source[esd_start:]
    assert "if not trainable_params:" in esd_source
    assert "InTAct selected no trainable parameters" in esd_source


def test_sd_unlearning_loops_do_not_use_inference_scheduler_or_artificial_sleep():
    for relative_path in [
        "SD/train-scripts/random_label.py",
        "SD/train-scripts/gradient_ascent.py",
        "SD/train-scripts/intact_unlearn.py",
        "SD/train-scripts/nsfw_removal.py",
        "SD/train-scripts/proximal_gradient.py",
    ]:
        source = _read(relative_path)
        assert "LMSDiscreteScheduler" not in source
        assert "scheduler = LMSDiscreteScheduler" not in source
        assert "sleep(" not in source


def test_sd_unlearning_loss_logging_keeps_model_mean_loss_scale():
    for relative_path in [
        "SD/train-scripts/random_label.py",
        "SD/train-scripts/gradient_ascent.py",
        "SD/train-scripts/intact_unlearn.py",
        "SD/train-scripts/nsfw_removal.py",
        "SD/train-scripts/proximal_gradient.py",
    ]:
        source = _read(relative_path)
        assert "/ batch_size" not in source
        assert "values = np.asarray(a, dtype=float)" in source


def test_sd_class_unlearning_averages_available_loss_terms_per_step():
    for relative_path in [
        "SD/train-scripts/random_label.py",
        "SD/train-scripts/gradient_ascent.py",
        "SD/train-scripts/intact_unlearn.py",
    ]:
        source = _read(relative_path)
        assert "sum(loss_terms) / len(loss_terms)" in source


def test_sd_dash_slurm_wrapper_does_not_override_config_method_by_default():
    source = _read("SD/scripts/slurm_sd_dash_rl.sh")

    assert 'UNLEARN_METHOD="${UNLEARN_METHOD:-}"' in source
    assert 'method_override = os.environ.get("UNLEARN_METHOD", "")' in source
    assert 'if method_override:' in source
    assert 'cfg["unlearn"]["method"] = os.environ.get("UNLEARN_METHOD", "rl")' not in source


def test_sd_save_model_checks_diffusers_export_artifact_exists():
    for relative_path in [
        "SD/train-scripts/random_label.py",
        "SD/train-scripts/gradient_ascent.py",
        "SD/train-scripts/intact_unlearn.py",
        "SD/train-scripts/nsfw_removal.py",
        "SD/train-scripts/proximal_gradient.py",
    ]:
        source = _read(relative_path)
        assert "Diffusers export failed or wrote no checkpoint" in source
        assert "os.path.exists(diffusers_path)" in source


def test_sd_final_checkpoint_save_is_config_gated_for_rl():
    random_label_source = _read("SD/train-scripts/random_label.py")
    pipeline_source = _read("SD/pipeline.py")
    config_source = _read("SD/configs/pipeline_class.yaml")

    assert "save_final_checkpoint: false" in config_source
    assert "save_final_checkpoint=False" in random_label_source
    assert "Final model checkpoint saved temporarily for final evaluation" in random_label_source
    assert 'uc.get("save_final_checkpoint")' in pipeline_source
    assert "save_final_checkpoint=save_final_checkpoint" in pipeline_source
    assert "def cleanup_temporary_final_checkpoint" in pipeline_source
    assert "atexit.register(cleanup_temporary_final_checkpoint, cfg, model_name)" in pipeline_source
    assert "cleanup_temporary_final_checkpoint(cfg, model_name)" in pipeline_source
    assert "final_checkpoint_retained" in pipeline_source
    assert "Skipping final image generation because unlearn.save_final_checkpoint=false" not in pipeline_source


def test_pipeline_warns_when_dash_config_is_inactive():
    pipeline_source = _read("SD/pipeline.py")

    assert "def warn_if_dash_config_is_inactive" in pipeline_source
    assert "DASH warm_start is false" in pipeline_source
    assert "warn_if_dash_config_is_inactive(cfg)" in pipeline_source


def test_intact_writes_training_history_csv():
    intact_source = _read("SD/train-scripts/intact_unlearn.py")

    assert "history_rows = []" in intact_source
    assert "training_history.csv" in intact_source
    assert '"base_loss": base_loss_value' in intact_source
    assert '"intact_loss": intact_loss_value' in intact_source


def test_sd_nsfw_rl_roft_dash_is_wired_without_intact():
    pipeline_source = _read("SD/pipeline.py")
    random_label_source = _read("SD/train-scripts/random_label.py")
    intact_source = _read("SD/train-scripts/intact_unlearn.py")
    config_source = _read("SD/configs/pipeline_nsfw_dash.yaml")
    naming_source = _read("SD/train-scripts/run_naming.py")

    assert 'method in {"rl", "roft"}' in pipeline_source
    assert "certain_label_nsfw" in pipeline_source
    assert "retain_only_finetune_nsfw" in pipeline_source
    assert "def certain_label_nsfw" in random_label_source
    assert 'rl_loss_mode="output_matching"' in random_label_source
    assert 'uc_for_name["rl_loss_mode"] = rl_loss_mode' in random_label_source
    assert 'rl_loss_mode=uc.get("rl_loss_mode", "output_matching")' in pipeline_source
    assert "def retain_only_finetune_nsfw" in random_label_source
    assert "setup_forget_nsfw_data" in random_label_source
    assert "run_dash_sd_warm_start(" in random_label_source
    assert random_label_source.index("run_dash_sd_warm_start(") < random_label_source.index("optimizer = torch.optim.Adam(parameters, lr=lr)")
    assert "intact_unlearn_nsfw" in pipeline_source
    nsfw_intact_start = intact_source.index("def intact_unlearn_nsfw")
    nsfw_intact_source = intact_source[nsfw_intact_start:]
    assert "run_dash_sd_warm_start(" not in nsfw_intact_source
    assert 'method:' in config_source
    assert "attention_head_wise: true" in config_source
    assert "loss_mode: \"denoise\"" in config_source
    assert "clip_score: true" in config_source
    assert "fid: true" in config_source
    assert 'setting == "sd_nsfw"' in naming_source


def test_sd_nsfw_dataset_generation_and_imagefolder_loader_are_wired():
    dataset_source = _read("SD/train-scripts/dataset.py")
    config_source = _read("SD/configs/pipeline_nsfw_dash.yaml")
    script_source = _read("SD/scripts/slurm_sd_generate_nsfw_data.sh")

    assert "def _load_image_train_dataset" in dataset_source
    assert 'load_dataset("imagefolder", data_dir=data_path)' in dataset_source
    assert "Provide a HuggingFace dataset id/script or an existing image folder" in dataset_source

    assert "/net/storage/pr3/plgrid/plggdnnp/plgkrzepk/SD/data/nsfw" in config_source
    assert "/net/storage/pr3/plgrid/plggdnnp/plgkrzepk/SD/data/not-nsfw" in config_source

    assert "a photo of a nude person" in script_source
    assert "a photo of a person wearing clothes" in script_source
    assert 'staging.glob("0_*.png")' in script_source
    assert 'staging.glob("1_*.png")' in script_source
    assert 'load_dataset("imagefolder", data_dir=path)' in script_source
    assert "DRY_RUN submit" in script_source
    assert "GEN_BATCH_SIZE" in script_source
    assert "n_outer" in script_source


def test_sd_nsfw_final_metrics_include_nudenet_clip_and_probe_fid():
    pipeline_source = _read("SD/pipeline.py")
    config_source = _read("SD/configs/pipeline_nsfw_dash.yaml")

    assert 'metrics["UA"] = ua' in pipeline_source
    assert 'compute_nudenet_i2p(' in pipeline_source
    assert 'cfg["paths"].get("nsfw_prompts"' in pipeline_source
    assert 'compute_sd_clip_score(images_dir, prompts_path, device_str)' in pipeline_source
    assert 'metrics["CLIP_NSFW_PROMPTS"] = clip_score' in pipeline_source
    assert 'compute_fid_nsfw(' in pipeline_source
    assert 'metrics["FID_NSFW_PROBE"] = fid_score' in pipeline_source
    assert 'time____eval_final/nsfw_probe_fid_seconds____final' in pipeline_source

    assert "clip_score: true" in config_source
    assert "probe:" in config_source
    assert "fid: true" in config_source
    assert "coco:" in config_source
    assert "clip: true" in config_source


def test_sd_nsfw_train_eval_logs_clip_fid_ua_on_interval_schedule():
    pipeline_source = _read("SD/pipeline.py")
    random_label_source = _read("SD/train-scripts/random_label.py")
    training_eval_source = _read("SD/train-scripts/training_eval.py")
    config_source = _read("SD/configs/pipeline_nsfw_dash.yaml")

    assert "train_eval:" in config_source
    assert 'setting: "sd_nsfw"' in config_source
    assert "run_before_first_epoch: true" in config_source
    assert "interval_epochs: 3" in config_source
    assert "keep_checkpoints: false" in config_source
    assert "unsafe-prompts-nudity-balanced-train125.csv" in config_source
    assert "max_prompts: 125" in config_source
    assert "max_generated_images: 125" in config_source
    assert "unsafe-prompts-nudity-balanced-final750.csv" in config_source
    assert "max_prompts: 750" in config_source
    assert "not_nsfw_data_path: null" in config_source

    assert 'train_eval_cfg = dict(cfg.get("train_eval", {}) or {})' in pipeline_source
    assert 'train_eval_cfg["prompts_path"] = cfg["paths"].get("nsfw_prompts"' in pipeline_source
    assert 'train_eval_cfg["not_nsfw_data_path"] = cfg["paths"].get("not_nsfw_data"' in pipeline_source
    assert "train_eval_config=train_eval_cfg" in pipeline_source

    assert "train_eval_config=None" in random_label_source
    nsfw_start = random_label_source.index("def certain_label_nsfw")
    nsfw_source = random_label_source[nsfw_start:]
    assert "should_run_pre_epoch_train_eval(train_eval_config)" in nsfw_source
    assert nsfw_source.index("should_run_pre_epoch_train_eval(train_eval_config)") < nsfw_source.index("run_dash_sd_warm_start(")
    assert "should_run_train_eval(epoch, epochs, train_eval_config)" in nsfw_source

    assert "def compute_nsfw_unlearning_accuracy" in training_eval_source
    assert "def compute_nsfw_fid_score" in training_eval_source
    assert 'metrics["CLIP_NSFW_PROMPTS"] = clip_score' in training_eval_source
    assert 'metrics["FID_NSFW_PROBE"] = fid_score' in training_eval_source
    assert 'elif key.startswith("nudenet/"):' in training_eval_source
    assert 'grouped_categories = ["Common", "Female", "Male"]' in training_eval_source
    assert 'NUDENET_CLASS_MAP_GROUPED' in pipeline_source
    assert 'artifact_paths = {' in pipeline_source
    assert 'Generated image directories and artifacts:' in pipeline_source
    assert 'return_group_scores=True' in training_eval_source
    assert 'CLIP/prompt_group/{group}' in training_eval_source
    assert 'UA/prompt_group/{group}' in training_eval_source
    assert 'return_group_scores=True' in pipeline_source
    assert 'UA/prompt_group/{group}' in pipeline_source
    assert "_write_nsfw_probe_prompts" in training_eval_source
    assert "eval_checkpoint_removed" in training_eval_source


def test_sd_dash_attention_head_wise_config_and_logging_are_wired():
    dash_runtime_source = _read("SD/train-scripts/dash_sd_runtime.py")
    class_config_source = _read("SD/configs/pipeline_class.yaml")

    assert "attention_head_wise" in dash_runtime_source
    assert "head_rows" in dash_runtime_source
    assert "head_cols" in dash_runtime_source
    assert "dash_sd_attention_headwise_fallback_tensor_count" in dash_runtime_source or "attention_headwise_fallback_tensor_count" in dash_runtime_source
    assert "dash_sd____alignment_attention/{group_key}" in dash_runtime_source
    assert "dash_sd____update_module/" in dash_runtime_source
    assert "attention_head_wise:" in class_config_source
    assert "attn2.to_out" in class_config_source
