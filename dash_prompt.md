You are working in my machine unlearning repository.

I already have a method called DASH implemented for classification. I want to port the same idea to Stable Diffusion.

The first milestone is class/concept forgetting, not NSFW.

NSFW unlearning is a later milestone. FLUX is also a later milestone. Do not implement FLUX now.

# Main goal

Implement a first working version of DASH warm-start for Stable Diffusion class/concept forgetting.

The implementation should support a setup where:

- forget_loader contains image-prompt pairs from selected visual classes/concepts,
- retain_loader contains image-prompt pairs from all remaining classes/concepts,
- DASH collects retain/forget gradients using the standard Stable Diffusion denoising objective,
- DASH modifies selected Stable Diffusion weights in-place before the main unlearning training,
- the first target is the Stable Diffusion U-Net,
- VAE and text encoder stay frozen in the first version unless the existing code already has a clean option for this.

Do not hard-code NSFW logic in the DASH runtime. The DASH implementation should be generic enough so that NSFW can later be expressed as a special kind of forget set.

# Important: first inspect the repository

Before writing code, inspect the current Stable Diffusion implementation in the repo.

Find and summarize:

1. Stable Diffusion entrypoints:
   - training scripts,
   - unlearning scripts,
   - config files,
   - command-line argument parsers.

2. Model loading:
   - how Stable Diffusion is loaded,
   - whether it uses diffusers,
   - where unet, vae, text_encoder, tokenizer, and noise_scheduler are created,
   - whether accelerate is used.

3. Current loss:
   - where the diffusion denoising loss is computed,
   - whether prediction target is epsilon, v_prediction, or something else,
   - whether the noise scheduler config determines the target.

4. Current data pipeline:
   - how datasets are loaded,
   - whether batches contain images, prompts, labels/classes, metadata,
   - how class/concept splits are represented,
   - whether there is already a class-forgetting or concept-forgetting setup.

5. Current unlearning pipeline:
   - where the main unlearning training loop starts,
   - where a warm-start method should be inserted,
   - whether there are existing hooks for pre-training, pre-unlearning, or stage-boundary operations.

6. Current logging:
   - whether W&B, TensorBoard, plain logs, or custom loggers are used,
   - where metrics should be logged.

7. Current evaluation:
   - whether image generation hooks exist,
   - whether CLIP score, FID, KID, safety checker, or class/concept evaluation exists.

After this inspection, implement DASH in the minimal clean place that fits the repo.

Do not guess paths or APIs. Use the repository structure.

# Ambiguity handling

If implementation details are unclear, do the following:

1. Prefer inspecting the code over asking questions.
2. If multiple reasonable integration points exist, choose the least invasive one and document the choice.
3. If a missing detail does not block implementation, implement a clean default and add a TODO.
4. Ask/block only if the ambiguity makes implementation unsafe or impossible.

Examples of blocking ambiguities:

- there is no Stable Diffusion code in the repo,
- there is no way to construct retain/forget loaders,
- the batch format cannot be inferred,
- the repo does not expose unet, vae, text_encoder, or equivalent objects,
- the current training loop cannot be located.

If blocked, report:

- what you searched,
- what you found,
- what is missing,
- exactly what information is needed.

Do not silently invent a fake Stable Diffusion pipeline.

# Existing DASH behavior to preserve conceptually

The current classification DASH behaves as follows:

- it is a warm-start before the main unlearning training,
- it does not run its own optimizer loop,
- it does not call optimizer.step(),
- it computes gradients from retain data and, depending on signal mode, also forget data,
- it aggregates gradients using either mean or ema,
- it supports signal modes:
  - retain_only,
  - forget_perp_retain,
  - preserve_complement,
- it supports granularity:
  - global,
  - per_filter,
- it supports EVR truncation:
  - dash_svd_truncate_evr,
  - dash_preserve_forget_evr,
- it directly shrinks selected parameter blocks using an alignment-based shrink rule,
- it clamps shrink using dash_min_shrink,
- it skips biases unless explicitly enabled.

Preserve this design as closely as possible.

DASH must remain a warm-start method. It must directly shrink selected parameters in-place before the main unlearning loop. It must not introduce a second optimizer loop.

# Stable Diffusion adaptation

For Stable Diffusion, DASH gradient collection should use the normal diffusion denoising loss.

The expected high-level computation is:

    images, prompts = batch["image"], batch["prompt"]

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    noise = torch.randn_like(latents)

    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=latents.device,
    ).long()

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    with torch.no_grad():
        text_inputs = tokenizer(...)
        encoder_hidden_states = text_encoder(...)

    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states,
    ).sample

    target = noise

    loss = F.mse_loss(
        model_pred.float(),
        target.float(),
        reduction="mean",
    )

However, do not blindly copy this if the repo already has a correct helper for Stable Diffusion loss. Reuse the existing code path whenever possible.

If the repo supports v_prediction, respect the scheduler prediction type. Use the same target construction as the existing training code.

# First milestone: class/concept forgetting

The first implementation target is class/concept forgetting.

Add or reuse config support for selecting forget classes/concepts.

Preferred interface if compatible with the repo:

    --forget_classes cat dog

or:

    --forget_concepts cat dog

If the repo already uses another config style, follow it.

The split should work as:

- forget set = samples whose class/concept is in forget_classes,
- retain set = samples whose class/concept is not in forget_classes.

If the repo already has a class/concept split mechanism, reuse it.

If no such mechanism exists, add a minimal clean utility that can split dataset items by class/concept metadata, but do not over-engineer the dataset layer.

# Do not implement NSFW yet

Do not implement NSFW-specific filtering, safety checker training, or NSFW prompt logic in this milestone.

It is fine to leave TODOs such as:

    # TODO: NSFW can later be implemented as a special forget-set provider.

But the DASH runtime itself should not contain NSFW-specific assumptions.

# DASH target in Stable Diffusion

Implement DASH over the U-Net first.

Default target:

    dash_target = unet

Eligible parameters:


- torch.nn.Linear.weight,
- torch.nn.Conv2d.weight.

po DASH masz mieć możliwość bn_recalibrate=true, więc BatchNorm może mieć zaktualizowane running_mean / running_var przez rekalkibrację BN na retain data. 


Skip by default:

- biases,
- embeddings,
- normalization parameters,
- VAE parameters,
- text encoder parameters.

Add optional bias support only if it matches the existing DASH style:

    --dash_include_bias

If easy and clean, add target subsets:

- unet_attn,
- unet_resnet,
- unet_all.

But if this is invasive, implement only unet and leave TODOs.

# Granularity

Support the same granularity options as classification DASH:

- global,
- per_filter.

For Stable Diffusion:

- for Linear, per_filter means one output row,
- for Conv2d, per_filter means one output filter reshaped as [Cout, Cin * kh * kw].

Reuse existing DASH utilities for flattening, SVD/EVR truncation, projections, shrink factors, and histogram logging if possible.

Do not fork mathematical code unnecessarily.

# Signal modes

Implement or reuse the following signal modes.

## retain_only

Use retain gradients as the preserve signal - the main intended mode.

## forget_perp_retain

Compute the component of forget gradients perpendicular to retain gradients.

## preserve_complement

...
High-level behavior:

1. collect retain gradients,
2. collect forget gradients,
3. optionally truncate forget gradients using dash_preserve_forget_evr,
4. compute preserve signal as retain component orthogonal to truncated forget gradient,
5. shrink the complement more aggressively.

Match the existing classification implementation as closely as possible.

# Gradient collection

DASH should collect gradients from the diffusion loss.

Important requirements:

1. Collect gradients only for selected DASH parameters.
2. Reset gradients after each collection step.
3. Do not call optimizer.step().
4. Do not update EMA weights, optimizer state, or scheduler state.
5. Do not update VAE.
6. Do not update text encoder in the first version.
7. U-Net is allowed to receive gradients during DASH collection, but weights should be changed only by the DASH shrink rule.

# Stochastic views / dash_num_aug

In classification, dash_num_aug means multiple stochastic views per batch.

For diffusion, interpret dash_num_aug as multiple stochastic denoising views per batch.

A view may differ by:

- sampled timestep,
- sampled Gaussian noise,
- optional image augmentation only if the repo already has a clean augmentation path.

For each batch:

1. compute view-level gradients,
2. average gradients across views,
3. then aggregate across batches using mean or ema.

Default for Stable Diffusion:

    dash_num_aug = 1

Reason: diffusion already has stochasticity from timestep and noise sampling.

# Gradient aggregation

Support:

- mean,
- ema.

For mean:

- compute exact average gradient over processed DASH batches.

For ema:

- use the same convention as existing classification DASH,
- use dash_alpha consistently with the existing implementation.

# AMP / GradScaler

If the current Stable Diffusion training uses AMP or GradScaler, ensure DASH gradients are not aggregated while scaled.

Valid approaches:

1. collect DASH gradients with AMP disabled, or
2. correctly unscale gradients before aggregation.

Prefer the safest implementation.

If AMP is disabled only during DASH, document this clearly in code comments and logs.

# Memory constraints

Stable Diffusion is memory-heavy. Implement DASH carefully.

Requirements:

- avoid storing activations after backward,
- detach accumulated gradients,
- consider CPU offload for accumulated gradient statistics if necessary,
- support limiting the number of DASH batches,
- call zero_grad(set_to_none=True) where appropriate,
- avoid computing gradients for VAE and text encoder,
- use torch.no_grad() for VAE encoding and text encoder forward if they are frozen.

Add these flags if compatible with the repo:

    --dash_retain_batches
    --dash_forget_batches

These should limit how many batches are used for DASH gradient estimation.

# CLI / config flags

Add Stable Diffusion DASH flags in the repo's existing config/argument style.

Prefer names consistent with existing DASH:

    --dash_warm_start
    --dash_target
    --dash_signal_mode {retain_only,forget_perp_retain,preserve_complement}
    --plasticity_granularity {global,per_filter}
    --dash_grad_aggregation {mean,ema}
    --dash_alpha
    --dash_num_aug
    --dash_aug_mode {default,none}
    --dash_min_shrink
    --dash_svd_truncate_evr
    --dash_preserve_forget_evr
    --dash_include_bias
    --dash_log_cosine_histograms
    --dash_cosine_hist_bins
    --dash_retain_batches
    --dash_forget_batches

Recommended defaults for Stable Diffusion class/concept forgetting:

    dash_warm_start = false
    dash_target = unet
    dash_signal_mode = preserve_complement
    plasticity_granularity = per_filter
    dash_grad_aggregation = mean
    dash_alpha = 0.1
    dash_num_aug = 5
    dash_aug_mode = default
    dash_min_shrink = 0.004
    dash_preserve_forget_evr = 0.95
    dash_svd_truncate_evr = 0.95
    dash_include_bias = false

Do not make DASH enabled by default unless the repo's existing experiment configs explicitly request it.

# Integration point

Insert DASH after:

1. model loading,
2. dataset/loader construction,
3. accelerator/model preparation if required by the repo,

but before:

1. main unlearning optimizer loop,
2. learning-rate scheduler stepping,
3. train-time checkpointing.

Expected pseudocode:

    pipeline_or_model = load_stable_diffusion(...)
    retain_loader, forget_loader = build_retain_forget_loaders(...)

    if args.dash_warm_start:
        run_dash_sd_warm_start(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            noise_scheduler=noise_scheduler,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            args=args,
            accelerator=accelerator_if_used,
            logger=logger,
        )

    run_main_unlearning_training(...)

Adapt this to the actual repository.

# Accelerate / distributed training

If the repo uses Hugging Face accelerate, handle model wrapping carefully.

Requirements:

- use the correct wrapped/unwrapped model object for parameter selection,
- ensure gradients are collected on the correct device,
- ensure logging happens only on the main process if needed,
- do not break distributed training.

If necessary, use:

    accelerator.unwrap_model(...)

but only if this matches the repo pattern.

# Logging

Log DASH metrics with a Stable Diffusion specific prefix.

Suggested metric names:

- dash_sd_enabled,
- dash_sd_target,
- dash_sd_signal_mode,
- dash_sd_granularity,
- dash_sd_target_tensor_count,
- dash_sd_target_param_count,
- dash_sd_updated_tensor_count,
- dash_sd_grad_norm_retain,
- dash_sd_grad_norm_forget,
- dash_sd_alignment_mean,
- dash_sd_alignment_std,
- dash_sd_alignment_min,
- dash_sd_alignment_max,
- dash_sd_shrink_mean,
- dash_sd_shrink_std,
- dash_sd_shrink_min,
- dash_sd_shrink_max,
- dash_sd_preserve_evr,
- dash_sd_delta_norm,
- dash_sd_relative_delta_norm.

Also log per-module or per-block summaries if easy, for example:

- dash_sd_down_blocks_*,
- dash_sd_mid_block_*,
- dash_sd_up_blocks_*,
- dash_sd_attention_*,
- dash_sd_resnet_*.

Do not overbuild logging if it complicates the first implementation.

# Evaluation hooks

Do not overbuild evaluation in this task, but do not block future evaluation.

For class/concept forgetting, the useful diagnostics are:

Forget-side:

- generate images from prompts containing forget classes/concepts,
- evaluate whether the forgotten concept appears,
- optionally use a classifier or CLIP-based concept score if available.

Retain-side:

- generate images from retain prompts,
- evaluate prompt-image alignment,
- optionally compute CLIP score if available,
- optionally compute FID/KID if the repo already supports it.

For the first implementation, it is enough to ensure the DASH pipeline does not prevent existing evaluation from running.

If no evaluation exists, leave TODOs.

# Minimal experiment command

After implementation, provide a command adapted to the real repo.

It should look conceptually like:

    python <stable_diffusion_unlearning_entrypoint>.py \
      --task class_forgetting \
      --forget_classes cat \
      --dash_warm_start \
      --dash_target unet \
      --dash_signal_mode preserve_complement \
      --plasticity_granularity per_filter \
      --dash_grad_aggregation mean \
      --dash_num_aug 1 \
      --dash_min_shrink 0.2 \
      --dash_preserve_forget_evr 0.95 \
      --dash_retain_batches 8 \
      --dash_forget_batches 8

Replace <stable_diffusion_unlearning_entrypoint>.py with the actual script.

# Code organization

Prefer a clean module-based implementation.

Possible structure, adapt to the repo:

    <sd_package_or_unlearning_package>/
      dash_sd_runtime.py
      dash_sd_targets.py
      dash_sd_losses.py

Responsibilities:

## dash_sd_runtime.py

- high-level run_dash_sd_warm_start,
- retain/forget gradient collection,
- aggregation,
- signal construction,
- shrink application,
- logging.

## dash_sd_targets.py

- U-Net parameter selection,
- target filtering,
- module name classification,
- bias filtering,
- target summaries.

## dash_sd_losses.py

- helper to compute diffusion denoising loss,
- reuse existing repo loss code if possible.

If the repository already has a DASH package, place SD-specific DASH code there.

If the repository already has an SD package, place DASH code there.

Choose the least invasive layout.

# Reuse existing classification DASH

Before implementing new math, inspect the existing DASH implementation and reuse utilities for:

- flattening parameters,
- per-filter reshaping,
- global reshaping,
- projection,
- orthogonal decomposition,
- EVR/SVD truncation,
- alignment computation,
- shrink factor computation,
- histogram logging,
- summary logging.

Do not duplicate math unless the existing code is too classification-specific.

If you must duplicate something, explain why.

# Tests / smoke checks

Add lightweight tests or smoke scripts if the repo has any test framework.

Minimum checks:

1. DASH can locate U-Net target parameters.
2. DASH skips VAE and text encoder parameters by default.
3. DASH skips biases by default.
4. DASH can run one retain batch and one forget batch.
5. DASH changes at least one selected U-Net parameter when enabled.
6. DASH changes no selected U-Net parameters when disabled.
7. retain_only runs without shape errors.
8. forget_perp_retain runs without shape errors.
9. preserve_complement runs without shape errors.
10. global granularity runs for Linear and Conv2d.
11. per_filter granularity runs for Linear and Conv2d.
12. dash_retain_batches=1 and dash_forget_batches=1 limit gradient collection correctly.
13. if AMP is used, gradients are not aggregated in scaled form.

If full tests are too expensive, create a small dry-run/smoke path using a tiny batch and a tiny number of DASH batches.

# Safety checks during implementation

Add clear runtime checks:

- if dash_signal_mode requires forget gradients but forget_loader is missing, raise a clear error,
- if no eligible U-Net parameters are found, raise a clear error,
- if a batch does not contain required fields, raise a clear error with available keys,
- if loss target type is unsupported, raise a clear error,
- if dash_num_aug < 1, raise a clear error.

# What not to do

Do not implement FLUX.

Do not implement NSFW-specific logic in this first milestone.

Do not train or modify VAE.

Do not train or modify text encoder by default.

Do not add a second optimizer loop for DASH.

Do not call optimizer.step() inside DASH.

Do not silently change the main unlearning objective.

Do not break existing classification DASH.

Do not rename existing DASH flags unless required by the repo's config system.

Do not hard-code dataset paths.

Do not make DASH enabled by default globally.

# Deliverables

After implementation, report:

1. which files were inspected,
2. where the current Stable Diffusion entrypoint is,
3. where the current diffusion loss is computed,
4. how batches are structured,
5. how retain/forget loaders are constructed or expected,
6. which files were added,
7. which files were modified,
8. where DASH is inserted in the Stable Diffusion pipeline,
9. which parameters are selected by default,
10. whether VAE/text encoder are frozen,
11. how DASH gradients are collected,
12. how dash_num_aug is interpreted,
13. how AMP/GradScaler is handled,
14. which signal modes work,
15. which granularities work,
16. what smoke checks/tests were added,
17. the minimal command to run class/concept forgetting with DASH,
18. known limitations,
19. TODOs for NSFW,
20. TODOs for FLUX.

# Preferred implementation order

Follow this order:

1. Inspect current Stable Diffusion code.
2. Identify entrypoint, model objects, loss path, loaders, config, logging.
3. Identify reusable classification DASH utilities.
4. Add U-Net target selector.
5. Add diffusion-loss gradient collection helper.
6. Add run_dash_sd_warm_start.
7. Wire CLI/config flags.
8. Insert DASH before main unlearning loop.
9. Add logging.
10. Add smoke checks.
11. Provide final report and minimal command.