# Debug Session: LoRA NaN Loss at Step 2+

**Session ID**: `lora-nan-loss-step2`
**Date**: 2026-06-08
**Status**: [OPEN]

## Symptoms
- Step 1: Loss = 0.3910 (normal)
- Step 2+: Loss = nan (persistent)
- Signature unchanged across 5+ runs despite applying these fixes:
  - xFormers removed
  - `unet.add_adapter` (diffusers native path)
  - `init_lora_weights="gaussian"`
  - time_ids in fp32
  - autocast("cuda", dtype=fp16) wrapping forward
  - `F.mse_loss(noise_pred.float(), noise.float())`
  - `DDPMScheduler(...)` explicit (not from_pretrained)
  - `enable_input_require_grads()` for gradient checkpointing + peft compatibility

## Environment
- torch 2.7.1+cu126
- diffusers 0.37.1
- peft 0.19.1
- transformers **5.10.2** ← potential culprit (5.x is 2025+ major version)
- xformers 0.0.31.post1 (installed but not enabled)
- Python 3.12.11
- GPU: 3060 6GB (Ampere, supports bf16)

## Hypotheses (3-5 falsifiable)

| # | Hypothesis | Falsifiable By |
|---|------------|----------------|
| H1 | Forward pass output (noise_pred) contains NaN/Inf at step 1 even though loss appears normal due to fp16 precision | Log min/max/has_nan of noise_pred right after forward, before loss |
| H2 | Gradient is NaN/Inf after backward even though loss is normal (NaN grad → NaN params via optimizer.step) | Log min/max/has_nan of first LoRA param's `.grad` after backward, before clip |
| H3 | AdamW internal state (exp_avg_sq) becomes NaN in fp16 storage (PyTorch 2.7 may not promote optimizer state) | Log optimizer state dtype + check exp_avg_sq of first param |
| H4 | transformers 5.10.2 (CLIPTextModel) produces NaN in encoder_hidden_states (5.x had breaking changes in LayerNorm/attention) | Log min/max/has_nan of encoder_hidden_states and pooled output from text encoders |
| H5 | Autocast leaves some op in fp16 that overflows (e.g., GroupNorm weight, scale factor in UNet) | Log has_inf/has_nan of intermediate UNet activations |

## Instrumentation Plan
Add NaN/Inf sentinel logging at:
1. After text encoder forward (encoder_hidden_states, pooled) — falsify H4
2. After UNet forward (noise_pred) — falsify H1
3. After loss.backward() (LoRA grad) — falsify H2
4. After clip_grad_norm_ (LoRA grad post-clip) — falsify H2
5. After optimizer.step() (LoRA params) — confirm corruption
6. Log optimizer state dtype and first param exp_avg_sq stats — falsify H3
7. Log `time_ids` min/max/has_nan — defensive

## Run
- 跑 `run_train.bat`
- 等 step 1, 2, 3 完成
- 收集 debug-lora-nan-loss-step2.ndjson
