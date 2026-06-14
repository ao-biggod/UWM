# UWM PushT

Unified World Model (UWM) applied to the PushT robotic manipulation task. Extends the UWM framework with PushT environment integration, cross-attention variants, and video auxiliary loss experiments.

## Project Structure

| Directory | Description |
|-----------|-------------|
| `scripts/` | Training, evaluation, and diagnostic scripts for all experiment phases |
| `diffusion_policy-main/` | Diffusion Policy official implementation (PushT baseline) |
| `unified-world-model-main/` | UWM official implementation (DROID, LIBERO, Robomimic) |
| `artifacts_keep/` | Trained model checkpoints (gitignored) |
| `outputs/` | Evaluation outputs, videos, metrics (gitignored) |

## Key Results

- **Cross-Attention DP-only** achieves best score of 0.491 at 20k steps
- **Phase B λ=0.05** video auxiliary loss: new SOTA at **0.614** vs baseline (0.534)
- Video branch hurts cross-attention performance (Δ=-0.136, p=0.011)

## Experiments

- **Phase 1**: Diffusion Policy PushT baseline reproduction
- **Phase 2**: Cross-attention variants comparison (5 variants)
- **Phase B**: Small video auxiliary loss as regularizer
- Diagnostic tools: rollout trajectory analysis, gradient conflict diagnosis, paired evaluation

## Setup

See `REPRO_COMMANDS.md` for full environment setup and reproduction commands.

## Reference

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [Unified World Model](https://github.com/unified-world-model/unified-world-model)
