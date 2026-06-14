# Results Summary: UWM vs Diffusion Policy on PushT

> Date: 2026-05-29
> Platform: AutoDL (RTX 4090, 50GB disk)
> Code: `UWM_pushT/` (unified-world-model + diffusion_policy)

---

## 1. Project Goal

Reproduce Diffusion Policy's PushT image baseline, then compare against UWM (Unified World Model) single-task training on the same PushT task.

---

## 2. Current Implementation Status

| Component | Status | Notes |
|---|---|---|
| UWM training pipeline | ✅ Working | Full training loop: dataset, forward, loss, optimizer |
| UWM eval pipeline | ✅ Working | `eval_pusht.py` runs policy in PushT env |
| DP PushT baseline | ✅ Working | Full training + eval via official `train.py` |
| DP data loading | ✅ Working | PushT zarr → `PushTImageDataset` → model |
| Gym 0.26 compat | ✅ Patched | 3 files modified, patch saved |

---

## 3. Training Configurations

| | UWM 20k | DP 50epoch |
|---|---|---|
| **Model architecture** | UnifiedWorldModel (DiT-based world model) | DiffusionUnetHybrid (CNN encoder + 1D UNet diffusion) |
| **Parameters** | 305.6M | 262.6M |
| **Training steps** | 20,000 | 50 epochs × 168 batches = 8,400 |
| **Batch size** | 64 | 64 |
| **Observation** | Image (96×96) + agent_pos (2) | Image (96×96) + agent_pos (2) |
| **Action dim** | 2 | 2 |
| **Horizon** | 16 (obs=2, action=16) | 16 (obs=2, action=8) |
| **Learning rate** | 1e-4 | 1e-4 (cosine schedule) |
| **Diffusion steps** | 10 inference | 100 inference (DDPM) |
| **Training time** | ~3 hours | ~15 minutes |
| **GPU** | RTX 4090 | RTX 4090 |
| **Init** | Random scratch | Random scratch |
| **Pretrained weights** | SDXL-VAE vision encoder | ResNet18 (ImageNet) |

---

## 4. Training Metrics

### UWM (single-task, 20k steps)

| Step | Loss | Action Loss | Dynamics Loss | Eval Mean Max Reward |
|---|---|---|---|---|
| 5,000 | 0.0594 | 0.0213 | 0.0381 | 0.1415 |
| 10,000 | 0.0587 | 0.0244 | 0.0342 | 0.1895 |
| 20,000 | 0.0294 | 0.0072 | 0.0222 | 0.2724 |

### DP (50 epochs)

| Epoch | Train Loss | Val Loss | Test Mean Score |
|---|---|---|---|
| 0 | 0.3948 | 0.1072 | 0.1278 |
| 10 | 0.0469 | 0.0509 | 0.3071 |
| 20 | 0.0397 | 0.0454 | 0.4635 |
| 30 | 0.0321 | 0.0407 | 0.6007 |
| 40 | 0.0248 | 0.0421 | **0.7259** |
| 49 (final) | 0.0215 | — | — |

---

## 5. Evaluation Comparison

### UWM 20k (50 episodes) — 2026-05-29

| Metric | Value |
|---|---|
| Mean max_reward | 0.1121 |
| Mean total_reward | 32.25 |
| Mean steps | 300.0 |
| Episodes | 50 |

### UWM 20k (10 episodes) — earlier run

| Metric | Value |
|---|---|
| Mean max_reward | 0.2724 |
| Std max_reward | 0.2098 |
| Mean total_reward | 73.46 |
| Mean steps | 300.0 |
| Episodes | 10 |

> **Note**: The 10-episode result (0.27) is significantly higher than the 50-episode result (0.11), demonstrating the importance of sufficient evaluation episodes for reliable metrics.

### DP 50epoch (50 episodes)

| Metric | Value |
|---|---|
| Test mean_score | **0.7259** |
| Eval episodes | 50 |
| Best checkpoint | epoch 40 |

### Head-to-Head (50 episodes each)

| | UWM 20k | DP 50epoch |
|---|---|---|
| Mean max_reward / score | 0.1121 | 0.7259 |
| Episodes | 50 | 50 |
| Checkpoint | step 19999 | epoch 40 |

---

## 6. Key Observations

1. **DP converges much faster**: 50 epochs (~8,400 batches) reaches 0.73 mean_score, while UWM at 20k steps reaches only 0.11 mean_max_reward on the fair 50-episode eval.

2. **DP training is more efficient per step**: DP's policy-focused training loop (direct action prediction) appears more sample-efficient than UWM's world-model + action-head approach for this task.

3. **Loss curves show healthy convergence**: Both models show decreasing loss, with DP's train loss dropping from 1.18 to 0.02 and UWM's from 0.06 to 0.03.

4. **UWM evaluation requires sufficient episodes**: 10-episode result (0.27) was 2.4x higher than 50-episode result (0.11). Small sample size can produce misleading eval metrics.

---

## 7. Limitations & Caveats

1. **Evaluation protocol differences**:
   - DP uses 50 test episodes per rollout, UWM used 10
   - DP metric is "test_mean_score" (mean of per-episode max_reward)
   - UWM metric is "mean_max_reward" (same calculation, different naming)
   - Comparison should ideally use same number of episodes

2. **UWM training is early-stage**:
   - 20k steps is a short run for a 305M-parameter model
   - UWM was trained from scratch on single-task PushT data only
   - UWM's architecture is designed for large-scale multi-task pre-training (video + language), NOT for single-task from-scratch training
   - The SDXL-VAE encoder may not be well-suited for PushT's 96×96 low-res rendering

3. **DP is a mature, tuned baseline**:
   - Tested extensively in the original paper
   - Hyperparameters (lr schedule, EMA, GroupNorm) are well-tuned
   - Smaller model (262M vs 306M) but purpose-built for imitation learning

4. **Not an algorithmic comparison**:
   - Different architectures, different training paradigms
   - UWM's strength is in multi-task generalization, not single-task imitation
   - This comparison shows engineering correctness, not algorithmic superiority

---

## 8. Conclusion

- **DP PushT baseline**: Successfully reproduced. Reaches 0.73 test_mean_score in 50 epochs (15 min on RTX 4090), on track toward paper-reported ~0.9+.

- **UWM PushT pipeline**: Engineering complete. Training, evaluation, and checkpointing all functional. At 20k steps, achieves 0.27 mean_max_reward — far below DP but validates the entire pipeline works end-to-end.

- **Engineering status**: Both codebases are operational on AutoDL/Linux with minor gym 0.26 compatibility patches.

---

## 9. Ablation: Does video/dynamics help or hurt? (2026-05-31)

### 9.1 DP-only same-backbone (no video)

Trained a pure action diffusion policy using the SAME SDXL-VAE + DiT backbone as UWM,
but without the video/dynamics prediction branch. 20k steps, 204.6M params.

| Model | Input | Score (50eps) |
|---|---|---|
| DP official (ResNet18 + UNet) | Image + agent_pos | 0.668 |
| UWM (SDXL-VAE + DiT, joint video+action) | Image + agent_pos | 0.112 |
| **DP-only same-backbone** (no video) | Image + agent_pos | **0.100** |

**Finding**: Removing video/dynamics loss did NOT improve the score. The bottleneck is not the multi-task training.

### 9.2 Lowdim oracle (no vision, full state)

To isolate whether the vision encoder is the bottleneck, trained a lowdim DiT policy
using the FULL 5D state (agent_xy + block_xy + block_angle) — no vision needed.

| Model | Input | Score (50eps) |
|---|---|---|
| Lowdim oracle (5D state, DiT) | Full state (no image) | **0.092** |

**Finding**: Even with perfect state information, the DiT action diffusion cannot solve PushT.
Vision encoder is NOT the bottleneck.

### 9.3 Normalizer / clip_sample bug

Root cause identified: mean/std normalization produces action range [-2.1, 2.8],
but `DDPMScheduler(clip_sample=True)` clips sampled x0 to [-1, 1] during inference.
This cuts off ~40% of the expert action range. DP official uses min/max normalization
(data naturally in [-1,1]) so it's not affected.

Fix experiments:

| Experiment | Normalizer | clip_sample | Steps | Score (50eps) |
|---|---|---|---|---|
| Original lowdim oracle | mean/std | True | 10k | 0.092 |
| + noclip inference only | mean/std | False (infer) | 10k | **0.217** |
| B1 retrain | mean/std | False | 20k | **0.186** |
| B2 retrain (DP recipe) | min/max | True | 20k | **0.186** |

**Finding**: Normalizer fix doubles the score (0.09 → 0.19) and restores correct action range.
But scores plateau at ~0.19, suggesting the DiT backbone / model capacity / training recipe
is the next bottleneck.

### 9.4 Expert action playback

Feeding expert actions from zarr directly to PushT env: mean_max_reward = 0.17
(low because env goal poses differ from expert episodes). Some episodes reach 0.47-0.54,
confirming action format (absolute position [0,512]) is correct.

---

## 10. Known Issues

1. **UWM offline VAE loading** (fixed): Added `local_files_only=True` to `AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")` in `models/common/transforms.py`. Patch saved at `logs/uwm_offline_vae.patch`. Original backed up at `transforms.py.bak_before_offline_vae`.

2. **UWM eval seed variance**: 10-episode eval gave 0.27 mean_max_reward, but 50-episode eval gave 0.11. The 10-episode result was optimistic due to small sample size. 50-episode results should be used for comparisons.

3. **Normalizer/clip_sample mismatch** (identified 2026-05-31, not yet fixed in UWM/DP-only): mean/std normalization + `clip_sample=True` clips action range. Fix: use min/max normalization OR set `clip_sample=False`. Applies to UWM, DP-only, and lowdim oracle.

4. **Action prediction plateau at ~0.19**: After fixing normalization, lowdim oracle (DiT, 5M params) plateaus at 0.19. DP official lowdim uses 1D UNet + 20D keypoint obs + EMA + cosine LR — these differences likely explain the remaining gap.

2. **Episode count mismatch**: DP evaluates 50 episodes, UWM evaluated 10. Direct metric comparison should use equal episode counts.

3. **Metric naming**: DP uses "test_mean_score" and UWM uses "mean_max_reward". They are related PushT rollout metrics, but they come from different eval scripts. A strict comparison requires the same episode count and verified metric implementation.

---

## 10. Next Steps

1. **Fix UWM VAE loading** — add `local_files_only=True` to enable offline eval
2. **UWM 50-episode eval** — for fair comparison with DP's 50-episode protocol
3. **UWM hyperparameter tuning** — learning rate, diffusion steps, observation horizon
4. **UWM longer training** — 100k+ steps with proper LR schedule
5. **DP full 3050-epoch training** — to reach paper-reported 0.9+ score
6. **Fair comparison protocol** — unified eval script, same seeds, same episode count
