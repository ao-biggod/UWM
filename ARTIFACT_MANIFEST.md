# Artifact Manifest — UWM-PushT Project

> Generated: 2026-05-29
> Machine: AutoDL (RTX 4090, 50GB, CUDA 12.2)

---

## 1. UWM Checkpoints

| File | Size | Description |
|---|---|---|
| `artifacts_keep/uwm_20k/checkpoint_20k_latest.pt` | 1.2 GB | UWM training step 19999 (latest/final). 305.6M params. Used for 50-ep eval. |
| `artifacts_keep/uwm_20k/checkpoint_10k.pt` | 1.2 GB | UWM training step 10000 (mid-training reference). |
| `outputs/uwm_pusht/short_20k_bs64_resume5k/latest.pt` | 1.2 GB | Original location copy of the 20k checkpoint. |
| `outputs/uwm_pusht/short_20k_bs64_resume5k/checkpoint_step0010000.pt` | 1.2 GB | Original location copy of the 10k checkpoint. |

## 2. DP Checkpoints

| File | Size | Description |
|---|---|---|
| `artifacts_keep/dp_50epoch/epoch=0040-test_mean_score=0.726.ckpt` | 4.0 GB | DP best checkpoint (epoch 40, score 0.726). 262.6M params. |
| `artifacts_keep/dp_50epoch/latest.ckpt` | 4.0 GB | DP final checkpoint (epoch 49). |
| `diffusion_policy-main/data/outputs/dp_pusht_50epoch_seed42/checkpoints/` | 8.0 GB | Original DP output directory (both ckpts). |

## 3. Eval Logs

| File | Size | Description |
|---|---|---|
| `logs/uwm_20k_eval_50eps.log` | ~3 KB | UWM 20k, 50 episodes eval. mean_max_reward=0.1121 |
| `artifacts_keep/uwm_20k/eval_20k.log` | 2.6 KB | UWM 20k, 10 episodes eval (earlier run). mean_max_reward=0.2724 |
| `artifacts_keep/uwm_20k/eval_10k.log` | 2.6 KB | UWM 10k, 10 episodes eval. |
| `artifacts_keep/uwm_20k/eval_5k.log` | 2.6 KB | UWM 5k, 10 episodes eval. |
| `artifacts_keep/uwm_20k/eval_1k.log` | 2.6 KB | UWM 1k, 10 episodes eval. |

## 4. Training Logs & Metrics

| File | Size | Description |
|---|---|---|
| `artifacts_keep/uwm_20k/train_log_20k.jsonl` | 2.4 MB | UWM training log (steps 5000-19999). Contains loss, action_loss, dynamics_loss, lr per step. |
| `artifacts_keep/uwm_20k/uwm_20k_metrics.csv` | ~500 B | Sampled UWM metrics at key steps. |
| `artifacts_keep/dp_50epoch/logs.json.txt` | 824 KB | DP full training log (epochs 0-49). Contains train_loss, val_loss, test_mean_score per step. |
| `artifacts_keep/dp_50epoch/dp_50epoch_metrics.csv` | ~300 B | DP epoch-end metrics. |
| `artifacts_keep/dp_50epoch/hydra_config.yaml` | 3.7 KB | Full DP training config. |
| `artifacts_keep/dp_50epoch/hydra_overrides.yaml` | 198 B | Hydra overrides used for the 50-epoch run. |

## 5. Patches

| File | Size | Description |
|---|---|---|
| `logs/dp_gym026_compat.patch` | 3.1 KB | Fixes for gym 0.26.2 API changes in DP code. 2 files modified. |
| `logs/uwm_offline_vae.patch` | 677 B | Adds `local_files_only=True` to SDXL-VAE loading in UWM. Backup: `transforms.py.bak_before_offline_vae`. |
| `artifacts_keep/dp_50epoch/dp_gym026_compat.patch` | 3.1 KB | Duplicate for safekeeping. |

## 6. Documentation

| File | Size | Description |
|---|---|---|
| `RESULTS_SUMMARY.md` | ~5 KB | English results summary with comparison tables. |
| `FINAL_REPORT_CN.md` | — | Chinese final report (see Step 3). |
| `REPRO_COMMANDS.md` | — | Reproduction commands (see Step 2). |
| `PHASE1_DP_BASELINE.md` | ~8 KB | DP baseline reproduction notes. |
| `PROJECT_CONTEXT.md` | ~8 KB | Project context and setup. |
| `REPRO_PLAN.md` | ~8 KB | Original reproduction plan. |
| `CODE_MAP.md` | ~12 KB | Codebase structure map. |

## 7. Scripts

| File | Description |
|---|---|
| `scripts/autodl_dp_00_check.sh` | AutoDL environment check. |
| `scripts/autodl_dp_01_setup_env.sh` | Verifies & fixes robodiff env. |
| `scripts/autodl_dp_02_smoke_pusht.sh` | DP PushT smoke test (zarr, dataset, env, debug train). |

---

## 8. Retention Guidance

### MUST KEEP (core results)

- `artifacts_keep/uwm_20k/checkpoint_20k_latest.pt`
- `artifacts_keep/uwm_20k/train_log_20k.jsonl`
- `artifacts_keep/uwm_20k/eval_*.log`
- `artifacts_keep/dp_50epoch/epoch=0040-test_mean_score=0.726.ckpt`
- `artifacts_keep/dp_50epoch/logs.json.txt`
- `logs/uwm_20k_eval_50eps.log`
- `logs/dp_gym026_compat.patch`
- `logs/uwm_offline_vae.patch`
- `RESULTS_SUMMARY.md`
- `FINAL_REPORT_CN.md`
- `REPRO_COMMANDS.md`

### SAFE TO DELETE

- `outputs/uwm_pusht/short_20k_bs64_resume5k/checkpoint_step0005000.pt` (already deleted)
- `outputs/uwm_pusht/short_20k_bs64_resume5k/checkpoint_step0015000.pt` (already deleted)
- `diffusion_policy-main/data/outputs/` — can delete after archiving ckpts
- `diffusion_policy-main.zip` (13 MB, keep if needed for reference)
- `unified-world-model-main.zip` (120 KB, keep if needed for reference)
- Any `__pycache__/` directories

### MIGRATION MINIMUM (if moving to another machine)

```
artifacts_keep/uwm_20k/checkpoint_20k_latest.pt  (1.2 GB)
artifacts_keep/uwm_20k/train_log_20k.jsonl        (2.4 MB)
artifacts_keep/uwm_20k/eval_20k.log
artifacts_keep/dp_50epoch/epoch=0040-test_mean_score=0.726.ckpt  (4.0 GB)
artifacts_keep/dp_50epoch/logs.json.txt            (824 KB)
artifacts_keep/dp_50epoch/hydra_config.yaml
logs/uwm_20k_eval_50eps.log
logs/dp_gym026_compat.patch
logs/uwm_offline_vae.patch
RESULTS_SUMMARY.md
FINAL_REPORT_CN.md
REPRO_COMMANDS.md
diffusion_policy-main/  (code)
unified-world-model-main/  (code)
data/pusht/pusht_cchi_v7_replay.zarr/  (~400 MB)
```
