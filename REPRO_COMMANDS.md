# Reproduction Commands — UWM-PushT Project

> All paths relative to `/root/autodl-tmp/UWM_pushT/`
> Conda environment: `robodiff`

---

## 1. Environment

```bash
# Activate environment
conda activate robodiff

# Required env vars
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export PYTHONPATH=/root/autodl-tmp/UWM_pushT/unified-world-model-main:/root/autodl-tmp/UWM_pushT/diffusion_policy-main:$PYTHONPATH
```

Key versions:
- Python 3.9.25
- PyTorch 2.5.1+cu121
- CUDA 12.1 (Driver 12.2)
- gym 0.26.2

---

## 2. Data

```bash
# PushT data should be at:
ls diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr/

# If missing, download:
cd diffusion_policy-main && mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip && cd ../..
```

---

## 3. Apply Required Patches

### 3.1 gym 0.26 compatibility (DP code)

```bash
cd diffusion_policy-main
patch -p1 < /root/autodl-tmp/UWM_pushT/logs/dp_gym026_compat.patch
```

Modifies:
- `diffusion_policy/env_runner/pusht_image_runner.py` — `shared_memory=False`
- `diffusion_policy/gym_util/async_vector_env.py` — gym 0.26 API signatures

### 3.2 Offline VAE loading (UWM code)

```bash
cd unified-world-model-main
patch -p1 < /root/autodl-tmp/UWM_pushT/logs/uwm_offline_vae.patch
```

Modifies:
- `models/common/transforms.py` — adds `local_files_only=True` to SDXL-VAE loading

---

## 4. Fix diffusion_policy Import

```bash
conda activate robodiff
cd diffusion_policy-main
python setup.py develop   # NOT pip install -e . (broken with pip 26.x)
```

---

## 5. DP PushT Baseline

### 5.1 Smoke test (10 steps)

```bash
cd diffusion_policy-main
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.debug=True \
  training.device=cuda:0 \
  logging.mode=disabled \
  hydra.run.dir='data/outputs/smoke_test'
```

### 5.2 50-epoch training (budget baseline)

```bash
cd diffusion_policy-main
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  logging.mode=disabled \
  training.num_epochs=50 \
  training.rollout_every=10 \
  training.checkpoint_every=10 \
  training.val_every=10 \
  checkpoint.topk.k=1 \
  hydra.run.dir='data/outputs/dp_pusht_50epoch_seed42'
```

Expected: ~18 min on RTX 4090. Best checkpoint: epoch 40, `test_mean_score=0.726`.

### 5.3 Full 3050-epoch training (NOT recommended with 50GB disk)

```bash
# Requires ~30GB free disk for checkpoints
cd diffusion_policy-main
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  logging.mode=disabled \
  hydra.run.dir='data/outputs/dp_pusht_baseline_seed42'
```

Expected: ~470 GPU-hours (~19.5 days). Paper reports `test_mean_score ≈ 0.9+`.

### 5.4 DP eval on checkpoint

```bash
cd diffusion_policy-main
python eval.py \
  --checkpoint data/outputs/dp_pusht_50epoch_seed42/checkpoints/epoch=0040-test_mean_score=0.726.ckpt \
  --output_dir data/pusht_eval_output \
  --device cuda:0
```

---

## 6. UWM-PushT

### 6.1 Training (20k steps)

Command was executed via `scripts/smoke_uwm_pusht_train.py` with parameters extrapolated from training log:

```bash
# The exact command is not fully recoverable from logs.
# Based on log header: device=cuda:0, batch_size=64, num_steps=20000, lr=1e-4
# Resume from 5k checkpoint to reach 20k.
```

Training config inferred from log:
- device: cuda:0
- batch_size: 64
- num_steps: 20000 (resumed from 5000)
- lr: 1e-4
- save_every: 5000

### 6.2 Eval (50 episodes)

```bash
cd /root/autodl-tmp/UWM_pushT

python -u unified-world-model-main/experiments/uwm/eval_pusht.py \
  --checkpoint outputs/uwm_pusht/short_20k_bs64_resume5k/latest.pt \
  --device cuda:0 \
  --num-episodes 50 \
  --max-steps 300 \
  --n-action-steps 8 \
  --output-dir outputs/uwm_pusht_eval/short_20k_bs64_50eps
```

Expected on RTX 4090: ~15 min for 50 episodes. Result: mean_max_reward=0.1121.

---

## 7. Quick Start (on a fresh machine)

```bash
# 1. Clone/move code and data
cd /path/to/UWM_pushT/
ls diffusion_policy-main/
ls unified-world-model-main/
ls diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr/

# 2. Activate conda env (must have: python 3.9, torch 2.x, zarr, cv2, pymunk, pygame, hydra)
conda activate robodiff

# 3. Apply patches
cd diffusion_policy-main && patch -p1 < ../logs/dp_gym026_compat.patch && cd ..
cd unified-world-model-main && patch -p1 < ../logs/uwm_offline_vae.patch && cd ..

# 4. Install diffusion_policy
cd diffusion_policy-main && python setup.py develop && cd ..

# 5. Set env vars
export HF_HUB_OFFLINE=1
export PYTHONPATH=$(pwd)/unified-world-model-main:$(pwd)/diffusion_policy-main:$PYTHONPATH

# 6. DP smoke test
cd diffusion_policy-main
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.debug=True training.device=cuda:0 logging.mode=disabled \
  hydra.run.dir='data/outputs/smoke_test'

# 7. UWM eval
cd /path/to/UWM_pushT
python unified-world-model-main/experiments/uwm/eval_pusht.py \
  --checkpoint artifacts_keep/uwm_20k/checkpoint_20k_latest.pt \
  --device cuda:0 --num-episodes 50 --max-steps 300 --n-action-steps 8 \
  --output-dir outputs/uwm_pusht_eval/test
```

---

## 8. Important Notes

1. **Do NOT run full DP 3050-epoch training** on a 50GB disk. Each checkpoint is 4GB. Full training needs ~30GB for checkpoints alone.

2. **Always use 50 episodes for evaluation.** The earlier 10-episode UWM result (0.27) was 2.4x higher than the 50-episode result (0.11). Small eval sample sizes produce unreliable metrics.

3. **The UWM 10-episode result (0.2724) should NOT be used as the primary comparison result.**

4. **gym version must be 0.26.x** if applying `dp_gym026_compat.patch`. If using gym 0.21, revert the patch.

5. **SDXL-VAE must be cached** at `~/.cache/huggingface/hub/models--stabilityai--sdxl-vae/` (320 MB). If not available, remove `local_files_only=True` from the patch but requires network access.
