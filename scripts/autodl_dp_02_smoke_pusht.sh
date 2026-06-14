#!/bin/bash
# AutoDL DP Baseline: Step 02 - PushT Smoke Test
# Uses existing robodiff environment (NOT dp-pusht).
#
# Prerequisites:
#   1. robodiff conda env exists
#   2. data/pusht/pusht_cchi_v7_replay.zarr exists
#   3. diffusion_policy importable (via pip install -e . or PYTHONPATH)
#
# Verifies:
#   - zarr data exists and has correct structure
#   - PushTImageDataset can instantiate and return correct shapes
#   - PushTImageEnv can be created
#   - Debug training (2 epochs, 3 steps) runs end-to-end
#
# Usage:
#   cd /root/autodl-tmp/UWM_pushT
#   bash scripts/autodl_dp_02_smoke_pusht.sh 2>&1 | tee logs/phase1_pusht_smoke.log

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$REPO_DIR/logs/phase1_pusht_smoke.log"
mkdir -p "$REPO_DIR/logs"

cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robodiff

# Fix PYTHONPATH if diffusion_policy import broken
if ! python -c "import diffusion_policy" 2>/dev/null; then
    echo "[!] diffusion_policy not importable, setting PYTHONPATH fallback" | tee -a $LOG_FILE
    export PYTHONPATH="$REPO_DIR/diffusion_policy-main:$PYTHONPATH"
fi

echo "==========================================" | tee -a $LOG_FILE
echo " Phase 2: PushT Smoke Test (robodiff env)" | tee -a $LOG_FILE
echo " Repo: $REPO_DIR" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE

# =============================================
# [1/5] Check zarr data
# =============================================
echo "" | tee -a $LOG_FILE
echo "[1/5] Checking zarr data..." | tee -a $LOG_FILE
ZARR_PATH="diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"

if [ -d "$ZARR_PATH" ]; then
    echo "  Zarr path exists: $ZARR_PATH" | tee -a $LOG_FILE
    echo "  Data directory:"  | tee -a $LOG_FILE
    ls -la "$ZARR_PATH/data/" 2>&1 | tee -a $LOG_FILE
    echo "  Meta directory:"  | tee -a $LOG_FILE
    ls -la "$ZARR_PATH/meta/" 2>&1 | tee -a $LOG_FILE
else
    echo "  ERROR: Zarr path not found at $ZARR_PATH" | tee -a $LOG_FILE
    echo "  Download with:" | tee -a $LOG_FILE
    echo "    cd diffusion_policy-main && mkdir -p data && cd data" | tee -a $LOG_FILE
    echo "    wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip" | tee -a $LOG_FILE
    echo "    unzip pusht.zip && rm -f pusht.zip" | tee -a $LOG_FILE
    exit 1
fi

# =============================================
# [2/5] Instantiate Dataset, verify shapes
# =============================================
echo "" | tee -a $LOG_FILE
echo "[2/5] Testing PushTImageDataset..." | tee -a $LOG_FILE
python -c "
import os, sys
import numpy as np

zarr_path = os.path.join('diffusion_policy-main', 'data', 'pusht', 'pusht_cchi_v7_replay.zarr')
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

dataset = PushTImageDataset(
    zarr_path=zarr_path,
    horizon=16,
    pad_before=1,
    pad_after=7,
    seed=42,
    val_ratio=0.02,
    max_train_episodes=90,
)
print(f'  Dataset length: {len(dataset)}')

sample = dataset[0]
print(f'  Keys: {list(sample.keys())}')
print(f'  obs keys: {list(sample[\"obs\"].keys())}')

img = sample['obs']['image']
agent = sample['obs']['agent_pos']
act = sample['action']

print(f'  image shape:       {tuple(img.shape)}   dtype={img.dtype}  range=[{img.min():.3f},{img.max():.3f}]')
print(f'  agent_pos shape:   {tuple(agent.shape)}  dtype={agent.dtype}')
print(f'  action shape:      {tuple(act.shape)}   dtype={act.dtype}')

# Verify expected shapes
assert img.shape == (16, 3, 96, 96), f'FAIL: expected image (16,3,96,96), got {img.shape}'
# dtype is torch.float32 after dict_apply
import torch
assert img.dtype == torch.float32, f'FAIL: expected torch.float32, got {img.dtype}'
assert agent.shape == (16, 2), f'FAIL: expected agent_pos (16,2), got {agent.shape}'
assert act.shape == (16, 2), f'FAIL: expected action (16,2), got {act.shape}'
print('  PushTImageDataset shapes: ALL CORRECT')
" 2>&1 | tee -a $LOG_FILE

# =============================================
# [3/5] Test validation split & normalizer
# =============================================
echo "" | tee -a $LOG_FILE
echo "[3/5] Testing validation split & normalizer..." | tee -a $LOG_FILE
python -c "
import os

zarr_path = os.path.join('diffusion_policy-main', 'data', 'pusht', 'pusht_cchi_v7_replay.zarr')
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

dataset = PushTImageDataset(
    zarr_path=zarr_path, horizon=16, pad_before=1, pad_after=7,
    seed=42, val_ratio=0.02, max_train_episodes=90,
)
val = dataset.get_validation_dataset()
print(f'  Val dataset length: {len(val)}')

norm = dataset.get_normalizer()
norm_keys = list(norm.params_dict.keys())
print(f'  Normalizer keys: {norm_keys}')
assert 'action' in norm_keys, 'FAIL: action not in normalizer'
assert 'agent_pos' in norm_keys, 'FAIL: agent_pos not in normalizer'
assert 'image' in norm_keys, 'FAIL: image not in normalizer'

img_norm = norm['image']
pd = img_norm.params_dict
print(f'  Image normalizer: scale={pd[\"scale\"].detach().numpy()}, offset={pd[\"offset\"].detach().numpy()}')
print('  Validation & normalizer: OK')
" 2>&1 | tee -a $LOG_FILE

# =============================================
# [4/5] Test PushTImageEnv
# =============================================
echo "" | tee -a $LOG_FILE
echo "[4/5] Testing PushTImageEnv instantiation & step..." | tee -a $LOG_FILE
python -c "
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
import numpy as np

env = PushTImageEnv(legacy=True, render_size=96)
print(f'  Observation space: {env.observation_space}')
print(f'  Action space: {env.action_space}')

obs = env.reset()
print(f'  obs[image] shape:     {obs[\"image\"].shape}')
print(f'  obs[agent_pos] shape: {obs[\"agent_pos\"].shape}')
print(f'  obs image dtype: {obs[\"image\"].dtype}, range=[{obs[\"image\"].min():.3f},{obs[\"image\"].max():.3f}]')

action = env.action_space.sample()
next_obs, reward, done, info = env.step(action)
print(f'  Step reward: {reward}')
print(f'  next_obs[image] shape: {next_obs[\"image\"].shape}')

assert obs['image'].shape == (3, 96, 96), f'FAIL: expected env image (3,96,96), got {obs[\"image\"].shape}'
assert obs['agent_pos'].shape == (2,), f'FAIL: expected env agent_pos (2,), got {obs[\"agent_pos\"].shape}'
print('  PushTImageEnv: OK')
del env
" 2>&1 | tee -a $LOG_FILE

# =============================================
# [5/5] Debug training smoke test
# =============================================
echo "" | tee -a $LOG_FILE
echo "[5/5] Running debug (2-epoch) training smoke test..." | tee -a $LOG_FILE
echo "  Config: image_pusht_diffusion_policy_cnn.yaml" | tee -a $LOG_FILE
echo "  Mode: training.debug=True, logging.mode=disabled" | tee -a $LOG_FILE

cd diffusion_policy-main

python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.debug=True \
  training.device=cuda:0 \
  logging.mode=disabled \
  hydra.run.dir='data/outputs/smoke_test' 2>&1 | tee -a $LOG_FILE

TRAIN_EXIT_CODE=$?
cd "$REPO_DIR"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "  Debug training smoke: PASSED" | tee -a $LOG_FILE
else
    echo "" | tee -a $LOG_FILE
    echo "  Debug training smoke: FAILED (exit code $TRAIN_EXIT_CODE)" | tee -a $LOG_FILE
    exit $TRAIN_EXIT_CODE
fi

echo "" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo " Smoke test complete." | tee -a $LOG_FILE
echo " See: $LOG_FILE" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
