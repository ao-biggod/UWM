#!/bin/bash
# AutoDL DP Baseline: Step 01 - Verify & Fix robodiff Environment
# REUSES existing robodiff conda env (does NOT create dp-pusht).
#
# Usage:
#   cd /root/autodl-tmp/UWM_pushT
#   bash scripts/autodl_dp_01_setup_env.sh 2>&1 | tee logs/phase1_env_setup.log

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$REPO_DIR/logs/phase1_env_setup.log"
mkdir -p "$REPO_DIR/logs"

cd "$REPO_DIR"

echo "==========================================" | tee -a $LOG_FILE
echo " Phase 1: Verify robodiff Environment" | tee -a $LOG_FILE
echo " Repo: $REPO_DIR" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE

# Step 1: Verify robodiff exists
echo "" | tee -a $LOG_FILE
echo "[1/5] Checking robodiff conda environment..." | tee -a $LOG_FILE
if conda env list | grep -q "^robodiff "; then
    echo "  robodiff exists: OK" | tee -a $LOG_FILE
else
    echo "  ERROR: robodiff environment not found." | tee -a $LOG_FILE
    echo "  Please create it first: conda env create -f diffusion_policy-main/conda_environment.yaml" | tee -a $LOG_FILE
    exit 1
fi

# Step 2: Activate and fix/verify diffusion_policy editable install
echo "" | tee -a $LOG_FILE
echo "[2/5] Activating robodiff and fixing diffusion_policy..." | tee -a $LOG_FILE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robodiff

cd diffusion_policy-main

# Check if diffusion_policy is importable
if python -c "import diffusion_policy" 2>/dev/null; then
    echo "  diffusion_policy already importable: OK" | tee -a $LOG_FILE
else
    echo "  diffusion_policy import failed, reinstalling via setup.py develop..." | tee -a $LOG_FILE
    pip uninstall -y diffusion_policy 2>/dev/null || true
    python setup.py develop 2>&1 | tee -a $LOG_FILE

    # Verify fix
    if python -c "import diffusion_policy" 2>/dev/null; then
        echo "  diffusion_policy fixed: OK" | tee -a $LOG_FILE
    else
        echo "  WARNING: pip install -e . didn't fix import." | tee -a $LOG_FILE
        echo "  Falling back to PYTHONPATH method." | tee -a $LOG_FILE
        export PYTHONPATH="$REPO_DIR/diffusion_policy-main:$PYTHONPATH"
        python -c "import diffusion_policy" 2>&1 || {
            echo "  FATAL: diffusion_policy still not importable" | tee -a $LOG_FILE
            exit 1
        }
        echo "  diffusion_policy importable via PYTHONPATH: OK" | tee -a $LOG_FILE
    fi
fi

cd "$REPO_DIR"

# Step 3: Verify critical imports
echo "" | tee -a $LOG_FILE
echo "[3/5] Verifying critical imports..." | tee -a $LOG_FILE

python -c "import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>&1 | tee -a $LOG_FILE
python -c "import zarr; print(f'  zarr {zarr.__version__}')" 2>&1 | tee -a $LOG_FILE
python -c "import cv2; print(f'  cv2 {cv2.__version__}')" 2>&1 | tee -a $LOG_FILE
python -c "import pymunk; print(f'  pymunk {pymunk.version}')" 2>&1 | tee -a $LOG_FILE
python -c "import pygame; print(f'  pygame {pygame.ver}')" 2>&1 | tee -a $LOG_FILE
python -c "import gym; print(f'  gym {gym.__version__}')" 2>&1 | tee -a $LOG_FILE
python -c "import hydra; print(f'  hydra {hydra.__version__}')" 2>&1 | tee -a $LOG_FILE
python -c "
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
print('  diffusion_policy (PushTImageDataset/PushTImageRunner/Workspace): OK')
" 2>&1 | tee -a $LOG_FILE

# Step 4: Python & CUDA details
echo "" | tee -a $LOG_FILE
echo "[4/5] Environment details..." | tee -a $LOG_FILE
python --version 2>&1 | tee -a $LOG_FILE
pip --version 2>&1 | tee -a $LOG_FILE
python -c "
import torch
print(f'  CUDA version: {torch.version.cuda}')
print(f'  Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  Device name: {torch.cuda.get_device_name(0)}')
" 2>&1 | tee -a $LOG_FILE

# Step 5: Save pip list for reference
echo "" | tee -a $LOG_FILE
echo "[5/5] Saving pip freeze..." | tee -a $LOG_FILE
pip freeze > "$REPO_DIR/logs/phase1_pip_freeze.txt" 2>&1
echo "  Saved to logs/phase1_pip_freeze.txt" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo " Environment ready. To activate: conda activate robodiff" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
