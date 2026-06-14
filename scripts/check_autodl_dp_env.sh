#!/bin/bash
# =============================================================================
# AutoDL DP PushT Baseline — One-Click Environment Check
# =============================================================================
# Usage:
#   chmod +x scripts/check_autodl_dp_env.sh
#   bash scripts/check_autodl_dp_env.sh
#
# This script ONLY checks prerequisites. It does NOT download data or train.
# Run scripts/download_pusht_data.sh first if data is missing.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DP_DIR="$PROJECT_ROOT/diffusion_policy-main"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
check_cmd() {
    if command -v "$1" &> /dev/null; then
        pass "$1 found: $(command -v $1)"
    else
        fail "$1 not found"
    fi
}

echo "=============================================================================="
echo "  AutoDL DP PushT Baseline — Environment Check"
echo "  Time: $(date)"
echo "  Project root: $PROJECT_ROOT"
echo "=============================================================================="

# ---- 1. System Info ----
echo ""
echo "--- [1/8] System Info ---"
echo "  Hostname: $(hostname)"
echo "  Current dir: $(pwd)"
echo "  User: $(whoami)"

# ---- 2. GPU Check ----
echo ""
echo "--- [2/8] NVIDIA GPU ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | while read line; do
        pass "GPU: $line"
    done
else
    fail "nvidia-smi not found — is CUDA installed?"
fi

# ---- 3. Conda ----
echo ""
echo "--- [3/8] Conda ---"
check_cmd conda
if command -v conda &> /dev/null; then
    echo "  Conda env: ${CONDA_DEFAULT_ENV:-none}"
    if [ "${CONDA_DEFAULT_ENV:-}" != "robodiff" ]; then
        warn "Expected conda env 'robodiff', got '${CONDA_DEFAULT_ENV:-none}'"
        warn "Run: conda activate robodiff"
    else
        pass "Conda env 'robodiff' active"
    fi
fi

# ---- 4. Python and Packages ----
echo ""
echo "--- [4/8] Python & Core Packages ---"
check_cmd python
if command -v python &> /dev/null; then
    pass "Python: $(python --version 2>&1)"

    TORCH_OK=$(python -c "
import torch
print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.cuda.device_count() if torch.cuda.is_available() else 0}')
" 2>/dev/null)
    if [ -n "$TORCH_OK" ]; then
        TORCH_VER=$(echo "$TORCH_OK" | cut -d'|' -f1)
        CUDA_OK=$(echo "$TORCH_OK" | cut -d'|' -f2)
        GPU_CNT=$(echo "$TORCH_OK" | cut -d'|' -f3)
        pass "torch $TORCH_VER, CUDA=$CUDA_OK, GPU_count=$GPU_CNT"
    else
        fail "torch import failed"
    fi

    for pkg in hydra zarr gym pygame pymunk wandb cv2 numpy diffusers; do
        python -c "import $pkg" 2>/dev/null && pass "import $pkg" || fail "import $pkg"
    done

    # Check diffusion_policy package
    python -c "import diffusion_policy" 2>/dev/null && pass "import diffusion_policy" || {
        fail "import diffusion_policy"
        warn "Run: cd diffusion_policy-main && pip install -e ."
    }
fi

# ---- 5. Data Check ----
echo ""
echo "--- [5/8] PushT Data ---"
ZARR_PATH="$DP_DIR/data/pusht/pusht_cchi_v7_replay.zarr"
if [ -d "$ZARR_PATH" ]; then
    pass "Data found: $ZARR_PATH"
    python -c "
import zarr
z = zarr.open('$ZARR_PATH', mode='r')
print(f'  Episodes: {len(z[\"meta\"][\"episode_ends\"])}')
print(f'  Total steps: {z[\"meta\"][\"episode_ends\"][-1]}')
print(f'  Keys: {list(z[\"data\"].keys())}')
" 2>/dev/null && pass "zarr readable" || fail "zarr read failed"
else
    fail "Data not found at $ZARR_PATH"
    echo ""
    echo "  Download with:"
    echo "    bash scripts/download_pusht_data.sh"
fi

# ---- 6. Dataset Smoke Test ----
echo ""
echo "--- [6/8] Dataset Smoke Test ---"
DS_SCRIPT="$DP_DIR/scripts/smoke_pusht_dataset.py"
if [ -f "$DS_SCRIPT" ]; then
    cd "$DP_DIR"
    if python scripts/smoke_pusht_dataset.py 2>&1 | tail -3; then
        pass "Dataset smoke test passed"
    else
        fail "Dataset smoke test failed"
    fi
    cd "$PROJECT_ROOT"
else
    fail "smoke_pusht_dataset.py not found at $DS_SCRIPT"
fi

# ---- 7. Env Smoke Test ----
echo ""
echo "--- [7/8] Env Smoke Test ---"
export SDL_VIDEODRIVER=dummy
ENV_SCRIPT="$DP_DIR/scripts/smoke_pusht_env.py"
if [ -f "$ENV_SCRIPT" ]; then
    cd "$DP_DIR"
    if python scripts/smoke_pusht_env.py 2>&1 | tail -3; then
        pass "Env smoke test passed"
    else
        fail "Env smoke test failed"
    fi
    cd "$PROJECT_ROOT"
else
    fail "smoke_pusht_env.py not found at $ENV_SCRIPT"
fi

# ---- 8. Train Smoke Test ----
echo ""
echo "--- [8/8] Train Smoke Test ---"
TRAIN_SCRIPT="$DP_DIR/scripts/smoke_pusht_train.py"
if [ -f "$TRAIN_SCRIPT" ]; then
    cd "$DP_DIR"
    if python scripts/smoke_pusht_train.py --num-steps 10 --device cuda:0 2>&1 | tail -5; then
        pass "Train smoke test passed"
    else
        fail "Train smoke test failed"
    fi
    cd "$PROJECT_ROOT"
else
    fail "smoke_pusht_train.py not found at $TRAIN_SCRIPT"
fi

# ---- Summary ----
echo ""
echo "=============================================================================="
echo "  Summary: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "=============================================================================="

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "Troubleshooting priority:"
    echo "  1. Install missing packages: pip install <pkg>"
    echo "  2. Verify conda env: conda activate robodiff"
    echo "  3. Install diffusion_policy: cd diffusion_policy-main && pip install -e ."
    echo "  4. Download data: bash scripts/download_pusht_data.sh"
    echo "  5. Check GPU/CUDA: nvidia-smi"
    exit 1
else
    echo ""
    echo "All checks passed! Ready for full training:"
    echo "  cd diffusion_policy-main"
    echo "  python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn \\"
    echo "    training.seed=42 training.device=cuda:0 \\"
    echo "    hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'"
    exit 0
fi
