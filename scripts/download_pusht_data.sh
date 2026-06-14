#!/bin/bash
# =============================================================================
# Download PushT training data for Diffusion Policy
# =============================================================================
# Usage:
#   bash scripts/download_pusht_data.sh
#
# Data will be placed at:
#   diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr
# =============================================================================

set -e

# Determine script location — assume we're at project root (D:\UWM_pushT equivalent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DP_DIR="$PROJECT_ROOT/diffusion_policy-main"
DATA_DIR="$DP_DIR/data"
PUSHT_DIR="$DATA_DIR/pusht"
ZARR_PATH="$PUSHT_DIR/pusht_cchi_v7_replay.zarr"
ZIP_URL="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"

echo "=============================================================================="
echo "  Download PushT Training Data"
echo "  Project root: $PROJECT_ROOT"
echo "  Target:       $ZARR_PATH"
echo "=============================================================================="

# Check if data already exists
if [ -d "$ZARR_PATH" ]; then
    echo ""
    echo "Data already exists at $ZARR_PATH"
    echo "Size: $(du -sh "$ZARR_PATH" | cut -f1)"
    echo "Skipping download."
    exit 0
fi

# Check if DP directory exists
if [ ! -d "$DP_DIR" ]; then
    echo "ERROR: diffusion_policy-main not found at $DP_DIR"
    exit 1
fi

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Download
echo ""
echo "Downloading pusht.zip from:"
echo "  $ZIP_URL"
echo ""
wget "$ZIP_URL" -O pusht.zip

# Extract
echo ""
echo "Extracting..."
unzip -q pusht.zip
rm -f pusht.zip

# Verify
echo ""
if [ -d "$ZARR_PATH" ]; then
    echo "SUCCESS: Data downloaded to $ZARR_PATH"
    ls -lah "$ZARR_PATH"
else
    echo "ERROR: Data not found after extraction. Expected: $ZARR_PATH"
    echo "Contents of $PUSHT_DIR:"
    ls -lah "$PUSHT_DIR" 2>/dev/null || echo "  (directory does not exist)"
    exit 1
fi

echo ""
echo "Done."
