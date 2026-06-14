"""
Smoke test for Diffusion Policy PushT Image Dataset.

Verifies:
- Dataset loads from zarr
- Returns correct dict structure
- Image shape/range/dtype are correct
- Action dim = 2
- Validation split works

Usage:
    cd diffusion_policy-main
    python scripts/smoke_pusht_dataset.py

Requires: pip install -e . (from diffusion_policy-main root)
"""

import os
import sys
import numpy as np
import torch

# Ensure the diffusion_policy package is importable
# Assumes this script is at diffusion_policy-main/scripts/smoke_pusht_dataset.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset


# ---- Config ----
ZARR_PATH = os.path.join(REPO_ROOT, "data", "pusht", "pusht_cchi_v7_replay.zarr")
HORIZON = 16
PAD_BEFORE = 1   # n_obs_steps - 1
PAD_AFTER = 7    # n_action_steps - 1
SEED = 42
VAL_RATIO = 0.02


def main():
    print("=" * 60)
    print("PushT Image Dataset Smoke Test")
    print("=" * 60)

    # ---- 1. Check data path ----
    print(f"\n[1/6] Checking data path: {ZARR_PATH}")
    if not os.path.exists(ZARR_PATH):
        print(f"ERROR: Data not found at {ZARR_PATH}")
        print("Download it with:")
        print("  mkdir -p data && cd data")
        print("  wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip")
        print("  unzip pusht.zip")
        sys.exit(1)
    print("  OK - data path exists")

    # ---- 2. Create dataset ----
    print(f"\n[2/6] Creating PushTImageDataset (horizon={HORIZON}, val_ratio={VAL_RATIO})")
    dataset = PushTImageDataset(
        zarr_path=ZARR_PATH,
        horizon=HORIZON,
        pad_before=PAD_BEFORE,
        pad_after=PAD_AFTER,
        seed=SEED,
        val_ratio=VAL_RATIO,
        max_train_episodes=90,
    )
    print(f"  OK - train dataset created, len={len(dataset)}")

    # ---- 3. Check sample structure ----
    print("\n[3/6] Checking sample structure")
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    assert "obs" in sample, "Missing 'obs' key"
    assert "action" in sample, "Missing 'action' key"
    print(f"  obs keys: {list(sample['obs'].keys())}")

    obs = sample["obs"]
    assert "image" in obs, "Missing obs['image']"
    assert "agent_pos" in obs, "Missing obs['agent_pos']"

    # ---- 4. Check shapes ----
    print("\n[4/6] Checking tensor shapes")
    image = obs["image"]
    agent_pos = obs["agent_pos"]
    action = sample["action"]

    print(f"  image shape:     {tuple(image.shape)}")
    print(f"  agent_pos shape: {tuple(agent_pos.shape)}")
    print(f"  action shape:    {tuple(action.shape)}")

    assert image.ndim == 4, f"image should be 4D (T,C,H,W), got {image.ndim}D"
    assert image.shape[0] == HORIZON, f"image T={image.shape[0]}, expected {HORIZON}"
    assert image.shape[1] == 3, f"image C={image.shape[1]}, expected 3 (RGB)"
    assert image.shape[2] == 96, f"image H={image.shape[2]}, expected 96"
    assert image.shape[3] == 96, f"image W={image.shape[3]}, expected 96"

    assert agent_pos.ndim == 2, f"agent_pos should be 2D (T,2), got {agent_pos.ndim}D"
    assert agent_pos.shape[0] == HORIZON
    assert agent_pos.shape[1] == 2, f"agent_pos dim={agent_pos.shape[1]}, expected 2"

    assert action.ndim == 2, f"action should be 2D (T,2), got {action.ndim}D"
    assert action.shape[0] == HORIZON
    assert action.shape[1] == 2, f"action dim={action.shape[1]}, expected 2"
    print("  OK - all shapes correct")

    # ---- 5. Check dtypes and ranges ----
    print("\n[5/6] Checking dtypes and ranges")
    print(f"  image dtype: {image.dtype}")
    print(f"  agent_pos dtype: {agent_pos.dtype}")
    print(f"  action dtype: {action.dtype}")

    assert image.dtype == torch.float32, f"image dtype={image.dtype}, expected float32"
    img_min, img_max = image.min().item(), image.max().item()
    print(f"  image range: [{img_min:.3f}, {img_max:.3f}]")
    assert 0.0 <= img_min <= img_max <= 1.0, f"image range [{img_min}, {img_max}] not in [0,1]"

    assert agent_pos.dtype == torch.float32
    print("  OK - dtypes correct")

    # ---- 6. Check validation split ----
    print("\n[6/6] Checking validation split")
    val_dataset = dataset.get_validation_dataset()
    print(f"  Validation dataset len={len(val_dataset)}")
    assert len(val_dataset) > 0, "Validation dataset should not be empty"
    print("  OK - validation split works")

    # Bonus: check a few more samples
    print("\nBonus: checking 5 random samples for consistency")
    for i in np.random.choice(len(dataset), min(5, len(dataset)), replace=False):
        s = dataset[i]
        assert s["obs"]["image"].shape == (HORIZON, 3, 96, 96)
        assert s["obs"]["agent_pos"].shape == (HORIZON, 2)
        assert s["action"].shape == (HORIZON, 2)

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
