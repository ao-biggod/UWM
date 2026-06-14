"""
Minimal 10-step training smoke test for Diffusion Policy PushT image model.

Uses the REAL PushTImageDataset and DiffusionUnetHybridImagePolicy classes.
Does NOT use workspace, hydra.main, or wandb. No rollout, no checkpoint.

Usage:
    cd diffusion_policy-main
    python scripts/smoke_pusht_train.py

    # With custom args:
    python scripts/smoke_pusht_train.py --zarr-path data/pusht/pusht_cchi_v7_replay.zarr --device cuda:0 --num-steps 10
"""

import argparse
import os
import sys
import numpy as np

# Ensure the diffusion_policy package is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra

# Real classes from the repo
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="10-step DP PushT training smoke test")
    p.add_argument("--zarr-path", type=str,
                   default=os.path.join(REPO_ROOT, "data", "pusht", "pusht_cchi_v7_replay.zarr"),
                   help="Path to PushT zarr data")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device for training (cuda:0 or cpu)")
    p.add_argument("--num-steps", type=int, default=10,
                   help="Number of training steps to run")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for training")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("PushT DP Training Smoke Test")
    print(f"  zarr_path:  {args.zarr_path}")
    print(f"  device:     {args.device}")
    print(f"  num_steps:  {args.num_steps}")
    print(f"  batch_size: {args.batch_size}")
    print("=" * 60)

    # ---- 1. Check data ----
    print("\n[1/5] Checking data path...")
    if not os.path.exists(args.zarr_path):
        print(f"ERROR: Data not found at {args.zarr_path}")
        sys.exit(1)
    print("  OK")

    # ---- 2. Create dataset and dataloader ----
    print("\n[2/5] Creating dataset and dataloader...")
    # Use exact parameters from image_pusht_diffusion_policy_cnn.yaml
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8
    pad_before = n_obs_steps - 1   # = 1
    pad_after = n_action_steps - 1  # = 7

    dataset = PushTImageDataset(
        zarr_path=args.zarr_path,
        horizon=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=90,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,           # avoid multiprocessing in smoke test
        pin_memory=False,
    )
    print(f"  Dataset len: {len(dataset)}, Batches: {len(dataloader)}")

    # ---- 3. Create model ----
    print("\n[3/5] Creating model (DiffusionUnetHybridImagePolicy)...")
    device = torch.device(args.device)

    # Load the full config to extract model parameters
    config_path = os.path.join(REPO_ROOT, "image_pusht_diffusion_policy_cnn.yaml")
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params / 1e6:.1f}M")

    # Set normalizer (REQUIRED before compute_loss)
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)
    model.normalizer.to(device)
    print("  Normalizer set OK")

    # ---- 4. Create optimizer ----
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    print(f"  Optimizer: {type(optimizer).__name__}, lr={cfg.optimizer.lr}")

    # ---- 5. Training loop (10 steps) ----
    print(f"\n[4/5] Running {args.num_steps} training steps...")
    model.train()

    data_iter = iter(dataloader)
    losses = []

    for step in range(args.num_steps):
        # Get next batch (recycle iterator if needed)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Device transfer (mimics workspace line 161)
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

        # Forward + backward
        loss = model.compute_loss(batch)
        loss.backward()

        # Gradient clipping (from config: not specified, skip)
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        losses.append(loss_val)
        print(f"  Step {step:3d}: loss={loss_val:.6f}")

    # ---- 6. Validation ----
    print(f"\n[5/5] Validation...")
    print(f"  Mean loss: {np.mean(losses):.6f}")
    print(f"  Min  loss: {np.min(losses):.6f}")
    print(f"  Max  loss: {np.max(losses):.6f}")

    all_finite = all(np.isfinite(l) for l in losses)
    if not all_finite:
        print("ERROR: Loss contains NaN or Inf!")
        sys.exit(1)

    # Basic sanity: loss should be positive and not exploding
    mean_loss = np.mean(losses)
    assert mean_loss > 0, f"Loss should be > 0, got {mean_loss}"
    assert mean_loss < 1e6, f"Loss should not explode, got {mean_loss}"
    print(f"  All losses finite and reasonable: OK")

    print("\n" + "=" * 60)
    print("TRAINING SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
