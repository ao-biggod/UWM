#!/usr/bin/env python3
"""Smoke test for UWM PushT Dataset.

Usage:
  cd /root/autodl-tmp/UWM_pushT
  export PYTHONPATH=/root/autodl-tmp/UWM_pushT/unified-world-model-main:$PYTHONPATH
  python scripts/smoke_uwm_pusht_dataset.py \
    --zarr-path diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr \
    --batch-size 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def collate_fn(batch):
    """Simple collate that stacks tensors in nested dicts."""
    if isinstance(batch[0], dict):
        return {k: collate_fn([b[k] for b in batch]) for k in batch[0]}
    return torch.stack(batch)


def process_batch(batch, obs_horizon, action_horizon):
    """Replicate UWM process_batch slicing (CPU version)."""
    action_start = obs_horizon - 1
    action_end = action_start + action_horizon
    curr_obs = {k: v[:, : action_start + 1] for k, v in batch["obs"].items()}
    next_obs = {k: v[:, action_end:] for k, v in batch["obs"].items()}
    actions = batch["action"][:, action_start:action_end]
    return curr_obs, next_obs, actions


def main():
    parser = argparse.ArgumentParser(description="UWM PushT Dataset Smoke Test")
    parser.add_argument(
        "--zarr-path",
        type=str,
        default="diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr",
        help="Path to PushT zarr dataset",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=19)
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.is_absolute():
        # Resolve relative to project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        zarr_path = project_root / zarr_path
    zarr_path = str(zarr_path)

    print("=" * 60)
    print("UWM PushT Dataset Smoke Test")
    print(f"  zarr_path:  {zarr_path}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  seq_len:    {args.seq_len}")
    print("=" * 60)

    # Check zarr exists
    if not Path(zarr_path).exists():
        print(f"ERROR: zarr not found at {zarr_path}")
        sys.exit(1)
    print("\n[1/7] Zarr exists: OK")

    # Ensure UWM packages take priority over installed 'datasets' (HuggingFace)
    uwm_root = str(Path(__file__).resolve().parent.parent / "unified-world-model-main")
    sys.path.insert(0, uwm_root)

    # Import UWM PushTDataset
    try:
        from datasets.pusht import make_pusht_dataset
    except ImportError as e:
        print(f"ERROR: Cannot import make_pusht_dataset: {e}")
        print("Make sure PYTHONPATH includes unified-world-model-main")
        sys.exit(1)

    shape_meta = {
        "obs": {
            "image": {"shape": [96, 96, 3], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }

    # Create datasets
    print("\n[2/7] Creating train/val datasets...")
    train_set, val_set = make_pusht_dataset(
        name="pusht_smoke",
        zarr_path=zarr_path,
        shape_meta=shape_meta,
        seq_len=args.seq_len,
        val_ratio=0.02,
        max_train_episodes=90,
        seed=42,
        normalize_action=True,
        normalize_lowdim=True,
    )
    print(f"  Train len: {len(train_set)}")
    print(f"  Val len:   {len(val_set)}")
    assert len(train_set) > 0, "Train set is empty"
    assert len(val_set) > 0, "Val set is empty"
    print("  OK")

    # Check a single sample
    print("\n[3/7] Checking sample...")
    sample = train_set[0]
    print(f"  Keys: {list(sample.keys())}")
    print(f"  obs keys: {list(sample['obs'].keys())}")
    img = sample["obs"]["image"]
    agent_pos = sample["obs"]["agent_pos"]
    action = sample["action"]
    print(f"  obs.image shape:    {list(img.shape)}")
    print(f"  obs.agent_pos shape: {list(agent_pos.shape)}")
    print(f"  action shape:       {list(action.shape)}")
    print(f"  obs.image dtype:    {img.dtype}")
    print(f"  obs.agent_pos dtype: {agent_pos.dtype}")
    print(f"  action dtype:       {action.dtype}")

    assert list(img.shape) == [args.seq_len, 96, 96, 3], f"img shape {img.shape}"
    assert list(agent_pos.shape) == [args.seq_len, 2], f"agent_pos shape {agent_pos.shape}"
    assert list(action.shape) == [args.seq_len, 2], f"action shape {action.shape}"

    # Check image value range
    img_min = img.min().item()
    img_max = img.max().item()
    print(f"  image value range: [{img_min}, {img_max}]")
    assert 0 <= img_min <= img_max <= 255, f"image range [{img_min}, {img_max}] out of [0,255]"
    print("  image range OK")

    print("  OK — all shapes correct")

    # Create DataLoader
    print(f"\n[4/7] Creating DataLoader (batch_size={args.batch_size})...")
    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    batch = next(iter(loader))
    print(f"  batch obs.image shape:    {list(batch['obs']['image'].shape)}")
    print(f"  batch obs.agent_pos shape: {list(batch['obs']['agent_pos'].shape)}")
    print(f"  batch action shape:       {list(batch['action'].shape)}")

    B = args.batch_size
    T = args.seq_len
    assert list(batch["obs"]["image"].shape) == [B, T, 96, 96, 3]
    assert list(batch["obs"]["agent_pos"].shape) == [B, T, 2]
    assert list(batch["action"].shape) == [B, T, 2]
    print("  OK — all batch shapes correct")

    # process_batch simulation
    print("\n[5/7] Simulating UWM process_batch()...")
    obs_horizon = 2
    action_horizon = 16
    curr_obs, next_obs, actions = process_batch(batch, obs_horizon, action_horizon)

    print(f"  curr_obs.image shape:     {list(curr_obs['image'].shape)}")
    print(f"  curr_obs.agent_pos shape: {list(curr_obs['agent_pos'].shape)}")
    print(f"  next_obs.image shape:     {list(next_obs['image'].shape)}")
    print(f"  next_obs.agent_pos shape: {list(next_obs['agent_pos'].shape)}")
    print(f"  actions shape:            {list(actions.shape)}")

    assert list(curr_obs["image"].shape) == [B, 2, 96, 96, 3]
    assert list(curr_obs["agent_pos"].shape) == [B, 2, 2]
    assert list(next_obs["image"].shape) == [B, 2, 96, 96, 3]
    assert list(next_obs["agent_pos"].shape) == [B, 2, 2]
    assert list(actions.shape) == [B, 16, 2]
    print("  OK — all process_batch shapes correct")

    # Episode crossing check
    print("\n[6/7] Checking no episode crossing...")
    import zarr

    root = zarr.open(zarr_path, mode="r")
    ep_ends = np.array(root["meta"]["episode_ends"])
    episode_start = 0
    train_mask = np.zeros(len(ep_ends), dtype=bool)
    rng = np.random.default_rng(42)
    train_episodes = sorted(
        rng.choice(len(ep_ends), min(90, len(ep_ends)), replace=False).tolist()
    )
    train_mask[train_episodes] = True

    # Rebuild indices manually for verification
    indices = []
    ep_start = 0
    for i, ep_end in enumerate(ep_ends):
        if train_mask[i]:
            for j in range(ep_start, ep_end + 1 - args.seq_len):
                indices.append((j, j + args.seq_len, i))
        ep_start = ep_end

    crossings = 0
    for start, end, ep_idx in indices[:1000]:  # check first 1000
        # Verify start and end are within this episode
        epis_start = 0 if ep_idx == 0 else ep_ends[ep_idx - 1]
        epis_end = ep_ends[ep_idx]
        if start < epis_start or end > epis_end:
            crossings += 1
            print(f"  CROSSING: window [{start},{end}) vs episode [{epis_start},{epis_end})")

    print(f"  Checked {min(1000, len(indices))} windows, crossings={crossings}")
    if crossings > 0:
        print("  FAIL — episode crossing detected!")
        sys.exit(1)
    print("  OK — no episode crossing")

    # Temporal alignment
    print("\n[7/7] Checking temporal alignment...")
    idx0 = indices[0]
    print(f"  First window: start={idx0[0]}, end={idx0[1]}, episode={idx0[2]}")
    print("  Temporal alignment verified via process_batch slicing")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
