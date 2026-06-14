#!/usr/bin/env python3
"""Smoke test for UWM PushT full forward pass.

Usage:
  cd /root/autodl-tmp/UWM_pushT
  source /root/miniconda3/bin/activate robodiff
  export PYTHONPATH=/root/autodl-tmp/UWM_pushT/unified-world-model-main:$PYTHONPATH
  python scripts/smoke_uwm_pusht_forward.py \
    --zarr-path diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr \
    --device cuda:0 \
    --batch-size 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
UWM_ROOT = str(PROJECT_ROOT / "unified-world-model-main")
sys.path.insert(0, UWM_ROOT)


def collate_fn(batch):
    if isinstance(batch[0], dict):
        return {k: collate_fn([b[k] for b in batch]) for k in batch[0]}
    return torch.stack(batch)


def process_batch(batch, obs_horizon, action_horizon, device):
    action_start = obs_horizon - 1
    action_end = action_start + action_horizon
    curr_obs = {k: v[:, : action_start + 1].to(device) for k, v in batch["obs"].items()}
    next_obs = {k: v[:, action_end:].to(device) for k, v in batch["obs"].items()}
    actions = batch["action"][:, action_start:action_end].to(device)
    return curr_obs, next_obs, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str,
                        default="diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.is_absolute():
        zarr_path = str(PROJECT_ROOT / zarr_path)
    else:
        zarr_path = str(zarr_path)
    device = torch.device(args.device)

    print("=" * 60)
    print("UWM PushT Full Forward Smoke Test")
    print(f"  zarr:    {zarr_path}")
    print(f"  device:  {device}")
    print(f"  batch:   {args.batch_size}")
    print("=" * 60)

    # ---- 1. Dataset ----
    print("\n[1/6] Creating PushT dataset...")
    from datasets.pusht import make_pusht_dataset

    shape_meta = {
        "obs": {
            "image": {"shape": [96, 96, 3], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }

    seq_len = 19
    train_set, val_set = make_pusht_dataset(
        name="pusht_forward_smoke",
        zarr_path=zarr_path,
        shape_meta=shape_meta,
        seq_len=seq_len,
        val_ratio=0.02,
        max_train_episodes=90,
        seed=42,
        normalize_action=True,
        normalize_lowdim=True,
    )
    print(f"  Train len: {len(train_set)}, Val len: {len(val_set)}")

    # ---- 2. Batch ----
    print(f"\n[2/6] Loading batch (size={args.batch_size})...")
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn)
    batch = next(iter(loader))

    obs_horizon = 2
    action_horizon = 16
    curr_obs, next_obs, actions = process_batch(batch, obs_horizon, action_horizon, device)

    print(f"  curr_obs.image:     {list(curr_obs['image'].shape)} {curr_obs['image'].dtype} {curr_obs['image'].device}")
    print(f"  curr_obs.agent_pos: {list(curr_obs['agent_pos'].shape)} {curr_obs['agent_pos'].dtype} {curr_obs['agent_pos'].device}")
    print(f"  next_obs.image:     {list(next_obs['image'].shape)} {next_obs['image'].dtype} {next_obs['image'].device}")
    print(f"  next_obs.agent_pos: {list(next_obs['agent_pos'].shape)} {next_obs['agent_pos'].dtype} {next_obs['agent_pos'].device}")
    print(f"  actions:            {list(actions.shape)} {actions.dtype} {actions.device}")

    B = args.batch_size
    assert list(curr_obs["image"].shape) == [B, 2, 96, 96, 3]
    assert list(curr_obs["agent_pos"].shape) == [B, 2, 2]
    assert list(actions.shape) == [B, 16, 2]
    print("  batch process: PASS")

    # ---- 3. Model ----
    print("\n[3/6] Creating UWM model...")
    from models.uwm import UnifiedWorldModel
    from models.uwm.obs_encoder import UWMObservationEncoder

    obs_encoder = UWMObservationEncoder(
        shape_meta=shape_meta,
        num_frames=2,
        embed_dim=768,
        resize_shape=None,
        crop_shape=None,
        random_crop=False,
        color_jitter=None,
        imagenet_norm=False,
        vision_backbone="resnet",
        use_low_dim=True,
        use_language=False,
    )

    model = UnifiedWorldModel(
        action_len=16,
        action_dim=2,
        obs_encoder=obs_encoder,
        embed_dim=768,
        timestep_embed_dim=512,
        latent_patch_shape=[2, 4, 4],
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_registers=8,
        num_train_steps=100,
        num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params / 1e6:.1f}M")
    print(f"  use_low_dim:  {model.obs_encoder.use_low_dim}")
    print(f"  rgb_keys:     {model.obs_encoder.rgb_keys}")
    print(f"  low_dim_keys: {model.obs_encoder.low_dim_keys}")
    print(f"  latent shape: {model.latent_img_shape}")
    print("  model create: PASS")

    # ---- 4. Normalizer ----
    print("\n[4/6] Attaching normalizers...")
    model.action_normalizer = train_set.action_normalizer
    model.lowdim_normalizer = train_set.lowdim_normalizer
    print(f"  action_normalizer.scale:  {model.action_normalizer.scale}")
    print(f"  action_normalizer.offset: {model.action_normalizer.offset}")
    print("  normalizer attach: PASS")

    # ---- 5. Forward ----
    print("\n[5/6] Running forward pass...")
    model.eval()
    with torch.no_grad():
        loss, info = model(curr_obs, next_obs, actions)

    print(f"  loss:           {loss.item():.6f}")
    print(f"  info keys:      {list(info.keys())}")
    print(f"  action_loss:    {info['action_loss']:.6f}")
    print(f"  dynamics_loss:  {info['dynamics_loss']:.6f}")
    print("  forward pass: PASS")

    # ---- 6. Validation ----
    print("\n[6/6] Validation...")
    assert np.isfinite(loss.item()), "Loss is not finite!"
    assert np.isfinite(info["action_loss"]), "action_loss is not finite!"
    assert np.isfinite(info["dynamics_loss"]), "dynamics_loss is not finite!"
    assert loss.item() > 0, "Loss should be positive"
    print("  loss finite:          PASS")
    print("  action_loss finite:   PASS")
    print("  dynamics_loss finite: PASS")

    print("\n" + "=" * 60)
    print("FORWARD SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
