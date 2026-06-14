#!/usr/bin/env python3
"""UWM PushT policy inference smoke test.

Verifies model.sample() works on PushT data.

Usage:
  cd /root/autodl-tmp/UWM_pushT
  source /root/miniconda3/bin/activate robodiff
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH=/root/autodl-tmp/UWM_pushT/unified-world-model-main:$PYTHONPATH
  python scripts/smoke_uwm_pusht_policy_infer.py \
    --zarr-path diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr \
    --device cuda:0 \
    --batch-size 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
UWM_ROOT = str(PROJECT_ROOT / "unified-world-model-main")
sys.path.insert(0, UWM_ROOT)


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
    B = args.batch_size

    print("=" * 60)
    print("UWM PushT Policy Inference Smoke Test")
    print(f"  zarr:   {zarr_path}")
    print(f"  device: {device}")
    print(f"  batch:  {B}")
    print("=" * 60)

    # ---- 1. Dataset ----
    print("\n[1/5] Creating dataset...")
    from datasets.pusht import make_pusht_dataset
    shape_meta = {
        "obs": {
            "image": {"shape": [96, 96, 3], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }
    train_set, val_set = make_pusht_dataset(
        name="pusht_infer_smoke", zarr_path=zarr_path, shape_meta=shape_meta,
        seq_len=19, val_ratio=0.02, max_train_episodes=90, seed=42,
        normalize_action=True, normalize_lowdim=True,
    )
    sample = train_set[0]
    print(f"  Train len: {len(train_set)}")
    print("  dataset: PASS")

    # ---- 2. Model ----
    print("\n[2/5] Creating UWM model...")
    from models.uwm import UnifiedWorldModel
    from models.uwm.obs_encoder import UWMObservationEncoder

    obs_encoder = UWMObservationEncoder(
        shape_meta=shape_meta, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False,
        vision_backbone="resnet", use_low_dim=True, use_language=False,
    )
    model = UnifiedWorldModel(
        action_len=16, action_dim=2, obs_encoder=obs_encoder,
        embed_dim=768, timestep_embed_dim=512,
        latent_patch_shape=[2, 4, 4], depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, num_registers=8,
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=True,
    )
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params / 1e6:.1f}M")
    print("  model: PASS")

    # ---- 3. Normalizer ----
    print("\n[3/5] Attaching normalizers...")
    model.action_normalizer = train_set.action_normalizer
    model.lowdim_normalizer = train_set.lowdim_normalizer
    action_scale = torch.tensor(model.action_normalizer.scale, device=device).float()
    action_offset = torch.tensor(model.action_normalizer.offset, device=device).float()
    print(f"  action_scale:  {action_scale.cpu().numpy()}")
    print(f"  action_offset: {action_offset.cpu().numpy()}")
    print("  normalizer: PASS")

    # ---- 4. Policy inference ----
    print(f"\n[4/5] Running policy inference (batch_size={B})...")

    # Build curr_obs from dataset sample (first 2 frames = obs_horizon)
    curr_obs = {
        "image": sample["obs"]["image"][:2].unsqueeze(0).to(device),
        "agent_pos": sample["obs"]["agent_pos"][:2].unsqueeze(0).to(device),
    }

    print(f"  curr_obs.image shape:     {list(curr_obs['image'].shape)}")
    print(f"  curr_obs.image dtype:     {curr_obs['image'].dtype}")
    print(f"  curr_obs.image device:    {curr_obs['image'].device}")
    print(f"  curr_obs.agent_pos shape: {list(curr_obs['agent_pos'].shape)}")

    with torch.no_grad():
        action_norm = model.sample(curr_obs)
        # Also try sample_marginal_action directly
        action_norm2 = model.sample_marginal_action(curr_obs)

    print(f"  action (normalized) shape: {list(action_norm.shape)}")
    print(f"  action min: {action_norm.min().item():.4f}, max: {action_norm.max().item():.4f}")

    # Unnormalize
    action_raw = action_norm * action_scale[None, None, :] + action_offset[None, None, :]
    print(f"  action (raw) shape:       {list(action_raw.shape)}")
    print(f"  action raw min: {action_raw.min().item():.2f}, max: {action_raw.max().item():.2f}")

    assert list(action_norm.shape) == [B, 16, 2], f"action shape {action_norm.shape}"
    assert torch.isfinite(action_norm).all(), "action not finite"
    assert action_raw.min() >= 0, f"raw action min {action_raw.min()} < 0"
    assert action_raw.max() <= 512, f"raw action max {action_raw.max()} > 512"

    print("  policy inference: PASS")

    # ---- 5. Validation ----
    print("\n[5/5] Validation...")
    print(f"  action finite:           PASS")
    print(f"  action shape [1,16,2]:   PASS")
    print(f"  action range reasonable: PASS")

    print("\n" + "=" * 60)
    print("POLICY INFERENCE SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
