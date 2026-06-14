#!/usr/bin/env python3
"""10-step UWM PushT training smoke test.

Replicates experiments/uwm/train.py training loop without eval/wandb/distributed.

Usage:
  cd /root/autodl-tmp/UWM_pushT
  source /root/miniconda3/bin/activate robodiff
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH=/root/autodl-tmp/UWM_pushT/unified-world-model-main:$PYTHONPATH
  python scripts/smoke_uwm_pusht_train.py \
    --zarr-path diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr \
    --device cuda:0 \
    --batch-size 1 \
    --num-steps 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    """Mirrors experiments/uwm/train.py:process_batch."""
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
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", type=str, default=None,
                        help="Save checkpoint to this path")
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.is_absolute():
        zarr_path = str(PROJECT_ROOT / zarr_path)
    else:
        zarr_path = str(zarr_path)
    device = torch.device(args.device)

    obs_horizon = 2
    action_horizon = 16

    print("=" * 60)
    print("UWM PushT 10-Step Training Smoke Test")
    print(f"  zarr:    {zarr_path}")
    print(f"  device:  {device}")
    print(f"  batch:   {args.batch_size}")
    print(f"  steps:   {args.num_steps}")
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
    train_set, val_set = make_pusht_dataset(
        name="pusht_train_smoke",
        zarr_path=zarr_path,
        shape_meta=shape_meta,
        seq_len=19,
        val_ratio=0.02,
        max_train_episodes=90,
        seed=42,
        normalize_action=True,
        normalize_lowdim=True,
    )
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}, Batches: {len(loader)}")
    print("  dataset: PASS")

    # ---- 2. Model ----
    print("\n[2/6] Creating UWM model...")
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params / 1e6:.1f}M, device: {device}")
    print("  model: PASS")

    # ---- 3. Optimizer ----
    print("\n[3/6] Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-6,
        betas=(0.9, 0.999), eps=1e-8,
    )
    # Mirror train.py: use GradScaler (though use_amp=False for smoke)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    print(f"  Optimizer: AdamW, lr={args.lr}")
    print("  optimizer: PASS")

    # ---- 4. Normalizer binding ----
    print("\n[4/6] Binding normalizers...")
    model.action_normalizer = train_set.action_normalizer
    model.lowdim_normalizer = train_set.lowdim_normalizer
    print(f"  action_normalizer: scale={model.action_normalizer.scale}")
    print("  normalizer: PASS")

    # ---- 5. Training loop ----
    print(f"\n[5/6] Running {args.num_steps} training steps...")

    # Warm up GPU memory tracking
    torch.cuda.reset_peak_memory_stats(device)

    data_iter = iter(loader)
    losses = []
    action_losses = []
    dynamics_losses = []

    for step in range(args.num_steps):
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # process_batch (mirrors train.py:134-136)
        curr_obs, next_obs, actions = process_batch(
            batch, obs_horizon, action_horizon, device)

        # Forward (mirrors train.py:140-143)
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            loss, info = model(curr_obs, next_obs, actions)

        # Backward (mirrors train.py:146-152)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        al = info["action_loss"]
        dl = info["dynamics_loss"]
        losses.append(loss_val)
        action_losses.append(al)
        dynamics_losses.append(dl)

        mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"  step {step:3d}: loss={loss_val:.6f}  action={al:.6f}  dynamics={dl:.6f}  mem={mem:.2f}GB")

        # Assert finite
        assert np.isfinite(loss_val), f"Loss is not finite at step {step}"
        assert np.isfinite(al), f"action_loss not finite at step {step}"
        assert np.isfinite(dl), f"dynamics_loss not finite at step {step}"

    # ---- 6. Summary ----
    print(f"\n[6/6] Summary...")
    print(f"  Steps completed: {len(losses)}")
    print(f"  Mean loss:       {np.mean(losses):.6f}")
    print(f"  Min loss:        {np.min(losses):.6f}")
    print(f"  Max loss:        {np.max(losses):.6f}")
    print(f"  Final loss:      {losses[-1]:.6f}")
    print(f"  Mean action:     {np.mean(action_losses):.6f}")
    print(f"  Mean dynamics:   {np.mean(dynamics_losses):.6f}")
    final_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"  Max GPU memory:  {final_mem:.2f} GB")

    # Sanity: loss should be positive and not exploding
    mean_loss = np.mean(losses)
    assert mean_loss > 0, f"Mean loss should be > 0, got {mean_loss}"
    assert mean_loss < 1e6, f"Mean loss should not explode, got {mean_loss}"

    print("\n  loss finite:          PASS")
    print("  action_loss finite:   PASS")
    print("  dynamics_loss finite: PASS")
    print("  no eval, no wandb, no checkpoint")

    # ---- Optional: Save checkpoint ----
    if args.save_path:
        print(f"\n[Save] Writing checkpoint to {args.save_path}...")
        save_dir = Path(args.save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        an = train_set.action_normalizer
        ln = train_set.lowdim_normalizer
        ckpt = {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "step": args.num_steps,
            "config": {
                "action_len": 16,
                "action_dim": 2,
                "obs_num_frames": 2,
                "n_action_steps": 8,
                "shape_meta": shape_meta,
            },
            "action_normalizer": {
                "scale": np.asarray(an.scale).tolist(),
                "offset": np.asarray(an.offset).tolist(),
            },
            "lowdim_normalizer": {
                "agent_pos": {
                    "scale": np.asarray(ln["agent_pos"].scale).tolist(),
                    "offset": np.asarray(ln["agent_pos"].offset).tolist(),
                },
            },
            "loss_history": {
                "losses": [float(x) for x in losses],
                "action_losses": [float(x) for x in action_losses],
                "dynamics_losses": [float(x) for x in dynamics_losses],
            },
        }
        torch.save(ckpt, args.save_path)
        size_mb = Path(args.save_path).stat().st_size / 1024**2
        print(f"  Saved: {args.save_path} ({size_mb:.1f} MB)")
        print(f"  Keys:  {list(ckpt.keys())}")

    print("\n" + "=" * 60)
    print("TRAINING SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
