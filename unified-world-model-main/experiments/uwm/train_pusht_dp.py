#!/usr/bin/env python3
"""Pure Diffusion Policy (no world model) PushT training — ablation of UWM dynamics head.

Same dataset, same vision encoder, same transformer backbone, same hyperparams as UWM,
but without the dynamics/video prediction component.

Usage:
  cd /root/autodl-tmp/UWM_pushT
  source /root/miniconda3/bin/activate robodiff
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH=unified-world-model-main:$PYTHONPATH

  python unified-world-model-main/experiments/uwm/train_pusht_dp.py \
    --num-steps 20000 --batch-size 64 --device cuda:0 \
    --output-dir outputs/dp_pusht/run_20k_bs64
"""

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def collate_fn(batch):
    if isinstance(batch[0], dict):
        return {k: collate_fn([b[k] for b in batch]) for k in batch[0]}
    return torch.stack(batch)


def build_dataset(zarr_path):
    from datasets.pusht import make_pusht_dataset
    shape_meta = {
        "obs": {
            "image": {"shape": [96, 96, 3], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }
    return make_pusht_dataset(
        name="pusht_train", zarr_path=zarr_path, shape_meta=shape_meta,
        seq_len=19, val_ratio=0.02, max_train_episodes=90, seed=42,
        normalize_action=True, normalize_lowdim=True,
    )


def build_model(device):
    from models.dp import ImageDiffusionPolicy, ImageObservationEncoder
    from models.dp import TransformerNoisePredictionNet

    shape_meta = {
        "obs": {
            "image": {"shape": [96, 96, 3], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }

    obs_encoder = ImageObservationEncoder(
        shape_meta=shape_meta, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False,
        pretrained_weights=None,  # ResNet from scratch — same as UWM
        use_low_dim=True, use_language=False,
    )

    model = ImageDiffusionPolicy(
        action_len=16, action_dim=2,
        obs_encoder=obs_encoder,
        noise_pred_net=partial(
            TransformerNoisePredictionNet,
            input_len=16, input_dim=2,
            timestep_embed_dim=256, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        ),
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=True,
    )
    return model.to(device)


def save_checkpoint(model, train_set, step, save_path):
    an = train_set.action_normalizer
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    # Get state dict on CPU
    state = {}
    for k, v in model.state_dict().items():
        state[k] = v.cpu()
    ckpt = {
        "model": state,
        "step": step,
        "config": {"action_len": 16, "action_dim": 2, "obs_num_frames": 2, "n_action_steps": 8},
        "action_normalizer": {
            "scale": np.asarray(an.scale).tolist(),
            "offset": np.asarray(an.offset).tolist(),
        },
    }
    torch.save(ckpt, save_path)
    return ckpt


def run_eval(model, val_set, device, obs_horizon, action_horizon):
    loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.eval()
    total_loss = 0.0
    n = 0
    max_eval = min(20, len(loader))
    for i, batch in enumerate(loader):
        if i >= max_eval:
            break
        curr_obs = {k: v[:, :obs_horizon].to(device) for k, v in batch["obs"].items()}
        action_target = batch["action"][:, obs_horizon - 1 : obs_horizon - 1 + action_horizon].to(device)
        with torch.no_grad():
            loss = model(curr_obs, action_target)
        total_loss += loss.item()
        n += 1
    model.train()
    return {"val_loss": total_loss / n}


def main():
    parser = argparse.ArgumentParser(description="DP PushT Training (no world model)")
    parser.add_argument("--zarr-path", type=str,
                        default="/root/autodl-tmp/UWM_pushT/diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="outputs/dp_pusht/run1")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    obs_horizon = 2
    action_horizon = 16
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DP PushT Training (no world model / dynamics head)")
    print(f"  device:      {device}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  num_steps:   {args.num_steps}")
    print(f"  lr:          {args.lr}")
    print(f"  save_every:  {args.save_every}")
    print(f"  eval_every:  {args.eval_every}")
    print(f"  output_dir:  {args.output_dir}")
    print("=" * 60)

    # Dataset
    print("\n[1/5] Creating dataset...")
    train_set, val_set = build_dataset(args.zarr_path)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}, Batches: {len(loader)}")

    # Model
    print("\n[2/5] Creating model...")
    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params / 1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-6,
        betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"] + 1
        print(f"  Resumed from step {start_step}")

    # Log
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    log_file = open(log_path, "a")

    # Training loop
    print(f"\n[3/5] Training ({args.num_steps} steps)...")
    data_iter = iter(loader)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    for step in range(start_step, args.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Extract curr_obs (first obs_horizon frames) and action (following action_horizon frames)
        curr_obs = {k: v[:, :obs_horizon].to(device) for k, v in batch["obs"].items()}
        action_target = batch["action"][:, obs_horizon - 1 : obs_horizon - 1 + action_horizon].to(device)

        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            loss = model(curr_obs, action_target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        log_entry = {
            "step": step,
            "loss": loss.item(),
            "lr": args.lr,
            "gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        }

        # Eval
        if step > 0 and args.eval_every > 0 and step % args.eval_every == 0:
            eval_stats = run_eval(model, val_set, device, obs_horizon, action_horizon)
            log_entry.update(eval_stats)
            print(f"\n[Eval] step={step}: {json.dumps(eval_stats)}")

        # Save
        if step > 0 and args.save_every > 0 and step % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_step{step:07d}.pt")
            save_checkpoint(model, train_set, step, save_path)
            latest_path = os.path.join(args.output_dir, "latest.pt")
            save_checkpoint(model, train_set, step, latest_path)
            print(f"\n[Save] step={step}: {save_path}")

        log_file.write(json.dumps(log_entry) + "\n")

        if step % 100 == 0 or step < 10:
            elapsed = time.time() - t0
            sps = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            print(f"  step {step:6d}: loss={loss.item():.4f}  "
                  f"mem={log_entry['gpu_mem_gb']:.2f}GB  {sps:.1f} s/s")

    # Final save
    print(f"\n[4/5] Final save...")
    final_path = os.path.join(args.output_dir, "latest.pt")
    save_checkpoint(model, train_set, args.num_steps - 1, final_path)
    log_file.close()
    print(f"  Saved: {final_path}")
    print(f"  Log:   {log_path}")

    # Summary
    print(f"\n[5/5] Training complete.")
    final_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    elapsed = time.time() - t0
    print(f"  Steps: {args.num_steps}")
    print(f"  Time:  {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Max GPU memory: {final_mem:.2f} GB")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
