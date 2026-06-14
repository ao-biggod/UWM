#!/usr/bin/env python3
"""Retroactively add EMA to existing converged DP-only checkpoint.

Loads the trained model, continues training for a few steps with EMA,
then saves both raw and EMA weights.
"""
import argparse, json, os, sys, time
from pathlib import Path
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent / "unified-world-model-main"
sys.path.insert(0, str(PROJECT_ROOT))


class EMAModel(nn.Module):
    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)


def collate_fn(batch):
    if isinstance(batch[0], dict):
        return {k: collate_fn([b[k] for b in batch]) for k in batch[0]}
    return torch.stack(batch)


def build_dataset(zarr_path):
    from datasets.pusht import make_pusht_dataset
    shape_meta = {"obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                           "agent_pos": {"shape": [2], "type": "low_dim"}},
                   "action": {"shape": [2]}}
    return make_pusht_dataset(name="pusht_ema", zarr_path=zarr_path, shape_meta=shape_meta,
        seq_len=19, val_ratio=0.02, max_train_episodes=90, seed=42,
        normalize_action=True, normalize_lowdim=True)


def build_model(device):
    from models.dp import ImageDiffusionPolicy, ImageObservationEncoder
    from models.dp import TransformerNoisePredictionNet
    shape_meta = {"obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                           "agent_pos": {"shape": [2], "type": "low_dim"}},
                   "action": {"shape": [2]}}
    obs_encoder = ImageObservationEncoder(shape_meta=shape_meta, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False, pretrained_weights=None,
        use_low_dim=True, use_language=False)
    model = ImageDiffusionPolicy(action_len=16, action_dim=2, obs_encoder=obs_encoder,
        noise_pred_net=partial(TransformerNoisePredictionNet, input_len=16, input_dim=2,
            timestep_embed_dim=256, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True),
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=True)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str,
                        default="outputs/dp_pusht/run_20k_bs64/latest.pt")
    parser.add_argument("--zarr-path", type=str,
                        default="diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ema-steps", type=int, default=500,
                        help="Steps to run EMA over (updates EMA shadow from existing weights)")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--output-dir", type=str, default="outputs/ablate_r1_ema")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Adding EMA (decay={args.ema_decay}) to {args.base_ckpt}")
    print(f"Running {args.ema_steps} ema warmup steps")

    # Load base checkpoint
    base = torch.load(args.base_ckpt, map_location=device)
    action_normalizer = base["action_normalizer"]

    # Build model and load weights
    model = build_model(device)
    model.load_state_dict(base["model"])
    print(f"  Loaded base model from step {base.get('step', 'unknown')}")

    # Create EMA from current weights
    ema = EMAModel(model, decay=args.ema_decay)

    # Get dataset for a few batches
    train_set, val_set = build_dataset(args.zarr_path)
    loader = DataLoader(train_set, batch_size=64, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)
    data_iter = iter(loader)

    # Run ema_steps forward + backward passes to build EMA
    # We do actual training steps (not just ema.update) to keep weights stable
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6,
                                   betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    obs_horizon, action_horizon = 2, 16

    print(f"\nRunning {args.ema_steps} EMA warmup steps...")
    t0 = time.time()
    for step in range(args.ema_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        curr_obs = {k: v[:, :obs_horizon].to(device) for k, v in batch["obs"].items()}
        action_target = batch["action"][:, obs_horizon - 1 : obs_horizon - 1 + action_horizon].to(device)

        model.train()
        loss = model(curr_obs, action_target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        if step % 100 == 0 or step == args.ema_steps - 1:
            print(f"  EMA step {step}: loss={loss.item():.6f}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Save checkpoint with both raw and EMA weights
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "model": state,
        "step": base.get("step", 20000) + args.ema_steps,
        "action_normalizer": action_normalizer,
        "ema_model": {k: v.cpu() for k, v in ema.shadow.items()},
        "ema_decay": args.ema_decay,
    }
    save_path = os.path.join(args.output_dir, "latest.pt")
    torch.save(ckpt, save_path)
    print(f"\nSaved: {save_path}")
    print(f"  Raw weights from step {ckpt['step']}")
    print(f"  EMA weights with decay={args.ema_decay}")
    print(f"\nNext: evaluate with scripts/eval_ablation.py --checkpoint {save_path}")


if __name__ == "__main__":
    main()
