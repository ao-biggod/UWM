#!/usr/bin/env python3
"""Deterministic rollout + video recording for PushT env.

Usage:
  python scripts/record_rollout_video.py \
    --checkpoint outputs/uwm_pusht_r1_ft/lambda_0.05/latest.pt \
    --seed 100000 --device cuda:0 \
    --output outputs/videos/best.mp4
"""

import argparse, json, os, sys, random, time
from pathlib import Path
import numpy as np
import torch
import imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_cudnn_deterministic():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def build_model(device, ckpt):
    from models.uwm import UnifiedWorldModel
    from models.uwm.obs_encoder import UWMObservationEncoder

    cfg = ckpt.get("config", {})
    cond_type = ckpt.get("conditioning_type") or cfg.get("conditioning_type", "adaln")
    dp_only = ckpt.get("dp_only", cfg.get("dp_only", False))
    dyn_weight = cfg.get("dynamics_loss_weight", 1.0)
    sa_mask = cfg.get("self_attn_mask", None)

    sm = {"obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                  "agent_pos": {"shape": [2], "type": "low_dim"}},
          "action": {"shape": [2]}}
    oe = UWMObservationEncoder(
        shape_meta=sm, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False,
        vision_backbone="resnet", use_low_dim=True, use_language=False,
    )
    m = UnifiedWorldModel(
        action_len=16, action_dim=2, obs_encoder=oe,
        embed_dim=768, timestep_embed_dim=512,
        latent_patch_shape=[2, 4, 4], depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, num_registers=8,
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=False,
        conditioning_type=cond_type, dp_only=dp_only,
        dynamics_loss_weight=dyn_weight, self_attn_mask=sa_mask,
    )
    m.load_state_dict(ckpt["model"], strict=False)
    return m.to(device)


def rollout_and_record(checkpoint_path, seed, device, output_path, fps=10):
    """Run deterministic rollout and save mp4 video."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    an = ckpt["action_normalizer"]
    action_scale = np.array(an["scale"])
    action_offset = np.array(an["offset"])
    ln = ckpt["lowdim_normalizer"]["agent_pos"]
    ap_scale = np.array(ln["scale"])
    ap_offset = np.array(ln["offset"])

    model = build_model(device, ckpt)
    del ckpt
    model.eval()

    set_cudnn_deterministic()
    seed_everything(seed)

    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
    env.seed(seed)
    obs = env.reset()

    frames = []
    rewards = []
    step = 0
    done = False

    while not done and step < 300:
        # Render current frame
        frame = env.render(mode="rgb_array")
        frames.append(frame)

        # Preprocess observation
        img = obs["image"]
        agent_pos = obs["agent_pos"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(agent_pos, np.ndarray):
            agent_pos = torch.from_numpy(agent_pos)

        img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        ap = agent_pos.float().to(device)
        ap = (ap - torch.tensor(ap_offset, device=device).float()) / torch.tensor(ap_scale, device=device).float()

        obs_model = {"image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
                     "agent_pos": ap.unsqueeze(0)}

        with torch.no_grad():
            action_norm = model.sample(obs_model)[0]

        action_raw = (action_norm
                      * torch.tensor(action_scale, device=device).float()
                      + torch.tensor(action_offset, device=device).float())
        action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

        obs, reward, done, _ = env.step(action_raw_np[:8])
        rewards.append(float(reward))
        if np.all(done):
            break
        step += 1

    # Render final frame
    frame = env.render(mode="rgb_array")
    frames.append(frame)
    env.close()

    max_reward = float(max(rewards)) if rewards else 0.0

    # Save video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    size_mb = os.path.getsize(output_path) / 1024 / 1024

    print(f"  seed={seed} steps={step+1} max_reward={max_reward:.4f} frames={len(frames)}")
    print(f"  saved: {output_path} ({size_mb:.1f} MB)")

    return {"seed": seed, "steps": step + 1, "max_reward": max_reward, "frames": len(frames)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Recording rollout: ckpt={args.checkpoint} seed={args.seed}")
    result = rollout_and_record(args.checkpoint, args.seed, device, args.output, args.fps)

    # Save metadata alongside video
    meta_path = args.output.replace(".mp4", "_meta.json").replace(".gif", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, **result}, f, indent=2)

    return result["max_reward"]


if __name__ == "__main__":
    main()
