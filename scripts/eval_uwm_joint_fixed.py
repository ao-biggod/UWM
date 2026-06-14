#!/usr/bin/env python3
"""Eval UWM full/joint checkpoint with agent_pos normalization fix.

Tests clip_sample=True/False. No EMA available in checkpoint (raw only).
"""
import argparse, json, os, sys, time
from pathlib import Path
from collections import deque

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


def build_model(device, clip_sample=True):
    from models.uwm import UnifiedWorldModel
    from models.uwm.obs_encoder import UWMObservationEncoder
    shape_meta = {
        "obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                "agent_pos": {"shape": [2], "type": "low_dim"}},
        "action": {"shape": [2]},
    }
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
        beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample,
    )
    return model.to(device)


def obs_env_to_uwm(env_obs, device, norm_ap, ap_scale, ap_offset):
    """Convert env obs to UWM input. Optionally normalize agent_pos."""
    img = env_obs["image"]
    agent_pos = env_obs["agent_pos"]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if isinstance(agent_pos, np.ndarray):
        agent_pos = torch.from_numpy(agent_pos)

    img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    ap = agent_pos.float().to(device)

    if norm_ap:
        ap_off = torch.tensor(ap_offset, device=device).float()
        ap_sc = torch.tensor(ap_scale, device=device).float()
        ap = (ap - ap_off) / ap_sc

    return {
        "image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
        "agent_pos": ap.to(device).unsqueeze(0),
    }


def eval_run(model, action_scale, action_offset, ap_scale, ap_offset,
             norm_ap, device, n_eps=50, label=""):
    model.eval()
    results = []

    for ep in range(n_eps):
        seed = 100000 + ep
        env = MultiStepWrapper(
            PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
        env.seed(seed)
        obs = env.reset()

        if ep == 0:
            ap_raw = obs["agent_pos"]
            print(f"  [{label}] ep0 agent_pos raw: "
                  f"x∈[{ap_raw[:,0].min():.0f},{ap_raw[:,0].max():.0f}] "
                  f"y∈[{ap_raw[:,1].min():.0f},{ap_raw[:,1].max():.0f}]")
            if norm_ap:
                ap_n = (ap_raw - ap_offset) / ap_scale
                print(f"  [{label}] ep0 agent_pos norm: "
                      f"x∈[{ap_n[:,0].min():.3f},{ap_n[:,0].max():.3f}] "
                      f"y∈[{ap_n[:,1].min():.3f},{ap_n[:,1].max():.3f}]")

        rewards = []; done = False; step = 0

        while not done and step < 300:
            obs_uwm = obs_env_to_uwm(obs, device, norm_ap, ap_scale, ap_offset)

            with torch.no_grad():
                action_norm = model.sample(obs_uwm)[0]

            action_raw = action_norm * torch.tensor(action_scale, device=device).float() + torch.tensor(action_offset, device=device).float()
            action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
            exec_actions = action_raw_np[:8]

            if ep == 0 and step == 0:
                print(f"  [{label}] step0 action_norm: min={action_norm.min():.3f} max={action_norm.max():.3f}")
                print(f"  [{label}] step0 action_raw:  min={action_raw_np.min():.1f} max={action_raw_np.max():.1f}")

            obs, reward, done, info = env.step(exec_actions)
            rewards.append(float(reward))
            done = bool(np.all(done))
            step += 1

        max_r = float(max(rewards)) if rewards else 0.0
        results.append(max_r)
        if ep < 5 or ep % 10 == 0:
            print(f"  [{label}] Ep {ep:3d}: max_reward={max_r:.4f}", flush=True)

    scores = np.array(results)
    print(f"\n  [{label}] ({n_eps}eps): mean={scores.mean():.4f} median={np.median(scores):.4f} "
          f"ep>0.5={(scores>0.5).sum()}/{n_eps}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="artifacts_keep/uwm_20k/checkpoint_20k_latest.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str,
                        default="outputs/eval_uwm_joint_fixed")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Eval UWM Joint — agent_pos normalization fix")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    an = ckpt["action_normalizer"]
    action_scale = np.array(an["scale"])
    action_offset = np.array(an["offset"])
    ln = ckpt["lowdim_normalizer"]["agent_pos"]
    ap_scale = np.array(ln["scale"])
    ap_offset = np.array(ln["offset"])

    print(f"\n  action_normalizer:  scale={action_scale}  offset={action_offset}")
    print(f"  agent_pos_normalizer: scale={ap_scale}  offset={ap_offset}")
    has_ema = "ema" in ckpt
    print(f"  EMA in ckpt: {has_ema}")

    results = {}
    configs = [
        ("uwm_old", False, True),     # old eval, norm_ap=False, clip_sample=True
        ("uwm_C", True, False),        # norm_ap=True, clip_sample=False
        ("uwm_A", True, True),         # norm_ap=True, clip_sample=True
    ]

    for label, norm_ap, clip_sample in configs:
        print(f"\n{'='*60}")
        print(f"[{label}] norm_agent_pos={norm_ap}  clip_sample={clip_sample}")
        print(f"{'='*60}")

        model = build_model(device, clip_sample=clip_sample)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()

        scores = eval_run(model, action_scale, action_offset,
                         ap_scale, ap_offset, norm_ap, device,
                         n_eps=args.n_eval_episodes, label=label)
        results[label] = {
            "norm_agent_pos": norm_ap, "clip_sample": clip_sample,
            "mean": float(scores.mean()), "median": float(np.median(scores)),
            "std": float(scores.std()), "ep_gt_05": int((scores>0.5).sum()),
            "checkpoint": args.checkpoint,
        }

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':<12} {'norm_ap':<10} {'clip':<8} {'mean':>8} {'median':>8} {'ep>0.5':>8}")
    print("-" * 56)
    for label in ["uwm_old", "uwm_A", "uwm_C"]:
        r = results[label]
        print(f"  {label:<10} {str(r['norm_agent_pos']):<10} {str(r['clip_sample']):<8} {r['mean']:8.4f} {r['median']:8.4f} {r['ep_gt_05']:>6}/{args.n_eval_episodes}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
