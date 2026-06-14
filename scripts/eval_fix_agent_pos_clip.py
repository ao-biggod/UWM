#!/usr/bin/env python3
"""Fix eval: agent_pos normalization + clip_sample check.

Runs 4 combinations:
  old:   no agent_pos norm, clip_sample=True
  A:     agent_pos norm,    clip_sample=True
  B:     no agent_pos norm, clip_sample=False
  C:     agent_pos norm,    clip_sample=False
"""
import argparse, json, os, sys, time, copy
from pathlib import Path
from functools import partial
from collections import deque

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


def build_model(device, clip_sample=True):
    from models.dp import ImageDiffusionPolicy, ImageObservationEncoder
    from models.dp import TransformerNoisePredictionNet
    shape_meta = {
        "obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                "agent_pos": {"shape": [2], "type": "low_dim"}},
        "action": {"shape": [2]},
    }
    obs_encoder = ImageObservationEncoder(
        shape_meta=shape_meta, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False,
        pretrained_weights=None,
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
        beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample,
    )
    return model.to(device)


def get_lowdim_normalizer(ckpt):
    """Get or reconstruct agent_pos normalizer."""
    if "lowdim_normalizer" in ckpt:
        ln = ckpt["lowdim_normalizer"]
        scale = np.array(ln["agent_pos"]["scale"])
        offset = np.array(ln["agent_pos"]["offset"])
    else:
        # Reconstruct from zarr (same as training)
        import zarr
        z = zarr.open("diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr", "r")
        ep_ends = z["meta/episode_ends"][:]
        # Use first 90 episodes (same as training)
        train_end = ep_ends[89]
        state = z["data/state"][:train_end, :2]
        min_val = state.min(axis=0)
        max_val = state.max(axis=0)
        scale = (max_val - min_val) / 2.0
        offset = (max_val + min_val) / 2.0
    return scale, offset


def obs_env_to_model(env_obs, device, norm_agent_pos, ap_scale=None, ap_offset=None):
    """Convert env obs to model input. Optionally normalize agent_pos."""
    img = env_obs["image"]
    agent_pos = env_obs["agent_pos"]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if isinstance(agent_pos, np.ndarray):
        agent_pos = torch.from_numpy(agent_pos)

    img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    ap = agent_pos.float()

    if norm_agent_pos:
        ap = (ap - torch.tensor(ap_offset)) / torch.tensor(ap_scale)

    return {
        "image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
        "agent_pos": ap.to(device).unsqueeze(0),
    }


def eval_run(model, action_scale, action_offset, ap_scale, ap_offset,
             norm_agent_pos, device, n_eps=50, label=""):
    model.eval()
    from models.dp import ImageDiffusionPolicy
    results = []
    ap_stats = []  # (before_norm, after_norm)
    act_stats = []  # (sampled_norm, sampled_raw)

    for ep in range(n_eps):
        seed = 100000 + ep
        env = MultiStepWrapper(
            PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
        env.seed(seed)
        obs = env.reset()

        # Collect first obs stats
        if ep == 0:
            ap_raw = obs["agent_pos"]
            ap_proc = (obs["agent_pos"] - ap_offset) / ap_scale if norm_agent_pos else obs["agent_pos"]
            print(f"  [{label}] ep0 obs agent_pos raw: "
                  f"min={ap_raw.min():.1f} max={ap_raw.max():.1f} "
                  f"mean={ap_raw.mean():.1f}")
            if norm_agent_pos:
                print(f"  [{label}] ep0 obs agent_pos norm: "
                      f"min={ap_proc.min():.3f} max={ap_proc.max():.3f} "
                      f"mean={ap_proc.mean():.3f}")

        rewards = []; done = False; step = 0

        while not done and step < 300:
            obs_model = obs_env_to_model(obs, device, norm_agent_pos, ap_scale, ap_offset)
            with torch.no_grad():
                action_norm = model.sample(obs_model)[0]
            action_raw = action_norm * torch.tensor(action_scale, device=device).float() + torch.tensor(action_offset, device=device).float()
            action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
            exec_actions = action_raw_np[:8]

            if ep == 0 and step == 0:
                print(f"  [{label}] ep0 step0 action_norm: min={action_norm.min():.3f} max={action_norm.max():.3f}")
                print(f"  [{label}] ep0 step0 action_raw:  min={action_raw_np.min():.1f} max={action_raw_np.max():.1f}")
                print(f"  [{label}] ep0 step0 exec_actions: {exec_actions[:3].tolist()}")

            obs, reward, done, info = env.step(exec_actions)
            rewards.append(float(reward))
            done = bool(np.all(done))
            step += 1

        max_r = float(max(rewards)) if rewards else 0.0
        results.append(max_r)
        if ep < 5:
            print(f"  [{label}] Ep {ep:3d}: max_reward={max_r:.4f}", flush=True)

    scores = np.array(results)
    print(f"\n  [{label}] ({n_eps}eps): mean={scores.mean():.4f} median={np.median(scores):.4f} "
          f"ep>0.5={(scores>0.5).sum()}/{n_eps}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/dp_pusht/run_20k_bs64/latest.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str,
                        default="outputs/eval_fix_agent_pos_clip")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Eval Fix: agent_pos normalization + clip_sample")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    an = ckpt["action_normalizer"]
    action_scale = np.array(an["scale"])
    action_offset = np.array(an["offset"])
    ap_scale, ap_offset = get_lowdim_normalizer(ckpt)

    print(f"\n  action_normalizer:  scale={action_scale}  offset={action_offset}")
    print(f"  agent_pos_normalizer: scale={ap_scale}  offset={ap_offset}")
    print(f"  lowdim_normalizer in ckpt: {'yes' if 'lowdim_normalizer' in ckpt else 'NO — reconstructed from zarr'}")

    # Expert action stats for reference
    import zarr
    z = zarr.open("diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr", "r")
    expert_act = z["data/action"][:]
    print(f"  expert action raw:  min=[{expert_act[:,0].min():.0f},{expert_act[:,1].min():.0f}] "
          f"max=[{expert_act[:,0].max():.0f},{expert_act[:,1].max():.0f}]")

    results = {}
    configs = [
        ("old", False, True),
        ("A", True, True),
        ("B", False, False),
        ("C", True, False),
    ]

    for label, norm_ap, clip_sample in configs:
        print(f"\n{'='*60}")
        print(f"[Run {label}] norm_agent_pos={norm_ap}  clip_sample={clip_sample}")
        print(f"{'='*60}")

        model = build_model(device, clip_sample=clip_sample)
        model.load_state_dict(ckpt["model"])
        model.eval()

        scores = eval_run(model, action_scale, action_offset,
                         ap_scale, ap_offset, norm_ap, device,
                         n_eps=args.n_eval_episodes, label=label)
        results[label] = {
            "norm_agent_pos": norm_ap, "clip_sample": clip_sample,
            "mean": float(scores.mean()), "median": float(np.median(scores)),
            "std": float(scores.std()), "ep_gt_05": int((scores>0.5).sum()),
        }

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Run':<8} {'norm_ap':<10} {'clip':<8} {'mean':>8} {'median':>8} {'ep>0.5':>8}")
    print("-" * 50)
    for label, norm_ap, clip_sample in configs:
        r = results[label]
        print(f"  {label:<6} {str(norm_ap):<10} {str(clip_sample):<8} {r['mean']:8.4f} {r['median']:8.4f} {r['ep_gt_05']:>6}/{args.n_eval_episodes}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
