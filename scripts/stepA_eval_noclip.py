#!/usr/bin/env python3
"""Step A: Eval existing checkpoints with clip_sample=False (inference only, no retrain)."""
import argparse, json, os, sys, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))


def run_lowdim_eval(checkpoint, device, n_episodes, clip_sample):
    """Eval lowdim oracle with modified clip_sample."""
    from scripts.diag_lowdim_dp import LowdimStatePolicy
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    ckpt = torch.load(checkpoint, map_location=device)
    model = LowdimStatePolicy().to(device)
    model.load_state_dict(ckpt["model"])
    model.noise_scheduler.config.clip_sample = clip_sample
    model.eval()

    an = ckpt["action_normalizer"]
    a_mean = torch.tensor(an["offset"], device=device).float()
    a_std = torch.tensor(an["scale"], device=device).float()
    sn = ckpt["state_normalizer"]
    s_mean = torch.tensor(sn["mean"], device=device).float()
    s_std = torch.tensor(sn["std"], device=device).float()

    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
    results = []
    all_sampled_norm = []
    all_sampled_raw = []

    for ep in range(n_episodes):
        env.seed(ep)
        obs = env.reset()
        rewards = []
        state_buffer = []
        done = False
        step = 0

        while not done and step < 300:
            inner = env.env
            agent_p = np.array(inner.agent.position)
            block_p = np.array(inner.block.position)
            block_ang = inner.block.angle
            full_state = np.concatenate([agent_p, block_p, [block_ang]])
            state_buffer.append(full_state)
            if len(state_buffer) > 2:
                state_buffer = state_buffer[-2:]

            if len(state_buffer) < 2:
                obs, reward, done, info = env.step(np.zeros((8, 2)))
                rewards.append(float(reward))
                step += 1
                continue

            state_np = np.stack(state_buffer)
            state_t = (torch.from_numpy(state_np).float().to(device).unsqueeze(0) - s_mean) / s_std

            with torch.no_grad():
                action_norm = model.sample(state_t)[0]

            all_sampled_norm.append(action_norm.cpu().numpy())
            action_raw = action_norm * a_std + a_mean
            all_sampled_raw.append(action_raw.cpu().numpy())

            action_exec = action_raw[:8].cpu().numpy()
            obs, reward, done, info = env.step(action_exec)
            rewards.append(float(reward))
            step += 1

        results.append({
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "steps": step,
        })
        if ep < 5 or ep % 10 == 0:
            print(f"  Ep {ep:3d}: max_reward={results[-1]['max_reward']:.4f}")

    all_norm = np.concatenate([a.reshape(-1, 2) for a in all_sampled_norm], axis=0)
    all_raw = np.concatenate([a.reshape(-1, 2) for a in all_sampled_raw], axis=0)

    mean_max = float(np.mean([r["max_reward"] for r in results]))
    print(f"\n  Mean max_reward: {mean_max:.4f}")
    print(f"  Sampled norm: min={all_norm.min(axis=0)} max={all_norm.max(axis=0)} mean={all_norm.mean(axis=0)}")
    print(f"  Sampled raw:  min={all_raw.min(axis=0)} max={all_raw.max(axis=0)} mean={all_raw.mean(axis=0)}")

    return mean_max, all_norm, all_raw


def run_dp_only_eval(checkpoint, device, n_episodes, clip_sample):
    """Eval DP-only image model with modified clip_sample."""
    from models.dp import ImageDiffusionPolicy, ImageObservationEncoder
    from models.dp import TransformerNoisePredictionNet
    from functools import partial
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    shape_meta = {
        "obs": {"image": {"shape": [96, 96, 3], "type": "rgb"}, "agent_pos": {"shape": [2], "type": "low_dim"}},
        "action": {"shape": [2]},
    }
    obs_encoder = ImageObservationEncoder(
        shape_meta=shape_meta, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False, pretrained_weights=None,
        use_low_dim=True, use_language=False,
    )
    model = ImageDiffusionPolicy(
        action_len=16, action_dim=2,
        obs_encoder=obs_encoder,
        noise_pred_net=partial(TransformerNoisePredictionNet, input_len=16, input_dim=2,
                               timestep_embed_dim=256, embed_dim=768, depth=12, num_heads=12,
                               mlp_ratio=4, qkv_bias=True),
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample,
    )
    model.to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    # Scheduler is on DiffusionPolicy base class, not noise_pred_net
    model.noise_scheduler.config.clip_sample = clip_sample
    model.eval()

    an = ckpt["action_normalizer"]
    a_scale = torch.tensor(an["scale"], device=device).float()
    a_offset = torch.tensor(an["offset"], device=device).float()

    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
    results = []
    all_sampled_norm = []
    all_sampled_raw = []

    for ep in range(n_episodes):
        env.seed(ep)
        obs = env.reset()
        rewards = []
        done = False
        step = 0

        while not done and step < 300:
            img = obs["image"]
            agent_pos = obs["agent_pos"]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if isinstance(agent_pos, np.ndarray):
                agent_pos = torch.from_numpy(agent_pos)
            img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            obs_dict = {
                "image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
                "agent_pos": agent_pos.float().to(device).unsqueeze(0),
            }

            with torch.no_grad():
                action_norm = model.sample(obs_dict)[0]

            all_sampled_norm.append(action_norm.cpu().numpy())
            action_raw = action_norm * a_scale + a_offset
            all_sampled_raw.append(action_raw.cpu().numpy())

            action_exec = action_raw[:8].cpu().numpy()
            obs, reward, done, info = env.step(action_exec)
            rewards.append(float(reward))
            step += 1

        results.append({
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "steps": step,
        })
        if ep < 5 or ep % 10 == 0:
            print(f"  Ep {ep:3d}: max_reward={results[-1]['max_reward']:.4f}")

    all_norm = np.concatenate([a.reshape(-1, 2) for a in all_sampled_norm], axis=0)
    all_raw = np.concatenate([a.reshape(-1, 2) for a in all_sampled_raw], axis=0)

    mean_max = float(np.mean([r["max_reward"] for r in results]))
    print(f"\n  Mean max_reward: {mean_max:.4f}")
    print(f"  Sampled norm: min={all_norm.min(axis=0)} max={all_norm.max(axis=0)} mean={all_norm.mean(axis=0)}")
    print(f"  Sampled raw:  min={all_raw.min(axis=0)} max={all_raw.max(axis=0)} mean={all_raw.mean(axis=0)}")

    return mean_max, all_norm, all_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lowdim", "dp_only"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--clip-sample", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--output-dir", type=str, default="outputs/stepA_noclip")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Step A: {args.model} eval with clip_sample={args.clip_sample} ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Episodes: {args.n_episodes}")

    t0 = time.time()

    if args.model == "lowdim":
        score, norm, raw = run_lowdim_eval(args.checkpoint, device, args.n_episodes, args.clip_sample)
    else:
        score, norm, raw = run_dp_only_eval(args.checkpoint, device, args.n_episodes, args.clip_sample)

    elapsed = time.time() - t0

    model_name = "lowdim_oracle" if args.model == "lowdim" else "dp_only"
    clip_str = "noclip" if not args.clip_sample else "clip"
    tag = f"{model_name}_{clip_str}"

    summary = {
        "model": model_name,
        "clip_sample": args.clip_sample,
        "checkpoint": args.checkpoint,
        "n_episodes": args.n_episodes,
        "mean_max_reward": score,
        "sampled_norm_min": norm.min(axis=0).tolist(),
        "sampled_norm_max": norm.max(axis=0).tolist(),
        "sampled_raw_min": raw.min(axis=0).tolist(),
        "sampled_raw_max": raw.max(axis=0).tolist(),
        "elapsed_sec": elapsed,
    }
    log_path = os.path.join(args.output_dir, f"eval_{tag}.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to: {log_path}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\n  >>> {tag}: mean_max_reward = {score:.4f} <<<")


if __name__ == "__main__":
    main()
