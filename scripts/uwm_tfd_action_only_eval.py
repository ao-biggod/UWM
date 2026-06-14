#!/usr/bin/env python3
"""UWM-TfD-KP-ActionOnly Bridge.

Loads E4 TransformerForDiffusion EMA checkpoint and evaluates it
using the UWM eval pipeline (PushTKeypointsEnv, same 50 seeds).

Key difference from E4 standalone: this uses UWM-compatible obs_dict
interface, extracting 20D keypoint from PushTKeypointsEnv.
No video. No training. No DualNoisePredictionNet modification.
"""
import argparse, json, os, sys, time
from pathlib import Path
from collections import deque

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

# Reuse E4's model definition via importlib
import importlib.util
_e4_spec = importlib.util.spec_from_file_location(
    "e4_mod", str(Path(__file__).resolve().parent / "kp_transformer_for_diffusion.py"))
_e4_mod = importlib.util.module_from_spec(_e4_spec)
_e4_spec.loader.exec_module(_e4_mod)
TransformerForDiffusionPolicy = _e4_mod.TransformerForDiffusionPolicy

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def get_keypoint_20d_from_env(env):
    """Extract 20D keypoint obs from PushTKeypointsEnv internals.
    The env wraps PushTKeypointsEnv which has keypoint info."""
    # PushTKeypointsEnv observation: [data, mask] where
    # data = [keypoint(18D), agent_pos(2D)] = 20D
    # mask = 20D binary mask
    # MultiStepWrapper returns (n_obs_steps, 2*D) = (2, 40)
    # where first half = frame data, second half = mask
    inner = env
    # Walk through wrappers
    while hasattr(inner, 'env'):
        inner = inner.env
        if isinstance(inner, PushTKeypointsEnv):
            return inner
    # If it's directly MultiStepWrapper(PushTKeypointsEnv), check
    if hasattr(env, 'get_obs'):
        # MultiStepWrapper.get_obs() returns stacked obs
        pass
    return None


def load_e4_model(ckpt_path, device):
    """Load E4 TransformerForDiffusion model and EMA checkpoint."""
    sd = torch.load(ckpt_path, map_location=device)

    config = sd.get("config", {})
    # Build model matching E4
    model = TransformerForDiffusionPolicy(
        obs_dim=config.get("obs_dim", 20),
        action_dim=config.get("action_dim", 2),
        horizon=config.get("horizon", 16),
        n_obs_steps=2,
        n_layer=config.get("n_layer", 8),
        n_head=config.get("n_head", 4),
        n_emb=config.get("n_emb", 256),
        causal_attn=True, time_as_cond=True, obs_as_cond=True,
        n_cond_layers=0,
        num_train_timesteps=100, num_inference_steps=config.get("inf_steps", 100),
    ).to(device)
    model.eval()

    # Load raw weights first
    raw_state = sd["model"]
    missing, unexpected = model.load_state_dict(raw_state, strict=False)

    # Load normalizer
    an = sd["action_normalizer"]
    if isinstance(an, dict):
        action_offset = np.array(an["offset"])
        action_scale = np.array(an["scale"])
    else:
        action_offset = an.offset; action_scale = an.scale

    sn = sd["state_normalizer"]
    if isinstance(sn, dict):
        state_offset = np.array(sn["offset"])
        state_scale = np.array(sn["scale"])
    else:
        state_offset = sn.offset; state_scale = sn.scale

    info = {
        "missing": missing, "unexpected": unexpected,
        "action_offset": action_offset, "action_scale": action_scale,
        "state_offset": state_offset, "state_scale": state_scale,
        "step": sd.get("step", "unknown"),
        "n_params": sum(p.numel() for p in model.parameters()),
        "config": config,
    }
    return model, action_scale, action_offset, state_scale, state_offset, info


def evaluate(model, action_scale, action_offset, state_scale, state_offset,
             device, n_eps=50):
    """Fixed-buffer eval, same as E2/E3/E4 but using PushTKeypointsEnv keypoint."""
    model.eval()
    results = []
    all_raw_actions = []

    for ep in range(n_eps):
        seed = 100000 + ep
        env = MultiStepWrapper(
            PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0),
            n_obs_steps=2, n_action_steps=8)
        env.seed(seed)
        raw_obs = env.reset()

        # raw_obs from MultiStepWrapper(PushTKeypointsEnv):
        # shape (2, 40) = [frame0_data(20) + frame0_mask(20), frame1_data(20) + frame1_mask(20)]
        Do = raw_obs.shape[-1] // 2  # 20
        obs_buffer = deque(maxlen=2)
        obs_buffer.append(raw_obs[0, :Do])   # frame0: 20D keypoint data
        obs_buffer.append(raw_obs[1, :Do])   # frame1: 20D keypoint data

        rewards = []; done = False; step = 0

        while not done and step < 300:
            state_np = np.stack(list(obs_buffer))  # (2, 20)
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)  # (1, 2, 20)
            # Normalize
            state_norm = (state_t - torch.tensor(state_offset, device=device).float()) / torch.tensor(state_scale, device=device).float()

            with torch.no_grad():
                action_norm = model.sample(state_norm)[0]  # (16, 2)

            # Unnormalize
            action_raw = action_norm * torch.tensor(action_scale, device=device).float() + torch.tensor(action_offset, device=device).float()
            action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

            # UWM contract: execute pred[0:8]
            exec_actions = action_raw_np[:8]

            if ep == 0 and step == 0:
                all_raw_actions.append(exec_actions.copy())

            raw_obs, reward, done, info = env.step(exec_actions)
            rewards.append(float(reward))
            done = bool(np.all(done))
            Do = raw_obs.shape[-1] // 2
            obs_buffer.append(raw_obs[1, :Do])
            step += 1

        max_r = float(max(rewards)) if rewards else 0.0
        results.append(max_r)
        if ep < 5 or ep % 10 == 0:
            print(f"  Ep {ep:3d}: max_reward={max_r:.4f}", flush=True)

    scores = np.array(results)
    print(f"\n  eval ({n_eps}eps): mean={scores.mean():.4f} std={scores.std():.4f} "
          f"median={np.median(scores):.4f} ep>0.5={(scores>0.5).sum()}/{n_eps}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/e4_kp_tfd_uwm/latest.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str,
                        default="outputs/uwm_tfd_action_only_eval")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("UWM-TfD-KP-ActionOnly Bridge")
    print("  Loading E4 checkpoint → eval via PushTKeypointsEnv 20D keypoint")
    print("=" * 60)

    # ── Sanity A: Load checkpoint
    print("\n[Sanity A: Checkpoint load]")
    model, act_scale, act_offset, st_scale, st_offset, info = \
        load_e4_model(args.checkpoint, device)

    print(f"  checkpoint step: {info['step']}")
    print(f"  model params:     {info['n_params']/1e6:.2f}M")
    print(f"  missing keys:     {len(info['missing'])}")
    if info['missing']:
        for k in info['missing'][:5]:
            print(f"    missing: {k}")
    print(f"  unexpected keys:  {len(info['unexpected'])}")
    if info['unexpected']:
        for k in info['unexpected'][:5]:
            print(f"    unexpected: {k}")

    # ── Sanity B: One env observation
    print("\n[Sanity B: Env observation]")
    env_test = MultiStepWrapper(
        PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0),
        n_obs_steps=2, n_action_steps=8)
    env_test.seed(100000)
    raw_obs = env_test.reset()
    Do = raw_obs.shape[-1] // 2
    obs_f0 = raw_obs[0, :Do]  # 20D: keypoint(18) + agent_pos(2)
    obs_f1 = raw_obs[1, :Do]

    print(f"  raw_obs shape:     {raw_obs.shape}")
    print(f"  20D per frame:     {Do}")
    print(f"  frame0 20D:        {obs_f0}")
    print(f"  frame1 20D:        {obs_f1}")
    print(f"  frame0 range:      [{obs_f0.min():.1f}, {obs_f0.max():.1f}]")
    print(f"  agent_pos(frame0): [{obs_f0[18]:.1f}, {obs_f0[19]:.1f}]")

    obs_2f = np.stack([obs_f0, obs_f1])  # (2, 20)
    obs_t = torch.from_numpy(obs_2f).float().to(device).unsqueeze(0)
    obs_norm = (obs_t - torch.tensor(st_offset, device=device).float()) / torch.tensor(st_scale, device=device).float()
    print(f"  obs_20d shape:     {obs_t.shape}")
    print(f"  normalized range:  [{obs_norm.min():.3f}, {obs_norm.max():.3f}]")
    print(f"  state_scale:       {st_scale}")
    print(f"  state_offset:      {st_offset}")

    # ── Sanity C: Sample one action
    print("\n[Sanity C: Sample one action]")
    with torch.no_grad():
        act_norm = model.sample(obs_norm)
    print(f"  action_norm shape:     {act_norm.shape}")
    print(f"  action_norm range:     [{act_norm.min():.3f}, {act_norm.max():.3f}]")
    print(f"  action_norm mean/std:  {act_norm.mean():.3f} / {act_norm.std():.3f}")

    act_raw = act_norm[0] * torch.tensor(act_scale, device=device).float() + torch.tensor(act_offset, device=device).float()
    act_raw_np = np.clip(act_raw.cpu().numpy(), 0.0, 512.0)
    print(f"  action_raw range:      [{act_raw_np.min():.1f}, {act_raw_np.max():.1f}]")
    print(f"  action_raw mean/std:   {act_raw_np.mean():.1f} / {act_raw_np.std():.1f}")
    print(f"  action_scale:          {act_scale}")
    print(f"  action_offset:         {act_offset}")
    exec8 = act_raw_np[:8]
    print(f"  exec8 (pred[0:8]):     {exec8}")
    print(f"  exec8 range:           [{exec8[:,0].min():.1f},{exec8[:,0].max():.1f}] x [{exec8[:,1].min():.1f},{exec8[:,1].max():.1f}]")

    # ── Sanity D: 5 episode eval
    print(f"\n{'='*60}")
    print("[Sanity D: 5 episode eval]")
    scores_5 = evaluate(model, act_scale, act_offset, st_scale, st_offset, device, n_eps=5)
    if scores_5.mean() < 0.8:
        print(f"\nWARNING: 5ep mean={scores_5.mean():.4f} < 0.8. Check:")
        print("  - keypoint extraction correct?")
        print("  - normalizer matches E4?")
        print("  - EMA loaded?")
        print("  - scheduler 100-step DDPM?")
        print("  - exec slice pred[0:8]?")
        return

    # ── Full 50 episode eval
    print(f"\n{'='*60}")
    print(f"[Full eval: {args.n_eval_episodes} episodes]")
    scores = evaluate(model, act_scale, act_offset, st_scale, st_offset, device,
                      n_eps=args.n_eval_episodes)

    summary = {
        "model_name": "UWM-TfD-KP-ActionOnly",
        "checkpoint": args.checkpoint,
        "mean": float(scores.mean()), "std": float(scores.std()),
        "median": float(np.median(scores)), "n_eps": args.n_eval_episodes,
        "ep_gt_05": int((scores > 0.5).sum()),
    }
    with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.output_dir}/eval_summary.json")
    print(f"Result: mean={summary['mean']:.4f} median={summary['median']:.4f} "
          f"ep>0.5={summary['ep_gt_05']}/{args.n_eval_episodes}")


if __name__ == "__main__":
    main()
