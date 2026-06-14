#!/usr/bin/env python3
"""UWM PushT evaluation script.

Will be used in Phase 7 to evaluate trained UWM PushT checkpoints.

Usage (smoke test with random init):
  cd /root/autodl-tmp/UWM_pushT
  source /root/miniconda3/bin/activate robodiff
  export SDL_VIDEODRIVER=dummy
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH=/root/autodl-tmp/UWM_pushT/unified-world-model-main:/root/autodl-tmp/UWM_pushT/diffusion_policy-main:$PYTHONPATH

  python experiments/uwm/eval_pusht.py \
    --random-init \
    --device cuda:0 \
    --num-episodes 10 \
    --max-steps 300 \
    --n-action-steps 8 \
    --output-dir outputs/uwm_pusht_eval

Usage (with trained checkpoint):
  python experiments/uwm/eval_pusht.py \
    --checkpoint outputs/uwm_pusht/checkpoints/latest.pt \
    --device cuda:0 \
    --num-episodes 50 \
    --max-steps 300 \
    --n-action-steps 8 \
    --output-dir outputs/uwm_pusht_eval
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Also allow importing from diffusion_policy-main
DP_ROOT = str(PROJECT_ROOT.parent / "diffusion_policy-main")
sys.path.insert(0, DP_ROOT)


def build_model(device, conditioning_type="adaln", dp_only=False,
                 dynamics_loss_weight=1.0, self_attn_mask=None):
    from models.uwm import UnifiedWorldModel
    from models.uwm.obs_encoder import UWMObservationEncoder

    shape_meta = {
        "obs": {
            "image": {"shape": [96, 96, 3], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
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
        beta_schedule="squaredcos_cap_v2", clip_sample=True,
        conditioning_type=conditioning_type,
        dp_only=dp_only,
        dynamics_loss_weight=dynamics_loss_weight,
        self_attn_mask=self_attn_mask,
    )
    return model.to(device)


def build_normalizer(zarr_path, device):
    root = zarr.open(zarr_path, mode="r")
    actions_all = np.array(root["data"]["action"])
    min_a = actions_all.min(axis=0)
    max_a = actions_all.max(axis=0)
    scale = torch.tensor((max_a - min_a) / 2.0, device=device).float()
    offset = torch.tensor((max_a + min_a) / 2.0, device=device).float()
    return scale, offset


def obs_env_to_uwm(env_obs, device):
    img = env_obs["image"]
    agent_pos = env_obs["agent_pos"]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if isinstance(agent_pos, np.ndarray):
        agent_pos = torch.from_numpy(agent_pos)
    img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return {
        "image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
        "agent_pos": agent_pos.float().to(device).unsqueeze(0),
    }


def run_episode(env, model, action_scale, action_offset, device,
                max_steps, n_action_steps, ep_seed):
    env.seed(ep_seed)
    obs = env.reset()
    rewards = []
    done = False
    step = 0

    while not done and step < max_steps:
        obs_uwm = obs_env_to_uwm(obs, device)
        with torch.no_grad():
            action_norm = model.sample(obs_uwm)[0]
        action_raw = action_norm * action_scale + action_offset
        action_exec = action_raw[:n_action_steps].cpu().numpy()
        obs, reward, done, info = env.step(action_exec)

        if np.isscalar(reward):
            rewards.append(reward)
        else:
            rewards.append(float(reward))
        step += 1

    return {
        "seed": ep_seed,
        "steps": step,
        "total_reward": float(sum(rewards)),
        "max_reward": float(max(rewards)) if rewards else 0.0,
        "rewards": rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="UWM PushT Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint.pt")
    parser.add_argument("--random-init", action="store_true",
                        help="Use random init model (no checkpoint)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/uwm_pusht_eval")
    parser.add_argument("--zarr-path", type=str,
                        default=None,
                        help="Override default zarr path for normalizer stats")
    parser.add_argument("--conditioning-type", type=str, default="adaln",
                        choices=["adaln", "cross_attn"],
                        help="Conditioning type (overrides checkpoint if set)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("UWM PushT Evaluation")
    print(f"  device:     {device}")
    print(f"  episodes:   {args.num_episodes}")
    print(f"  max_steps:  {args.max_steps}")
    print(f"  n_act_exec: {args.n_action_steps}")
    print(f"  output:     {args.output_dir}")
    print("=" * 60)

    # Model
    print("\n[1/4] Loading model...")
    ckpt_cond_type = args.conditioning_type
    ckpt_dp_only = False
    ckpt_dyn_weight = 1.0
    ckpt_self_attn_mask = None
    if args.checkpoint:
        ckpt_temp = torch.load(args.checkpoint, map_location=device)
        if "conditioning_type" in ckpt_temp:
            ckpt_cond_type = ckpt_temp["conditioning_type"]
        elif "config" in ckpt_temp and "conditioning_type" in ckpt_temp["config"]:
            ckpt_cond_type = ckpt_temp["config"]["conditioning_type"]
        if "dp_only" in ckpt_temp:
            ckpt_dp_only = ckpt_temp["dp_only"]
        elif "config" in ckpt_temp and "dp_only" in ckpt_temp["config"]:
            ckpt_dp_only = ckpt_temp["config"]["dp_only"]
        cfg = ckpt_temp.get("config", {})
        ckpt_dyn_weight = cfg.get("dynamics_loss_weight", 1.0)
        ckpt_self_attn_mask = cfg.get("self_attn_mask", None)
        del ckpt_temp
    model = build_model(device, conditioning_type=ckpt_cond_type,
                        dp_only=ckpt_dp_only,
                        dynamics_loss_weight=ckpt_dyn_weight,
                        self_attn_mask=ckpt_self_attn_mask)
    model.eval()
    print(f"  conditioning_type: {ckpt_cond_type}")
    print(f"  dp_only: {ckpt_dp_only}")
    print(f"  dynamics_loss_weight: {ckpt_dyn_weight}")
    print(f"  self_attn_mask: {ckpt_self_attn_mask}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)

        # Load state_dict
        model_state = ckpt["model"]
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        # Load normalizer (support both object and dict format)
        an = ckpt["action_normalizer"]
        if isinstance(an, dict):
            action_scale = torch.tensor(an["scale"], device=device).float()
            action_offset = torch.tensor(an["offset"], device=device).float()
        else:
            action_scale = torch.tensor(an.scale, device=device).float()
            action_offset = torch.tensor(an.offset, device=device).float()

        print(f"  Loaded checkpoint: {args.checkpoint}")
        print(f"  Step: {ckpt.get('step', 'unknown')}")
    elif args.random_init:
        zarr_path = args.zarr_path or str(
            PROJECT_ROOT.parent / "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
        action_scale, action_offset = build_normalizer(zarr_path, device)
        print("  Using random init model")
    else:
        print("ERROR: specify --checkpoint or --random-init")
        sys.exit(1)

    print(f"  action_scale:  {action_scale.cpu().numpy()}")
    print(f"  action_offset: {action_offset.cpu().numpy()}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params / 1e6:.1f}M")
    print("  model: OK")

    # Env
    print("\n[2/4] Creating PushT env...")
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    env = MultiStepWrapper(
        PushTImageEnv(legacy=True),
        n_obs_steps=2,
        n_action_steps=args.n_action_steps,
    )
    print(f"  Env: {type(env).__name__}")
    print("  env: OK")

    # Run episodes
    print(f"\n[3/4] Running {args.num_episodes} episodes...")
    results = []
    for ep in range(args.num_episodes):
        result = run_episode(
            env, model, action_scale, action_offset, device,
            args.max_steps, args.n_action_steps, ep,
        )
        results.append(result)
        if ep < 5 or ep % 10 == 0:
            print(f"  Ep {ep:3d}: steps={result['steps']:3d}, "
                  f"max_reward={result['max_reward']:.4f}, "
                  f"total_reward={result['total_reward']:.4f}")

    # Summary
    print(f"\n[4/4] Summary...")
    max_rewards = [r["max_reward"] for r in results]
    total_rewards = [r["total_reward"] for r in results]
    steps = [r["steps"] for r in results]

    summary = {
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "n_action_steps": args.n_action_steps,
        "mean_max_reward": float(np.mean(max_rewards)),
        "std_max_reward": float(np.std(max_rewards)),
        "mean_total_reward": float(np.mean(total_rewards)),
        "mean_steps": float(np.mean(steps)),
        "results": results,
    }

    print(f"  Mean max_reward:   {summary['mean_max_reward']:.4f}")
    print(f"  Std  max_reward:   {summary['std_max_reward']:.4f}")
    print(f"  Mean total_reward: {summary['mean_total_reward']:.4f}")
    print(f"  Mean steps:        {summary['mean_steps']:.1f}")

    # Save
    log_path = os.path.join(args.output_dir, "eval_log.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to: {log_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
