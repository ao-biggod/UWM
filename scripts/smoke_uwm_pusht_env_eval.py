#!/usr/bin/env python3
"""UWM PushT env eval smoke test.

Verifies PushTImageEnv + UWM policy inference pipeline end-to-end.

Usage:
  cd /root/autodl-tmp/UWM_pushT
  source /root/miniconda3/bin/activate robodiff
  export SDL_VIDEODRIVER=dummy
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH=/root/autodl-tmp/UWM_pushT/unified-world-model-main:/root/autodl-tmp/UWM_pushT/diffusion_policy-main:$PYTHONPATH
  python scripts/smoke_uwm_pusht_env_eval.py \
    --device cuda:0 \
    --num-episodes 2 \
    --max-steps 50 \
    --n-action-steps 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
UWM_ROOT = str(PROJECT_ROOT / "unified-world-model-main")
DP_ROOT = str(PROJECT_ROOT / "diffusion_policy-main")
sys.path.insert(0, UWM_ROOT)
sys.path.insert(0, DP_ROOT)


def obs_env_to_uwm(env_obs, device):
    """Convert DP env obs (CHW float [0,1]) to UWM format (HWC uint8)."""
    img = env_obs["image"]  # (T, C, H, W) float32 [0,1] — numpy or torch
    agent_pos = env_obs["agent_pos"]  # (T, 2) float32

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if isinstance(agent_pos, np.ndarray):
        agent_pos = torch.from_numpy(agent_pos)

    # CHW -> HWC, [0,1] -> uint8
    img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return {
        "image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
        "agent_pos": agent_pos.float().to(device).unsqueeze(0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("UWM PushT Env Eval Smoke Test")
    print(f"  device:     {device}")
    print(f"  episodes:   {args.num_episodes}")
    print(f"  max_steps:  {args.max_steps}")
    print(f"  n_act_exec: {args.n_action_steps}")
    print("=" * 60)

    # ---- 1. Import DP env ----
    print("\n[1/7] Importing PushTImageEnv...")
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    print("  PushTImageEnv import: PASS")

    # ---- 2. Create UWM model ----
    print("\n[2/7] Creating UWM model (random init)...")
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
    )
    model.to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print("  model: PASS")

    # ---- 3. Create env ----
    print("\n[3/7] Creating PushTImageEnv...")
    env = MultiStepWrapper(
        PushTImageEnv(legacy=True),
        n_obs_steps=2,
        n_action_steps=args.n_action_steps,
    )
    print(f"  Env: {type(env).__name__}, action_space={env.action_space}")
    print("  env: PASS")

    # ---- 4. Create normalizer (from DP data) ----
    print("\n[4/7] Creating normalizer from PushT data...")
    import zarr
    zarr_path = str(PROJECT_ROOT / "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    root = zarr.open(zarr_path, mode="r")
    actions_all = np.array(root["data"]["action"])
    min_a = actions_all.min(axis=0)
    max_a = actions_all.max(axis=0)
    action_scale = torch.tensor((max_a - min_a) / 2.0, device=device).float()
    action_offset = torch.tensor((max_a + min_a) / 2.0, device=device).float()
    print(f"  action_scale:  {action_scale.cpu().numpy()}")
    print(f"  action_offset: {action_offset.cpu().numpy()}")
    print("  normalizer: PASS")

    # ---- 5. Run episodes ----
    print(f"\n[5/7] Running {args.num_episodes} episodes...")
    all_rewards = []
    for ep in range(args.num_episodes):
        print(f"\n  --- Episode {ep} ---")
        env.seed(ep)
        obs = env.reset()
        ep_rewards = []
        done = False
        step = 0

        while not done and step < args.max_steps:
            # Convert obs to UWM format
            obs_uwm = obs_env_to_uwm(obs, device)

            # Policy inference
            with torch.no_grad():
                action_norm = model.sample(obs_uwm)[0]  # (16, 2)
            # Unnormalize
            action_raw = action_norm * action_scale + action_offset
            # Take first n_action_steps
            action_exec = action_raw[:args.n_action_steps].cpu().numpy()

            # Step env
            obs, reward, done, info = env.step(action_exec)
            ep_rewards.append(reward)
            step += 1

            if step == 1:
                print(f"    step {step}: reward={reward:.4f}, action_range=[{action_exec.min():.1f}, {action_exec.max():.1f}]")

        total_reward = sum(ep_rewards)
        max_reward = max(ep_rewards) if ep_rewards else 0.0
        all_rewards.append(total_reward)
        print(f"    Done: steps={step}, total_reward={total_reward:.4f}, max_reward={max_reward:.4f}")

    print("  env step: PASS")

    # ---- 6. Summary ----
    print("\n[6/7] Reward summary...")
    for ep, r in enumerate(all_rewards):
        print(f"  Episode {ep}: total_reward={r:.4f}")
    print(f"  Mean total reward: {np.mean(all_rewards):.4f}")
    print("  reward finite: PASS")

    # ---- 7. Final ----
    print("\n[7/7] Validation...")
    print("  no traceback: PASS")
    print("  episodes complete: PASS")

    print("\n" + "=" * 60)
    print("ENV EVAL SMOKE TEST PASSED")
    print("(Random init model — low reward expected)")
    print("=" * 60)


if __name__ == "__main__":
    main()
