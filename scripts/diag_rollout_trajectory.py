#!/usr/bin/env python3
"""Diagnostic rollout capturing detailed state for trajectory analysis.

Captures per-step: reward, agent-block distance, coverage error,
action magnitude, agent pos, block pose, goal pose.

Usage:
  python scripts/diag_rollout_trajectory.py \
    --checkpoint <path> --seed 100000 --label R1 \
    --output outputs/diag_traj/
"""

import argparse, json, os, sys, random
from pathlib import Path
import numpy as np
import torch

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


def rollout_with_state(checkpoint_path, seed, device):
    """Run rollout and capture detailed per-step state."""
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

    inner_env = PushTImageEnv(legacy=True)
    env = MultiStepWrapper(inner_env, n_obs_steps=2, n_action_steps=8)
    env.seed(seed)
    obs = env.reset()

    from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely as _pts

    # Goal pose (constant)
    goal_pose = inner_env.goal_pose.copy()  # [x, y, angle]

    trajectory = {
        "seed": seed,
        "goal_pose": goal_pose.tolist(),
        "steps": [],
        "actions": [],
    }

    step = 0
    while step < 300:
        # Capture state BEFORE action
        agent_pos = np.array(inner_env.agent.position)
        block_pos = np.array(inner_env.block.position)
        block_angle = inner_env.block.angle % (2 * np.pi)

        # Compute derived metrics
        agent_to_block = float(np.linalg.norm(agent_pos - block_pos))

        # Coverage
        goal_body = inner_env._get_goal_pose_body(inner_env.goal_pose)
        goal_geom = _pts(goal_body, inner_env.block.shapes)
        block_geom = _pts(inner_env.block, inner_env.block.shapes)
        coverage = goal_geom.intersection(block_geom).area / goal_geom.area
        coverage_error = 1.0 - coverage

        # Preprocess obs
        img = obs["image"]
        ap_raw = obs["agent_pos"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(ap_raw, np.ndarray):
            ap_raw = torch.from_numpy(ap_raw)

        img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        ap = ap_raw.float().to(device)
        ap_norm = (ap - torch.tensor(ap_offset, device=device).float()) / torch.tensor(ap_scale, device=device).float()

        obs_model = {"image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
                     "agent_pos": ap_norm.unsqueeze(0)}

        with torch.no_grad():
            action_norm = model.sample(obs_model)[0]

        action_raw = (action_norm
                      * torch.tensor(action_scale, device=device).float()
                      + torch.tensor(action_offset, device=device).float())
        action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

        # Step env
        obs, reward, done, info = env.step(action_raw_np[:8])
        reward_val = float(reward) if not isinstance(reward, np.ndarray) else float(reward.item())

        # Record this step
        trajectory["steps"].append({
            "step": step,
            "agent_pos": agent_pos.tolist(),
            "block_pos": block_pos.tolist(),
            "block_angle": float(block_angle),
            "agent_to_block": agent_to_block,
            "coverage": float(coverage),
            "coverage_error": coverage_error,
            "reward": reward_val,
        })

        # Record action chunk (first 8 actions that were executed)
        chunk = action_raw_np[:8].tolist()
        action_mag = float(np.linalg.norm(action_raw_np[:8], axis=1).mean())
        trajectory["actions"].append({
            "step": step,
            "chunk": chunk,
            "mean_magnitude": action_mag,
        })

        if np.all(done):
            break
        step += 1

    # Final state
    agent_pos = np.array(inner_env.agent.position)
    block_pos = np.array(inner_env.block.position)
    block_angle = inner_env.block.angle % (2 * np.pi)

    goal_body = inner_env._get_goal_pose_body(inner_env.goal_pose)
    goal_geom = _pts(goal_body, inner_env.block.shapes)
    block_geom = _pts(inner_env.block, inner_env.block.shapes)
    coverage = goal_geom.intersection(block_geom).area / goal_geom.area

    trajectory["steps"].append({
        "step": step + 1,
        "agent_pos": agent_pos.tolist(),
        "block_pos": block_pos.tolist(),
        "block_angle": float(block_angle),
        "agent_to_block": float(np.linalg.norm(agent_pos - block_pos)),
        "coverage": float(coverage),
        "coverage_error": 1.0 - coverage,
        "reward": trajectory["steps"][-1]["reward"] if trajectory["steps"] else 0.0,
    })

    env.close()
    trajectory["total_steps"] = step + 1
    trajectory["max_reward"] = max(s["reward"] for s in trajectory["steps"])

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--label", type=str, default="model")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="outputs/diag_traj")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Rollout: {args.label} seed={args.seed}")
    traj = rollout_with_state(args.checkpoint, args.seed, device)

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"trajectory_{args.label}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(traj, f, indent=2)
    print(f"  steps={traj['total_steps']} max_reward={traj['max_reward']:.4f}")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
