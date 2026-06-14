#!/usr/bin/env python3
"""Step 1: Expert action playback — reset env to zarr initial state, then replay expert actions.

Each episode:
1. Read init_state = expert_states[0] from zarr
2. Create PushTImageEnv with reset_to_state=init_state
3. env.reset() → env is now at the expert's initial state
4. Execute expert_actions[t] step by step
5. Track max_reward
"""
import argparse, json, os, sys
import numpy as np
import zarr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str,
                        default="diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="outputs/diag_expert_playback")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    z = zarr.open(args.zarr_path, "r")
    all_action = z["data/action"][:]
    all_state = z["data/state"][:]      # (N, 5) = [agent_xy, block_xy, block_angle]
    ep_ends = z["meta/episode_ends"][:]

    print(f"Action stats: min={all_action.min(axis=0)}, max={all_action.max(axis=0)}, "
          f"mean={all_action.mean(axis=0)}, std={all_action.std(axis=0)}")
    print(f"State stats:  min={all_state.min(axis=0)}, max={all_state.max(axis=0)}, "
          f"mean={all_state.mean(axis=0)}, std={all_state.std(axis=0)}")
    print(f"State format: [agent_x, agent_y, block_x, block_y, block_angle]")
    print(f"Action format: absolute target position [x, y] in [0, 512]")

    results = []
    n_episodes = min(args.n_episodes, len(ep_ends))

    for ep_idx in range(n_episodes):
        ep_start = ep_ends[ep_idx - 1] if ep_idx > 0 else 0
        ep_end = ep_ends[ep_idx]
        expert_actions = all_action[ep_start:ep_end]
        expert_states = all_state[ep_start:ep_end]
        init_state = expert_states[0].copy()

        # Reset env to expert initial state (set reset_to_state before reset())
        env = PushTImageEnv(legacy=True)
        env.reset_to_state = init_state
        obs = env.reset()

        # Verify: env state after reset should match init_state
        env_agent = np.array(env.agent.position)
        env_block = np.array(env.block.position)
        env_angle = env.block.angle
        env_state = np.concatenate([env_agent, env_block, [env_angle]])

        t = 0
        rewards = []
        done = False
        while not done and t < len(expert_actions) and t < args.max_steps:
            action = expert_actions[t]
            obs, reward, done, info = env.step(action)
            r = float(reward)
            rewards.append(r)
            t += 1

        max_reward = float(max(rewards)) if rewards else 0.0

        # Debug print for first 3 episodes
        if ep_idx < 3:
            print(f"\n  === Ep {ep_idx} ===")
            print(f"    zarr init_state:    {init_state}")
            print(f"    env state after reset: {env_state}")
            print(f"    state diff:         {np.abs(init_state - env_state).max():.6f}")
            print(f"    first 5 expert actions: {expert_actions[:5].tolist()}")
            print(f"    first 5 rewards:       {rewards[:5]}")
            print(f"    max_reward:            {max_reward:.4f}  steps: {t}")
        elif ep_idx < 5 or ep_idx % 10 == 0:
            print(f"  Ep {ep_idx:3d}: max_reward={max_reward:.4f}  steps={t}")

        results.append({
            "ep_idx": int(ep_idx),
            "ep_len": int(ep_end - ep_start),
            "steps": int(t),
            "max_reward": float(max_reward),
            "state_diff": float(np.abs(init_state - env_state).max()),
        })

    mean_max = float(np.mean([r["max_reward"] for r in results]))
    std_max = float(np.std([r["max_reward"] for r in results]))
    high_score = sum(1 for r in results if r["max_reward"] > 0.5)

    print(f"\n=== Expert Action Playback ({len(results)} episodes) ===")
    print(f"  Mean max_reward: {mean_max:.4f}")
    print(f"  Std  max_reward: {std_max:.4f}")
    print(f"  Episodes > 0.5: {high_score}/{len(results)}")

    print(f"\n  Interpretation: ", end="")
    if mean_max > 0.6:
        print("HIGH — expert actions work. Action/env semantics verified.")
        print("  NOTE: goal pose is NOT set (not in zarr), so some episodes")
        print("  will have mismatched goals and low scores.")
    elif mean_max > 0.3:
        print("MODERATE — some episodes score >0.5, but many fail due to goal mismatch.")
        print("  -> Goal pose is random per seed and not recorded in zarr.")
    else:
        print("LOW — expert playback largely fails.")
        print("  -> Check state dimension order, action format, or physics step determinism.")

    summary = {
        "n_episodes": int(len(results)),
        "mean_max_reward": float(mean_max),
        "std_max_reward": float(std_max),
        "ep_gt_05": int(high_score),
        "results": [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in r.items()} for r in results],
    }
    with open(os.path.join(args.output_dir, "playback_log.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to: {args.output_dir}/playback_log.json")


if __name__ == "__main__":
    main()
