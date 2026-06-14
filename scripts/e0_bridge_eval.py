#!/usr/bin/env python3
"""E0 Bridge: DP official lowdim eval in official AND fixed-buffer eval stacks."""
import sys, os, torch, json, time, dill
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))


def load_policy_from_workspace_ckpt(checkpoint_path):
    """Load DP official lowdim policy from Hydra workspace checkpoint."""
    from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
    from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ema_sd = ckpt["state_dicts"]["ema_model"]
    # Extract epoch if available
    epoch = None
    if "pickles" in ckpt and "epoch" in ckpt["pickles"]:
        epoch_val = ckpt["pickles"]["epoch"]
        epoch = dill.loads(epoch_val) if isinstance(epoch_val, bytes) else epoch_val

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001, beta_end=0.02, variance_type="fixed_small",
        clip_sample=True, prediction_type="epsilon")

    transformer = TransformerForDiffusion(
        input_dim=2, output_dim=2, horizon=16, n_obs_steps=2, cond_dim=20,
        n_layer=8, n_head=4, n_emb=256, p_drop_emb=0.0, p_drop_attn=0.01,
        causal_attn=True, time_as_cond=True, obs_as_cond=True)

    # Build on CPU, load weights, then move to GPU
    policy = DiffusionTransformerLowdimPolicy(
        model=transformer, noise_scheduler=noise_scheduler,
        horizon=16, obs_dim=20, action_dim=2,
        n_action_steps=8, n_obs_steps=2,
        num_inference_steps=100, obs_as_cond=True,
        pred_action_steps_only=False)

    policy.load_state_dict(ema_sd)
    policy.eval()
    return policy, epoch


def get_full_state_from_env(env):
    agent_p = np.array(env.agent.position)
    block_p = np.array(env.block.position)
    block_ang = env.block.angle
    return np.concatenate([agent_p, block_p, [block_ang]])


# ===== E0-1: Official PushTKeypointsRunner eval =====
def run_official_eval(policy, n_episodes=50, seed_start=100000):
    from diffusion_policy.env_runner.pusht_keypoints_runner import PushTKeypointsRunner

    policy = policy.to("cuda:0")  # Runner needs GPU policy

    runner = PushTKeypointsRunner(
        keypoint_visible_rate=1.0,
        n_train=0, n_train_vis=0,
        n_test=n_episodes, n_test_vis=0,
        legacy_test=True,
        train_start_seed=0,
        test_start_seed=seed_start,
        max_steps=300,
        n_obs_steps=2, n_action_steps=8,
        n_latency_steps=0, fps=10,
        agent_keypoints=False, past_action=False,
        n_envs=None,
        output_dir="/tmp/e0_official_eval",
    )

    results = runner.run(policy)
    test_mean = results.get("test/mean_score", results.get("test_mean_score", "N/A"))
    return test_mean, results


# ===== E0-2: Our fixed-buffer eval =====
def run_fixed_buffer_eval(policy, n_episodes=50, seed_start=100000):
    """DP official lowdim policy in OUR fixed-buffer eval with keypoint obs."""
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

    device = "cuda:0"
    policy = policy.to(device)

    results = []
    all_deltas = []

    for ep in range(n_episodes):
        seed = seed_start + ep
        env = PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0)
        env.seed(seed)
        raw_obs = env.reset()  # shape (40,) = [obs_data(20), mask(20)]

        obs_buffer = deque(maxlen=2)
        rewards = []
        done = False
        physical_step = 0
        ep_deltas = []

        # Prime: collect 2 frames of obs_data (first 20 dims)
        obs_buffer.append(raw_obs[:20])  # frame 0 data
        raw_obs, _, _, _ = env.step(np.zeros(2))
        physical_step += 1
        obs_buffer.append(raw_obs[:20])  # frame 1 data

        while not done and physical_step < 300:
            # Build obs tensor [1, T, 20] → policy takes first 2 frames
            full_obs = np.zeros((16, 20), dtype=np.float32)
            full_obs[0] = obs_buffer[0]  # frame 0
            full_obs[1] = obs_buffer[1]  # frame 1
            for t in range(2, 16):
                full_obs[t] = obs_buffer[1]  # pad with last frame

            obs_t = torch.from_numpy(full_obs).float().to(device).unsqueeze(0)
            # Match official runner: pass obs_dict with 'obs' key, policy normalizes internally
            obs_dict = {'obs': obs_t}
            with torch.no_grad():
                action_pred = policy.predict_action(obs_dict)['action'][0]  # [8, 2]

            action_raw = policy.normalizer['action'].unnormalize(action_pred)
            action_raw = action_raw.cpu().numpy()
            action_raw = np.clip(action_raw, 0.0, 512.0)
            exec_actions = action_raw[:8]

            delta = exec_actions[1:] - exec_actions[:-1]
            delta_l2 = np.linalg.norm(delta, axis=-1)
            ep_deltas.append(float(delta_l2.mean()))

            for k in range(8):
                act = exec_actions[k]
                raw_obs, reward, _done, _info = env.step(act)
                physical_step += 1
                obs_buffer.append(raw_obs[:20])  # extract data part only
                rewards.append(float(reward))
                if _done or physical_step >= 300:
                    done = True
                    break

        max_reward = float(max(rewards)) if rewards else 0.0
        results.append(max_reward)
        all_deltas.append(float(np.mean(ep_deltas)))

        if ep < 5 or ep % 10 == 0:
            print(f"  Ep {ep:3d}: max_reward={max_reward:.4f}", flush=True)

    scores = np.array(results)
    print(f"\n  Fixed-buffer: mean={scores.mean():.4f} std={scores.std():.4f} "
          f"median={np.median(scores):.4f} ep>0.5={(scores>0.5).sum()}/{n_episodes} "
          f"ep>0.8={(scores>0.8).sum()}")

    return {
        "mean_max_reward": float(scores.mean()),
        "std_max_reward": float(scores.std()),
        "median": float(np.median(scores)),
        "ep_gt_05": int((scores > 0.5).sum()),
        "ep_gt_08": int((scores > 0.8).sum()),
        "action_delta_mean": float(np.mean(all_deltas)),
    }


# ===== E0-3: Action slicing variants =====
def run_slicing_variants(policy, n_episodes=50, seed_start=100000):
    """Test action slicing sensitivity for DP official policy."""
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

    device = "cuda:0"
    policy = policy.to(device)

    slicing_results = {}
    for label, slice_start, slice_end in [
        ("official [1:9]", 1, 9),
        ("ours-style [0:8]", 0, 8),
        ("shifted+2 [2:10]", 2, 10),
    ]:
        results = []
        for ep in range(n_episodes):
            seed = seed_start + ep
            env = PushTKeypointsEnv(legacy=True)
            env.seed(seed)
            kp_obs = env.reset()

            obs_buffer = deque(maxlen=2)
            rewards = []
            done = False
            physical_step = 0

            obs_buffer.append(kp_obs)
            kp_obs, _, _, _ = env.step(np.zeros(2))
            physical_step += 1
            obs_buffer.append(kp_obs)

            while not done and physical_step < 300:
                s0_kp = obs_buffer[0]['keypoint'].reshape(-1)
                s0_ap = obs_buffer[0]['agent_pos']
                s1_kp = obs_buffer[1]['keypoint'].reshape(-1)
                s1_ap = obs_buffer[1]['agent_pos']

                full_obs = np.zeros((16, 20), dtype=np.float32)
                full_obs[0, :18] = s0_kp; full_obs[0, 18:] = s0_ap
                full_obs[1, :18] = s1_kp; full_obs[1, 18:] = s1_ap
                for t in range(2, 16):
                    full_obs[t, :18] = s1_kp; full_obs[t, 18:] = s1_ap

                obs_t = torch.from_numpy(full_obs).float().to(device).unsqueeze(0)
                n_obs = policy.normalizer.normalize({'obs': obs_t, 'action': torch.zeros(1, 16, 2, device=device)})

                with torch.no_grad():
                    action_pred = policy.predict_action(n_obs)['action'][0]

                action_raw = policy.normalizer['action'].unnormalize(action_pred)
                action_raw = action_raw.cpu().numpy()
                action_raw = np.clip(action_raw, 0.0, 512.0)

                exec_actions = action_raw[slice_start:slice_end]

                for k in range(8):
                    act = exec_actions[k]
                    kp_obs, reward, _done, _info = env.step(act)
                    physical_step += 1
                    obs_buffer.append(kp_obs)
                    rewards.append(float(reward))
                    if _done or physical_step >= 300:
                        done = True
                        break

            results.append(float(max(rewards)) if rewards else 0.0)

        scores = np.array(results)
        slicing_results[label] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "median": float(np.median(scores)),
            "ep_gt_05": int((scores > 0.5).sum()),
        }
        print(f"\n  {label:20s}: mean={scores.mean():.4f} std={scores.std():.4f} "
              f"median={np.median(scores):.4f} ep>0.5={(scores>0.5).sum()}")

    return slicing_results


# ===== Main =====
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/e0_lowdim_official_full/checkpoints/epoch=0190-test_mean_score=1.000.ckpt")
    parser.add_argument("--mode", type=str, default="all", choices=["e01", "e02", "e03", "all"])
    parser.add_argument("--n-eps", type=int, default=50)
    args = parser.parse_args()

    policy, epoch = load_policy_from_workspace_ckpt(args.checkpoint)
    print(f"E0 Bridge Eval | Checkpoint: {args.checkpoint} | Epoch: {epoch}")
    print(f"Episodes per config: {args.n_eps}")
    print(f"{'='*60}")

    if args.mode in ("e01", "all"):
        print(f"\n--- E0-1: DP official lowdim in OFFICIAL eval stack ---")
        t0 = time.time()
        score, full = run_official_eval(policy, args.n_eps)
        print(f"Elapsed: {time.time()-t0:.1f}s")
        print(f"  test_mean_score: {score}")

    if args.mode in ("e02", "all"):
        print(f"\n--- E0-2: DP official lowdim in OUR fixed-buffer eval stack ---")
        t0 = time.time()
        our = run_fixed_buffer_eval(policy, args.n_eps)
        print(f"Elapsed: {time.time()-t0:.1f}s")

    if args.mode in ("e03", "all"):
        print(f"\n--- E0-3: Action slicing variants ---")
        t0 = time.time()
        slicing = run_slicing_variants(policy, args.n_eps)
        print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
