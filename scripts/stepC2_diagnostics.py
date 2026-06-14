#!/usr/bin/env python3
"""Step C2: State spacing audit + replan frequency ablation + fixed-buffer eval.

Diagnostics to determine if the eval state_buffer spacing is the root cause
behind low rollout score (0.18) despite good offline prediction (RMSE/Std=0.14).

Hypothesis: training states are 1 physical step apart, but eval state_buffer
collects states 8 steps apart (because each env.step executes 8 actions).
This distribution shift causes poor closed-loop performance.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_model_and_normalizers(checkpoint_path, device):
    """Load B2 model and its normalizers."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stepB_retrain_lowdim",
        str(Path(__file__).resolve().parent / "stepB_retrain_lowdim.py"))
    stepB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stepB)
    LowdimStatePolicyV2 = stepB.LowdimStatePolicyV2

    ckpt = torch.load(checkpoint_path, map_location=device)
    clip_sample_flag = ckpt.get("clip_sample", True)
    model = LowdimStatePolicyV2(clip_sample=clip_sample_flag, num_inference_steps=10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    norm_type = ckpt.get("norm_type", "meanstd")
    an = ckpt["action_normalizer"]
    sn = ckpt["state_normalizer"]

    if norm_type == "minmax":
        a_offset = torch.tensor(an.get("offset", an.get("mean")), device=device).float()
        a_scale = torch.tensor(an.get("scale", an.get("std")), device=device).float()
        s_offset = torch.tensor(sn.get("offset", sn.get("mean")), device=device).float()
        s_scale = torch.tensor(sn.get("scale", sn.get("std")), device=device).float()

        def norm_state(x):
            return (x - s_offset) / s_scale
        def unnorm_action(x):
            return x * a_scale + a_offset
    else:
        a_mean = torch.tensor(an.get("mean", an.get("offset")), device=device).float()
        a_std = torch.tensor(an.get("std", an.get("scale")), device=device).float()
        s_mean = torch.tensor(sn.get("mean", sn.get("offset")), device=device).float()
        s_std = torch.tensor(sn.get("std", sn.get("scale")), device=device).float()

        def norm_state(x):
            return (x - s_mean) / s_std
        def unnorm_action(x):
            return x * a_std + a_mean

    return model, norm_state, unnorm_action, norm_type


def get_env_full_state(env):
    """Extract full 5D state from PushTImageEnv internals."""
    inner = env.env
    agent_p = np.array(inner.agent.position)
    block_p = np.array(inner.block.position)
    block_ang = inner.block.angle
    return np.concatenate([agent_p, block_p, [block_ang]])


# =========================================================================
# C2-1: State spacing audit with full logging
# =========================================================================
def c21_state_spacing_audit(checkpoint_path, device, n_episodes=3, n_policy_calls=10):
    """Log state_buffer spacing to confirm whether states are [t-1,t] or [t-8,t]."""
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    model, norm_state, unnorm_action, norm_type = load_model_and_normalizers(checkpoint_path, device)

    print("=" * 70)
    print("C2-1: STATE SPACING AUDIT")
    print("=" * 70)

    for ep in range(n_episodes):
        env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
        env.seed(ep)
        obs = env.reset()
        state_buffer = []
        done = False
        physical_step_counter = 0  # incremented for each raw env step
        policy_call_idx = 0

        print(f"\n--- Episode {ep} ---")
        print(f"  {'Policy':>6s} {'PhysStep':>9s} {'Buf[0]@step':>12s} {'Buf[1]@step':>12s} "
              f"{'Spacing':>7s} {'agent_xy[0]':>18s} {'agent_xy[1]':>18s} {'diff':>10s}")

        while not done and physical_step_counter < 300 and policy_call_idx < n_policy_calls:
            full_state = get_env_full_state(env)
            state_buffer.append((physical_step_counter, full_state))
            if len(state_buffer) > 2:
                state_buffer = state_buffer[-2:]

            if len(state_buffer) < 2:
                for i in range(8):
                    obs, reward, done, info = env.step(np.zeros((8, 2)))
                    physical_step_counter += 8
                policy_call_idx += 1
                continue

            ts0, s0 = state_buffer[0]
            ts1, s1 = state_buffer[1]
            spacing = ts1 - ts0

            state_np = np.stack([s0, s1])
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_normed = norm_state(state_t)

            with torch.no_grad():
                action_norm = model.sample(state_normed)[0]
            action_raw_t = unnorm_action(action_norm)
            action_raw = np.clip(action_raw_t.cpu().numpy(), 0.0, 512.0)

            # Execute 8 actions (standard)
            for a in action_raw[:8]:
                physical_step_counter += 1
            obs, reward, done, info = env.step(action_raw[:8])

            print(f"  {policy_call_idx:6d} {physical_step_counter:9d} {ts0:12d} {ts1:12d} "
                  f"{spacing:7d} {str(s0[:2]):>18s} {str(s1[:2]):>18s} {np.linalg.norm(s1[:2]-s0[:2]):10.1f}")

            policy_call_idx += 1

        print(f"  Episode {ep} done after {policy_call_idx} policy calls, {physical_step_counter} physical steps")


# =========================================================================
# C2-2: Replan frequency ablation (change n_action_exec, no retrain)
# =========================================================================
def run_eval_with_n_exec(model, norm_state, unnorm_action, device,
                         n_action_exec, n_episodes, verbose=True):
    """Eval with configurable n_action_exec. Model always predicts 16 actions."""
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=n_action_exec)
    results = []

    for ep in range(n_episodes):
        env.seed(ep)
        obs = env.reset()
        rewards = []
        state_buffer = []
        done = False
        step = 0

        while not done and step < 300:
            full_state = get_env_full_state(env)
            state_buffer.append(full_state)
            if len(state_buffer) > 2:
                state_buffer = state_buffer[-2:]

            if len(state_buffer) < 2:
                obs, reward, done, info = env.step(np.zeros((n_action_exec, 2)))
                rewards.append(float(reward))
                step += 1
                continue

            state_np = np.stack(state_buffer)
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_normed = norm_state(state_t)

            with torch.no_grad():
                action_norm = model.sample(state_normed)[0]  # always 16 actions
            action_raw = unnorm_action(action_norm)
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

            action_exec = action_raw[:n_action_exec]
            obs, reward, done, info = env.step(action_exec)
            rewards.append(float(reward))
            step += 1

        results.append({
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "steps": step,
        })

        if verbose and (ep < 5 or ep % 10 == 0):
            print(f"  Ep {ep:3d}: max_reward={results[-1]['max_reward']:.4f}", flush=True)

    mean_max = float(np.mean([r["max_reward"] for r in results]))
    std_max = float(np.std([r["max_reward"] for r in results]))
    high = sum(1 for r in results if r["max_reward"] > 0.5)

    if verbose:
        print(f"\n  >>> n_exec={n_action_exec}: mean={mean_max:.4f} std={std_max:.4f} "
              f"ep>0.5={high}/{len(results)}")

    return {"mean_max_reward": mean_max, "std_max_reward": std_max,
            "ep_gt_05": high, "n_episodes": n_episodes, "n_action_exec": n_action_exec}


# =========================================================================
# C2-3: Fixed-buffer eval (state_buffer uses 1-step spacing)
# =========================================================================
def run_eval_fixed_buffer(model, norm_state, unnorm_action, device,
                          n_action_exec, n_episodes, verbose=True):
    """Eval with FIXED state buffer: states are 1 physical step apart.

    Instead of collecting states at policy-call boundaries, we execute
    actions one by one and maintain a proper sliding observation buffer
    with 1-step spacing, matching training distribution.
    """
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=n_action_exec)
    results = []

    for ep in range(n_episodes):
        env.seed(ep)
        _ = env.reset()
        rewards = []
        # We maintain a state deque with states 1 physical step apart
        state_obs_buffer = []
        done = False
        physical_step = 0

        # Prime the buffer with 2 initial states
        while len(state_obs_buffer) < 2 and not done and physical_step < 300:
            full_state = get_env_full_state(env)
            state_obs_buffer.append(full_state)
            # Execute 1 zero action to move env forward
            obs, reward, done, info = env.step(np.zeros((n_action_exec, 2)))
            rewards.append(float(reward))
            physical_step += 1

        # Now state_obs_buffer has 2 states. But they're still n_action_exec apart!
        # To fix this properly, we need to execute actions ONE BY ONE
        # and collect state after EACH physical step.
        # Let's restart with a proper manual loop.

        # Actually, use raw env without MultiStepWrapper for fine-grained control
        raw_env = PushTImageEnv(legacy=True)
        raw_env.seed(ep)
        raw_env.reset()
        obs_deque = []
        # Prime: 2 dummy steps with zero action
        for _ in range(2):
            full_state = get_env_full_state_raw(raw_env)
            obs_deque.append(full_state)
            raw_env.step(np.zeros(2))  # single action

        done = False

        while not done and physical_step < 300:
            if len(obs_deque) > 2:
                obs_deque = obs_deque[-2:]

            state_np = np.stack(obs_deque)  # [2, 5], 1-step apart
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_normed = norm_state(state_t)

            with torch.no_grad():
                action_norm = model.sample(state_normed)[0]
            action_raw = unnorm_action(action_norm)
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

            # Execute n_action_exec actions, collecting state after EACH
            for a_idx in range(n_action_exec):
                act = action_raw[a_idx]
                raw_env.step(act)
                physical_step += 1
                full_state = get_env_full_state_raw(raw_env)
                obs_deque.append(full_state)
                # We don't have reward tracking at single-step level with raw env
                # Use block distance as reward proxy
                from diffusion_policy.env.pusht.pusht_env import PushTEnv
                if hasattr(raw_env, 'success_key'):
                    pass

            # Check if done
            if physical_step >= 300:
                done = True

        # Compute reward based on final block position
        block_pos = np.array(raw_env.block.position)
        goal_pos = np.array(raw_env.goal.position)
        coverage = raw_env._get_coverage()
        max_reward = float(coverage)

        results.append({
            "max_reward": max_reward,
            "steps": physical_step,
        })

        if verbose and (ep < 5 or ep % 10 == 0):
            print(f"  Ep {ep:3d}: max_reward={results[-1]['max_reward']:.4f}", flush=True)

    mean_max = float(np.mean([r["max_reward"] for r in results]))
    std_max = float(np.std([r["max_reward"] for r in results]))
    high = sum(1 for r in results if r["max_reward"] > 0.5)

    if verbose:
        print(f"\n  >>> fixed-buffer n_exec={n_action_exec}: mean={mean_max:.4f} std={std_max:.4f} "
              f"ep>0.5={high}/{len(results)}")

    return {"mean_max_reward": mean_max, "std_max_reward": std_max,
            "ep_gt_05": high, "n_episodes": n_episodes, "n_action_exec": n_action_exec}


def get_env_full_state_raw(raw_env):
    """Extract full 5D state from raw PushTImageEnv."""
    import gym
    agent_p = np.array(raw_env.agent.position)
    block_p = np.array(raw_env.block.position)
    block_ang = raw_env.block.angle
    return np.concatenate([agent_p, block_p, [block_ang]])


def run_eval_fixed_buffer_v2(model, norm_state, unnorm_action, device,
                             n_action_exec, n_episodes, verbose=True):
    """Fixed-buffer eval: use raw env (no MultiStepWrapper), execute actions
    one at a time, collect state after each physical step, predict every
    n_action_exec steps.

    This ensures state_buffer always has consecutive physical steps [t-1, t],
    matching training distribution exactly.
    """
    import gym
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    results = []

    for ep in range(n_episodes):
        raw_env = PushTImageEnv(legacy=True)
        raw_env.seed(ep)
        raw_env.reset()
        obs_deque = deque(maxlen=2)

        # Prime: execute 2 zero-action steps to fill buffer
        for _ in range(2):
            full_state = get_env_full_state_raw(raw_env)
            obs_deque.append(full_state)
            raw_env.step(np.zeros(2))

        coverage = 0.0
        done = False
        physical_step = 0

        while not done and physical_step < 300:
            # Predict using current buffer (last 2 consecutive states)
            state_np = np.stack(list(obs_deque))  # [2, 5]
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_normed = norm_state(state_t)

            with torch.no_grad():
                action_norm = model.sample(state_normed)[0]
            action_raw = unnorm_action(action_norm)
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

            # Execute n_action_exec actions, ONE BY ONE, collecting state after each
            for a_idx in range(n_action_exec):
                act = action_raw[a_idx]
                raw_env.step(act)
                physical_step += 1
                full_state = get_env_full_state_raw(raw_env)
                obs_deque.append(full_state)  # maintain consecutive states

                # Check coverage
                coverage = max(coverage, raw_env._get_coverage())

            if coverage >= 1.0:
                done = True

        results.append({
            "max_reward": float(coverage),
            "steps": physical_step,
        })

        if verbose and (ep < 5 or ep % 10 == 0):
            print(f"  Ep {ep:3d}: max_reward={results[-1]['max_reward']:.4f} steps={physical_step}",
                  flush=True)

    mean_max = float(np.mean([r["max_reward"] for r in results]))
    std_max = float(np.std([r["max_reward"] for r in results]))
    high = sum(1 for r in results if r["max_reward"] > 0.5)

    if verbose:
        print(f"\n  >>> fixed-buffer-v2 n_exec={n_action_exec}: mean={mean_max:.4f} "
              f"std={std_max:.4f} ep>0.5={high}/{len(results)}")

    return {"mean_max_reward": mean_max, "std_max_reward": std_max,
            "ep_gt_05": high, "n_episodes": n_episodes, "n_action_exec": n_action_exec}


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Step C2: State spacing audit + ablation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/stepB_retrain/B2_minmax_clip/latest.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/stepC2")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["c21", "c22", "c23", "all"])
    parser.add_argument("--n-eps", type=int, default=10,
                        help="Episodes per config (50 for final, 10 for quick)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # ===== C2-1: State spacing audit =====
    if args.mode in ("c21", "all"):
        c21_state_spacing_audit(args.checkpoint, device, n_episodes=3, n_policy_calls=10)

    # ===== C2-2: Replan frequency ablation =====
    if args.mode in ("c22", "all"):
        print("\n" + "=" * 70)
        print("C2-2: REPLAN FREQUENCY ABLATION")
        print("=" * 70)

        model, norm_state, unnorm_action, norm_type = load_model_and_normalizers(
            args.checkpoint, device)
        print(f"  Model: {args.checkpoint}")
        print(f"  Normalizer: {norm_type}")
        print(f"  Episodes per config: {args.n_eps}")

        replan_results = {}
        for n_exec in [8, 4, 2, 1]:
            print(f"\n  --- n_action_exec = {n_exec} ---")
            t0 = time.time()
            res = run_eval_with_n_exec(model, norm_state, unnorm_action, device,
                                      n_action_exec=n_exec, n_episodes=args.n_eps)
            elapsed = time.time() - t0
            print(f"  Elapsed: {elapsed:.1f}s")
            replan_results[f"replan_{n_exec}"] = res

        results["c22_replan"] = replan_results

        # Save
        with open(os.path.join(args.output_dir, "c22_replan_results.json"), "w") as f:
            json.dump({"checkpoint": args.checkpoint, "n_eps": args.n_eps,
                       "results": replan_results}, f, indent=2)

        print(f"\n  Results saved to {args.output_dir}/c22_replan_results.json")

    # ===== C2-3: Fixed-buffer eval =====
    if args.mode in ("c23", "all"):
        print("\n" + "=" * 70)
        print("C2-3: FIXED-BUFFER EVAL (1-step state spacing)")
        print("=" * 70)

        model, norm_state, unnorm_action, norm_type = load_model_and_normalizers(
            args.checkpoint, device)
        print(f"  Model: {args.checkpoint}")
        print(f"  Normalizer: {norm_type}")
        print(f"  Episodes per config: {args.n_eps}")

        fixed_results = {}
        for n_exec in [8, 1]:
            print(f"\n  --- fixed-buffer n_action_exec = {n_exec} ---")
            t0 = time.time()
            res = run_eval_fixed_buffer_v2(model, norm_state, unnorm_action, device,
                                           n_action_exec=n_exec, n_episodes=args.n_eps)
            elapsed = time.time() - t0
            print(f"  Elapsed: {elapsed:.1f}s")
            fixed_results[f"fixed_buffer_exec_{n_exec}"] = res

        results["c23_fixed_buffer"] = fixed_results

        with open(os.path.join(args.output_dir, "c23_fixed_buffer_results.json"), "w") as f:
            json.dump({"checkpoint": args.checkpoint, "n_eps": args.n_eps,
                       "results": fixed_results}, f, indent=2)

        print(f"\n  Results saved to {args.output_dir}/c23_fixed_buffer_results.json")

    # ===== Summary table =====
    if args.mode == "all":
        print("\n\n" + "=" * 70)
        print("C2 SUMMARY TABLE")
        print("=" * 70)

        print(f"\n  {'Run':<28s} {'State Spacing':<15s} {'n_exec':<8s} {'Score':>8s}")
        print(f"  {'-'*28} {'-'*15} {'-'*8} {'-'*8}")

        # Current (from C22 replan_8)
        if "c22_replan" in results:
            for key, res in results["c22_replan"].items():
                label = f"  replan_{res['n_action_exec']}"
                print(f"  {label:<28s} {'8-step (bug)':<15s} {res['n_action_exec']:<8d} "
                      f"{res['mean_max_reward']:8.4f}")

        # Fixed buffer
        if "c23_fixed_buffer" in results:
            for key, res in results["c23_fixed_buffer"].items():
                label = f"  fixed-buffer_exec_{res['n_action_exec']}"
                print(f"  {label:<28s} {'1-step (fix)':<15s} {res['n_action_exec']:<8d} "
                      f"{res['mean_max_reward']:8.4f}")

        print(f"\n  All results saved under: {args.output_dir}/")


if __name__ == "__main__":
    main()
