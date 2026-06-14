#!/usr/bin/env python3
"""Step C2-3: Fixed-buffer eval — keep n_action_exec=8, fix obs spacing to [t-1, t].

Bypasses MultiStepWrapper. Manually steps raw PushTImageEnv 8 times per
policy call, collecting full 5D state after each single physical step.
This ensures obs_buffer always has states 1 physical step apart,
matching the training distribution.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_model(checkpoint_path, device):
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


def get_full_state(env):
    """Extract full 5D state from PushTImageEnv internals."""
    agent_p = np.array(env.agent.position)
    block_p = np.array(env.block.position)
    block_ang = env.block.angle
    return np.concatenate([agent_p, block_p, [block_ang]])


def run_old_wrapper_eval(model, norm_state, unnorm_action, device, n_eps=10, verbose=True):
    """Current eval (MultiStepWrapper, spacing=8), with action chatter logging."""
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    if verbose:
        print(f"\n{'='*60}")
        print(f"OLD WRAPPER eval (n_exec=8, spacing=8)")
        print(f"{'='*60}")

    all_results = []
    all_deltas = []

    for ep in range(n_eps):
        env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
        env.seed(ep)
        obs = env.reset()
        rewards = []
        state_buffer = []
        done = False
        step = 0
        ep_deltas = []
        ep_phys_steps = []

        while not done and step < 300:
            full_state = get_full_state(env.env)
            state_buffer.append(full_state)
            if len(state_buffer) > 2:
                state_buffer = state_buffer[-2:]

            if len(state_buffer) < 2:
                obs, reward, done, info = env.step(np.zeros((8, 2)))
                rewards.append(float(reward))
                step += 1
                continue

            state_np = np.stack(state_buffer)
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_normed = norm_state(state_t)

            with torch.no_grad():
                action_norm = model.sample(state_normed)[0]
            action_raw = unnorm_action(action_norm)
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

            # Action chatter
            exec_actions = action_raw[:8]
            delta = exec_actions[1:8] - exec_actions[0:7]
            delta_l2 = np.linalg.norm(delta, axis=-1)
            ep_deltas.append({"mean": float(delta_l2.mean()), "max": float(delta_l2.max())})

            obs, reward, done, info = env.step(exec_actions)
            rewards.append(float(reward))
            step += 1

        all_results.append({
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "steps": step,
        })
        all_deltas.append({"mean": float(np.mean([d["mean"] for d in ep_deltas])),
                           "max": float(np.max([d["max"] for d in ep_deltas]))})

        if verbose and (ep < 5 or ep % 10 == 0):
            print(f"  Ep {ep:3d}: max_reward={all_results[-1]['max_reward']:.4f}  "
                  f"action_delta_mean={all_deltas[-1]['mean']:.1f}  max={all_deltas[-1]['max']:.1f}",
                  flush=True)

    mean_max = float(np.mean([r["max_reward"] for r in all_results]))
    std_max = float(np.std([r["max_reward"] for r in all_results]))
    mean_delta = float(np.mean([d["mean"] for d in all_deltas]))
    max_delta = float(np.max([d["max"] for d in all_deltas]))

    if verbose:
        print(f"\n  >>> old-wrapper: mean_max={mean_max:.4f} std={std_max:.4f}  "
              f"action_delta: mean={mean_delta:.1f} max={max_delta:.1f}")

    return {"mean_max_reward": mean_max, "std_max_reward": std_max,
            "action_delta_mean": mean_delta, "action_delta_max": max_delta,
            "results": all_results, "deltas": all_deltas}


def run_fixed_buffer_eval(model, norm_state, unnorm_action, device, n_eps=10,
                           n_action_exec=8, debug_ep=1, verbose=True):
    """Fixed-buffer eval: bypass MultiStepWrapper, step raw env 1 action at a time,
    collect state after EACH physical step → obs_buffer has [t-1, t] spacing."""
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    n_action_exec = 8  # HARDCODED: keep control variable fixed

    if verbose:
        print(f"\n{'='*60}")
        print(f"FIXED-BUFFER eval (n_exec={n_action_exec}, spacing=1)")
        print(f"{'='*60}")

    all_results = []
    all_deltas = []
    all_spacing_log = []

    for ep in range(n_eps):
        raw_env = PushTImageEnv(legacy=True)
        raw_env.seed(ep)
        raw_env.reset()

        obs_buffer = deque(maxlen=2)  # will store (full_state_5d, physical_step_idx)
        obs_time_buffer = deque(maxlen=2)  # physical step indices for spacing audit
        rewards = []
        done = False
        physical_step = 0
        policy_call = 0
        ep_deltas = []
        ep_spacing_log = []

        # Prime: read s0, execute 1 zero action, read s1.
        # obs = [s0, s1]. Env at s1, ready for model prediction.
        # Model(s0,s1) predicts action[0] which goes FROM s1.
        s0 = get_full_state(raw_env)
        obs_buffer.append(s0)
        obs_time_buffer.append(physical_step)

        raw_env.step(np.zeros(2))  # s0 →[zero]→ s1
        physical_step += 1

        s1 = get_full_state(raw_env)
        obs_buffer.append(s1)
        obs_time_buffer.append(physical_step)
        # Env is at s1. Next model prediction's action[0] goes from s1.

        while not done and physical_step < 300:
            # Build model input from obs_buffer: [t-1, t]
            state_np = np.stack(list(obs_buffer))  # [2, 5]
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_normed = norm_state(state_t)

            with torch.no_grad():
                action_norm = model.sample(state_normed)[0]
            action_raw = unnorm_action(action_norm)
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

            exec_actions = action_raw[:n_action_exec]

            # Action chatter metric
            delta = exec_actions[1:] - exec_actions[:-1]
            delta_l2 = np.linalg.norm(delta, axis=-1)
            ep_deltas.append({"mean": float(delta_l2.mean()), "max": float(delta_l2.max())})

            # Execute actions ONE BY ONE, collecting state after each
            for k in range(n_action_exec):
                act = exec_actions[k]
                _obs, reward, _done, _info = raw_env.step(act)
                physical_step += 1

                # Read state immediately after this single physical step
                full_state = get_full_state(raw_env)
                obs_buffer.append(full_state)
                obs_time_buffer.append(physical_step)

                # reward IS coverage (from PushTImageEnv)
                rewards.append(float(reward))

                if _done or physical_step >= 300:
                    done = True
                    break

            # Log spacing for debug
            if ep < debug_ep and policy_call < 10:
                spacing = obs_time_buffer[-1] - obs_time_buffer[-2]
                s0 = list(obs_buffer)[0][:2]  # agent_xy
                s1 = list(obs_buffer)[1][:2]
                ep_spacing_log.append({
                    "policy_call": policy_call,
                    "physical_step": physical_step,
                    "obs_time": list(obs_time_buffer),
                    "spacing": spacing,
                    "agent_xy_0": s0.tolist(),
                    "agent_xy_1": s1.tolist(),
                    "action_delta_mean": float(delta_l2.mean()),
                    "action_delta_max": float(delta_l2.max()),
                    "actions_first_8": exec_actions.tolist(),
                })

            if verbose and ep < debug_ep and policy_call < 10:
                spacing = obs_time_buffer[-1] - obs_time_buffer[-2]
                print(f"  policy_call={policy_call:2d}  phys_step={physical_step:4d}  "
                      f"obs_time={list(obs_time_buffer)}  spacing={spacing}  "
                      f"agent_xy[0]={s0}  agent_xy[1]={s1}  "
                      f"act_delta_mean={delta_l2.mean():.1f}  max={delta_l2.max():.1f}",
                      flush=True)

            policy_call += 1

        max_reward = float(max(rewards)) if rewards else 0.0
        all_results.append({"max_reward": max_reward, "steps": physical_step})
        all_deltas.append({"mean": float(np.mean([d["mean"] for d in ep_deltas])),
                           "max": float(np.max([d["max"] for d in ep_deltas]))})
        all_spacing_log.append(ep_spacing_log)

        if verbose and (ep < 5 or ep % 10 == 0):
            print(f"  Ep {ep:3d}: max_reward={max_reward:.4f}  "
                  f"action_delta_mean={all_deltas[-1]['mean']:.1f}  "
                  f"max={all_deltas[-1]['max']:.1f}  steps={physical_step}",
                  flush=True)

    mean_max = float(np.mean([r["max_reward"] for r in all_results]))
    std_max = float(np.std([r["max_reward"] for r in all_results]))
    mean_delta = float(np.mean([d["mean"] for d in all_deltas]))
    max_delta = float(np.max([d["max"] for d in all_deltas]))
    high = sum(1 for r in all_results if r["max_reward"] > 0.5)

    if verbose:
        print(f"\n  >>> fixed-buffer: mean_max={mean_max:.4f} std={std_max:.4f}  "
              f"ep>0.5={high}/{n_eps}  "
              f"action_delta: mean={mean_delta:.1f} max={max_delta:.1f}")

    return {"mean_max_reward": mean_max, "std_max_reward": std_max,
            "ep_gt_05": high,
            "action_delta_mean": mean_delta, "action_delta_max": max_delta,
            "results": all_results, "deltas": all_deltas,
            "spacing_log": all_spacing_log}


def main():
    parser = argparse.ArgumentParser(description="Step C2-3: Fixed-buffer eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/stepB_retrain/B2_minmax_clip/latest.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/stepC2")
    parser.add_argument("--n-eps", type=int, default=10,
                        help="10 for sanity, 50 for final")
    parser.add_argument("--skip-old", action="store_true",
                        help="Skip old-wrapper baseline (for speed)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.join(args.output_dir, "c23_fixed_buffer")
    os.makedirs(out_dir, exist_ok=True)

    model, norm_state, unnorm_action, norm_type = load_model(args.checkpoint, device)
    print(f"Loaded model: {args.checkpoint}")
    print(f"Normalizer: {norm_type}")
    print(f"Episodes: {args.n_eps}")

    all_data = {}

    # Old wrapper baseline
    if not args.skip_old:
        t0 = time.time()
        old_res = run_old_wrapper_eval(model, norm_state, unnorm_action, device,
                                       n_eps=args.n_eps)
        print(f"  Old wrapper elapsed: {time.time()-t0:.1f}s")
        all_data["old_wrapper"] = old_res

    # Fixed buffer
    t0 = time.time()
    new_res = run_fixed_buffer_eval(model, norm_state, unnorm_action, device,
                                    n_eps=args.n_eps, n_action_exec=8, debug_ep=1)
    print(f"  Fixed buffer elapsed: {time.time()-t0:.1f}s")
    all_data["fixed_buffer"] = new_res

    # Save results
    summary = {
        "checkpoint": args.checkpoint,
        "n_eps": args.n_eps,
        "old_wrapper_score": all_data.get("old_wrapper", {}).get("mean_max_reward"),
        "old_wrapper_std": all_data.get("old_wrapper", {}).get("std_max_reward"),
        "old_wrapper_action_delta_mean": all_data.get("old_wrapper", {}).get("action_delta_mean"),
        "old_wrapper_action_delta_max": all_data.get("old_wrapper", {}).get("action_delta_max"),
        "fixed_buffer_score": all_data["fixed_buffer"]["mean_max_reward"],
        "fixed_buffer_std": all_data["fixed_buffer"]["std_max_reward"],
        "fixed_buffer_ep_gt_05": all_data["fixed_buffer"]["ep_gt_05"],
        "fixed_buffer_action_delta_mean": all_data["fixed_buffer"]["action_delta_mean"],
        "fixed_buffer_action_delta_max": all_data["fixed_buffer"]["action_delta_max"],
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        # Convert to serializable
        json.dump(summary, f, indent=2)

    # Save detailed per-episode data
    with open(os.path.join(out_dir, "full_results.json"), "w") as f:
        full = {
            "old_wrapper_results": [{"max_reward": r["max_reward"], "steps": r["steps"]}
                                    for r in all_data.get("old_wrapper", {}).get("results", [])],
            "fixed_buffer_results": [{"max_reward": r["max_reward"], "steps": r["steps"]}
                                     for r in all_data["fixed_buffer"]["results"]],
            "fixed_buffer_deltas": all_data["fixed_buffer"]["deltas"],
            "fixed_buffer_spacing_log": all_data["fixed_buffer"].get("spacing_log", []),
        }
        json.dump(full, f, indent=2)

    print(f"\n  Saved to: {out_dir}/")
    print(f"\n{'='*60}")
    print(f"C2-3 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Run':<18s} {'n_exec':>6s} {'Spacing':>8s} {'ActDelta_mean':>14s} "
          f"{'ActDelta_max':>14s} {'Score':>8s} {'Std':>8s}")
    print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*14} {'-'*14} {'-'*8} {'-'*8}")
    if not args.skip_old:
        print(f"  {'old-wrapper':<18s} {8:>6d} {8:>8d} "
              f"{summary['old_wrapper_action_delta_mean']:>14.1f} "
              f"{summary['old_wrapper_action_delta_max']:>14.1f} "
              f"{summary['old_wrapper_score']:>8.4f} "
              f"{summary['old_wrapper_std']:>8.4f}")
    print(f"  {'fixed-buffer':<18s} {8:>6d} {1:>8d} "
          f"{summary['fixed_buffer_action_delta_mean']:>14.1f} "
          f"{summary['fixed_buffer_action_delta_max']:>14.1f} "
          f"{summary['fixed_buffer_score']:>8.4f} "
          f"{summary['fixed_buffer_std']:>8.4f}")

    # Spacing audit
    if all_data["fixed_buffer"].get("spacing_log") and all_data["fixed_buffer"]["spacing_log"][0]:
        log = all_data["fixed_buffer"]["spacing_log"][0]
        spacings = [e["spacing"] for e in log]
        print(f"\n  Spacing audit (ep 0): {spacings}")
        if all(s == 1 for s in spacings[2:]):
            print(f"  SPACING: OK (all = 1 after priming)")
        else:
            print(f"  SPACING: BUG (not all = 1)")


if __name__ == "__main__":
    main()
