#!/usr/bin/env python3
"""Step D1: Diffusion sampling / temporal consistency audit.

D1-1: Same-state resampling variance (K=32 samples per obs)
D1-2: No-retrain sampler ablation (deterministic, more steps, fixed noise)
D1-3: Action temporal consistency / cross-plan jump audit
D1-4: No-retrain temporal ensembling ablation
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---- Model loading ----
def load_model(checkpoint_path, device, num_inference_steps=10):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stepB_retrain_lowdim",
        str(Path(__file__).resolve().parent / "stepB_retrain_lowdim.py"))
    stepB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stepB)

    if num_inference_steps != 10:
        class LowdimPolicyV2CustomSteps(stepB.LowdimStatePolicyV2):
            def __init__(self, **kwargs):
                kwargs["num_inference_steps"] = num_inference_steps
                super().__init__(**kwargs)
        ModelClass = LowdimPolicyV2CustomSteps
    else:
        ModelClass = stepB.LowdimStatePolicyV2

    ckpt = torch.load(checkpoint_path, map_location=device)
    clip_sample_flag = ckpt.get("clip_sample", True)
    model = ModelClass(clip_sample=clip_sample_flag, num_inference_steps=num_inference_steps).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    norm_type = ckpt.get("norm_type", "meanstd")
    an = ckpt["action_normalizer"]
    sn = ckpt["state_normalizer"]

    if norm_type == "minmax":
        a_offset = torch.tensor(an["offset"], device=device).float()
        a_scale = torch.tensor(an["scale"], device=device).float()
        s_offset = torch.tensor(sn["offset"], device=device).float()
        s_scale = torch.tensor(sn["scale"], device=device).float()
        def norm_state(x):
            return (x - s_offset) / s_scale
        def unnorm_action(x):
            return x * a_scale + a_offset
        def norm_action(x):
            return (x - a_offset) / a_scale
    else:
        a_mean = torch.tensor(an["mean"], device=device).float()
        a_std = torch.tensor(an["std"], device=device).float()
        s_mean = torch.tensor(sn["mean"], device=device).float()
        s_std = torch.tensor(sn["std"], device=device).float()
        def norm_state(x):
            return (x - s_mean) / s_std
        def unnorm_action(x):
            return x * a_std + a_mean
        def norm_action(x):
            return (x - a_mean) / a_std

    return model, norm_state, unnorm_action, norm_action, norm_type


def get_full_state(env):
    agent_p = np.array(env.agent.position)
    block_p = np.array(env.block.position)
    block_ang = env.block.angle
    return np.concatenate([agent_p, block_p, [block_ang]])


# =========================================================================
# D1-1: Same-state resampling variance
# =========================================================================
def d11_same_state_variance(model, norm_state, unnorm_action, device, checkpoint_path):
    print("=" * 70)
    print("D1-1: SAME-STATE RESAMPLING VARIANCE")
    print("=" * 70)

    # Load validation states from zarr
    import zarr
    z = zarr.open("diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr", "r")
    all_state = z["data/state"][:]
    all_action = z["data/action"][:]
    ep_ends = z["meta/episode_ends"][:]

    # Use expert states: collect pairs [s_i, s_{i+1}] as model input
    obs_pairs = []
    gt_actions = []
    start = 0
    for end in ep_ends:
        for i in range(start, end - 2):
            obs_pairs.append(all_state[i:i+2])      # [2, 5]
            gt_actions.append(all_action[i+1:i+17])  # [16, 2]
        start = end

    # Take 128 random samples
    indices = np.random.RandomState(42).choice(len(obs_pairs), min(128, len(obs_pairs)), replace=False)
    obs_batch = np.stack([obs_pairs[i] for i in indices])  # [128, 2, 5]

    K = 32
    # Sample K times from each obs
    obs_t = torch.from_numpy(obs_batch).float().to(device)
    obs_norm = norm_state(obs_t)

    all_samples = []
    torch.manual_seed(12345)
    for k in range(K):
        with torch.no_grad():
            for i in range(0, len(obs_norm), 16):
                batch = obs_norm[i:i+16]
                pred_norm = model.sample(batch)
                all_samples.append(pred_norm.cpu())
    all_pred_norm = torch.cat(all_samples, dim=0)  # [128*K, 16, 2]

    # Unnormalize
    all_pred_raw_list = []
    for i in range(0, len(all_pred_norm), 128):
        pred_t = all_pred_norm[i:i+128].to(device)
        pred_raw = unnorm_action(pred_t).cpu().numpy()
        all_pred_raw_list.append(pred_raw)
    all_pred_raw = np.stack(all_pred_raw_list)  # [K, 128, 16, 2]

    # Per-sample variance statistics
    # first_action_std: std of first action across K samples, averaged over 128 obs
    first_actions = all_pred_raw[:, :, 0, :]  # [K, 128, 2]
    first_action_std = first_actions.std(axis=0)  # [128, 2]
    first_action_std_mean = first_action_std.mean(axis=0)

    # first_action_l2_std
    first_action_l2 = np.linalg.norm(first_actions - first_actions.mean(axis=0), axis=-1)  # [K, 128]
    first_action_l2_std = first_action_l2.std(axis=0).mean()

    # action_seq_std: std of full action sequence across K samples
    action_seq_std = all_pred_raw.std(axis=0)  # [128, 16, 2]
    action_seq_std_mean = action_seq_std.mean()
    action_seq_std_max = action_seq_std.max()

    # plan_delta: within-plan action deltas
    plan_deltas = []
    for k in range(K):
        for n in range(128):
            delta = all_pred_raw[k, n, 1:8] - all_pred_raw[k, n, 0:7]
            plan_deltas.append(np.linalg.norm(delta, axis=-1))
    plan_deltas = np.concatenate(plan_deltas)

    print(f"  Samples: {len(obs_batch)} obs x {K} resamples = {len(obs_batch)*K} action sequences")
    print(f"  first_action_std   dim0: {first_action_std_mean[0]:.2f}")
    print(f"  first_action_std   dim1: {first_action_std_mean[1]:.2f}")
    print(f"  first_action_l2_std:     {first_action_l2_std:.2f}")
    print(f"  action_seq_std_mean:     {action_seq_std_mean:.2f}")
    print(f"  action_seq_std_max:      {action_seq_std_max:.2f}")
    print(f"  plan_delta_mean:         {plan_deltas.mean():.2f}")
    print(f"  plan_delta_max:          {plan_deltas.max():.2f}")

    # Expert action range for context
    print(f"  Expert action std:       {all_action.std(axis=0)}")
    print(f"  Expert action range:     [{all_action.min(axis=0)}, {all_action.max(axis=0)}]")

    result = {
        "first_action_std_dim0": float(first_action_std_mean[0]),
        "first_action_std_dim1": float(first_action_std_mean[1]),
        "first_action_l2_std": float(first_action_l2_std),
        "action_seq_std_mean": float(action_seq_std_mean),
        "action_seq_std_max": float(action_seq_std_max),
        "plan_delta_mean": float(plan_deltas.mean()),
        "plan_delta_max": float(plan_deltas.max()),
    }

    # Also try deterministic: set seed before each sample call
    print(f"\n  --- Deterministic check (same seed per obs) ---")
    torch.manual_seed(42)
    all_pred_norm1 = []
    for i in range(0, len(obs_norm), 16):
        batch = obs_norm[i:i+16]
        with torch.no_grad():
            pred_norm = model.sample(batch)
        all_pred_norm1.append(pred_norm.cpu())
    pred1 = torch.cat(all_pred_norm1, dim=0)

    torch.manual_seed(42)  # reset
    all_pred_norm2 = []
    for i in range(0, len(obs_norm), 16):
        batch = obs_norm[i:i+16]
        with torch.no_grad():
            pred_norm = model.sample(batch)
        all_pred_norm2.append(pred_norm.cpu())
    pred2 = torch.cat(all_pred_norm2, dim=0)

    diff = (pred1 - pred2).abs().max().item()
    print(f"  Max diff between two seeded runs: {diff:.10f}")
    print(f"  Deterministic? {'YES' if diff < 1e-6 else 'NO (diff=' + str(diff) + ')'}")

    return result


# =========================================================================
# D1-2: Sampler ablation (no retrain, change inference config)
# =========================================================================
def run_eval_fixed_buffer_with_model(model, norm_state, unnorm_action, device,
                                     n_eps, n_action_exec=8, verbose=True,
                                     seed_offset=0, use_fixed_noise=False,
                                     label=""):
    """Fixed-buffer eval with optional fixed-noise mode."""
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    results = []
    all_deltas = []
    all_cross_jumps = []
    noise_history = None

    for ep in range(n_eps):
        raw_env = PushTImageEnv(legacy=True)
        raw_env.seed(ep + seed_offset)
        raw_env.reset()

        obs_buffer = deque(maxlen=2)
        obs_time_buffer = deque(maxlen=2)
        rewards = []
        done = False
        physical_step = 0
        policy_call = 0
        ep_deltas = []
        ep_cross_jumps = []
        previous_plan = None

        # Prime
        s0 = get_full_state(raw_env)
        obs_buffer.append(s0)
        obs_time_buffer.append(physical_step)
        raw_env.step(np.zeros(2))
        physical_step += 1
        s1 = get_full_state(raw_env)
        obs_buffer.append(s1)
        obs_time_buffer.append(physical_step)

        while not done and physical_step < 300:
            state_np = np.stack(list(obs_buffer))
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_normed = norm_state(state_t)

            if use_fixed_noise:
                torch.manual_seed(hash(str(policy_call)) % (2**31))

            with torch.no_grad():
                action_norm = model.sample(state_normed)[0]
            action_raw = unnorm_action(action_norm)
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

            exec_actions = action_raw[:n_action_exec]

            # Within-plan delta
            delta = exec_actions[1:] - exec_actions[:-1]
            delta_l2 = np.linalg.norm(delta, axis=-1)
            ep_deltas.append({"mean": float(delta_l2.mean()), "max": float(delta_l2.max())})

            # Cross-plan jump: current_plan[0] vs previous_plan[n_action_exec]
            if previous_plan is not None and len(previous_plan) > n_action_exec:
                jump = np.linalg.norm(exec_actions[0] - previous_plan[n_action_exec])
                ep_cross_jumps.append(float(jump))

            # Execute actions one by one
            for k in range(n_action_exec):
                act = exec_actions[k]
                _obs, reward, _done, _info = raw_env.step(act)
                physical_step += 1
                full_state = get_full_state(raw_env)
                obs_buffer.append(full_state)
                obs_time_buffer.append(physical_step)
                rewards.append(float(reward))
                if _done or physical_step >= 300:
                    done = True
                    break

            previous_plan = action_raw.copy()
            policy_call += 1

        max_reward = float(max(rewards)) if rewards else 0.0
        results.append({"max_reward": max_reward, "steps": physical_step})
        all_deltas.append({"mean": float(np.mean([d["mean"] for d in ep_deltas])),
                           "max": float(np.max([d["max"] for d in ep_deltas]))})
        if ep_cross_jumps:
            all_cross_jumps.append({"mean": float(np.mean(ep_cross_jumps)),
                                    "max": float(np.max(ep_cross_jumps))})

        if verbose and (ep < 5 or ep % 10 == 0):
            cj_str = f"cross_jump_mean={all_cross_jumps[-1]['mean']:.1f}" if all_cross_jumps else ""
            print(f"  Ep {ep:3d}: max_reward={max_reward:.4f}  delta_mean={all_deltas[-1]['mean']:.1f}  {cj_str}",
                  flush=True)

    mean_max = float(np.mean([r["max_reward"] for r in results]))
    std_max = float(np.std([r["max_reward"] for r in results]))
    mean_delta = float(np.mean([d["mean"] for d in all_deltas]))
    max_delta = float(np.max([d["max"] for d in all_deltas]))
    ep_gt_05 = sum(1 for r in results if r["max_reward"] > 0.5)
    cross_jump_mean = float(np.mean([c["mean"] for c in all_cross_jumps])) if all_cross_jumps else 0
    cross_jump_max = float(np.max([c["max"] for c in all_cross_jumps])) if all_cross_jumps else 0

    return {"mean_max_reward": mean_max, "std_max_reward": std_max,
            "ep_gt_05": ep_gt_05, "n_episodes": n_eps,
            "action_delta_mean": mean_delta, "action_delta_max": max_delta,
            "cross_jump_mean": cross_jump_mean, "cross_jump_max": cross_jump_max}


# =========================================================================
# D1-4: Temporal ensembling
# =========================================================================
def run_eval_temporal_ensemble(model, norm_state, unnorm_action, device,
                               n_eps, n_action_exec=8, lambda_=0.0,
                               variant="two_plan", verbose=True):
    """Temporal ensembling eval with fixed-buffer."""
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    results = []
    all_deltas = []
    all_cross_jumps = []

    for ep in range(n_eps):
        raw_env = PushTImageEnv(legacy=True)
        raw_env.seed(ep)
        raw_env.reset()

        obs_buffer = deque(maxlen=2)
        rewards = []
        done = False
        physical_step = 0
        policy_call = 0
        ep_deltas = []
        ep_cross_jumps = []
        previous_plan = None

        if variant == "action_buffer":
            action_candidates = defaultdict(list)

        # Prime
        s0 = get_full_state(raw_env)
        obs_buffer.append(s0)
        raw_env.step(np.zeros(2))
        physical_step += 1
        s1 = get_full_state(raw_env)
        obs_buffer.append(s1)

        while not done and physical_step < 300:
            state_np = np.stack(list(obs_buffer))
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_normed = norm_state(state_t)

            with torch.no_grad():
                action_norm = model.sample(state_normed)[0]
            action_raw = unnorm_action(action_norm)
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

            if variant == "two_plan":
                # Overlap ensemble
                exec_actions = np.zeros((n_action_exec, 2))
                for k in range(n_action_exec):
                    new_action = action_raw[k]
                    if previous_plan is not None and len(previous_plan) > n_action_exec + k:
                        old_action = previous_plan[n_action_exec + k]
                        w_new = np.exp(-lambda_ * k)
                        w_old = np.exp(-lambda_ * (n_action_exec + k))
                        exec_actions[k] = (w_new * new_action + w_old * old_action) / (w_new + w_old)
                    else:
                        exec_actions[k] = new_action
            elif variant == "action_buffer":
                # Action buffer ensemble
                abs_t_start = physical_step
                for h in range(16):
                    abs_t = abs_t_start + h
                    weight = np.exp(-lambda_ * h)
                    action_candidates[abs_t].append((action_raw[h], weight))

                exec_actions = np.zeros((n_action_exec, 2))
                for k in range(n_action_exec):
                    abs_t = physical_step + k
                    if abs_t in action_candidates:
                        cands = action_candidates[abs_t]
                        total_w = sum(w for _, w in cands)
                        exec_actions[k] = sum(a * w for a, w in cands) / total_w
                    else:
                        exec_actions[k] = action_raw[k]
            else:
                exec_actions = action_raw[:n_action_exec]

            # Within-plan delta
            delta = exec_actions[1:] - exec_actions[:-1]
            delta_l2 = np.linalg.norm(delta, axis=-1)
            ep_deltas.append({"mean": float(delta_l2.mean()), "max": float(delta_l2.max())})

            if previous_plan is not None and len(previous_plan) > n_action_exec:
                jump = np.linalg.norm(exec_actions[0] - previous_plan[n_action_exec])
                ep_cross_jumps.append(float(jump))

            for k in range(n_action_exec):
                act = exec_actions[k]
                _obs, reward, _done, _info = raw_env.step(act)
                physical_step += 1
                full_state = get_full_state(raw_env)
                obs_buffer.append(full_state)
                rewards.append(float(reward))
                # Cleanup old timesteps
                if variant == "action_buffer":
                    expired = [t for t in list(action_candidates.keys()) if t <= physical_step]
                    for t in expired:
                        del action_candidates[t]
                if _done or physical_step >= 300:
                    done = True
                    break

            previous_plan = action_raw.copy()
            policy_call += 1

        max_reward = float(max(rewards)) if rewards else 0.0
        results.append({"max_reward": max_reward, "steps": physical_step})
        all_deltas.append({"mean": float(np.mean([d["mean"] for d in ep_deltas])),
                           "max": float(np.max([d["max"] for d in ep_deltas]))})
        if ep_cross_jumps:
            all_cross_jumps.append({"mean": float(np.mean(ep_cross_jumps)),
                                    "max": float(np.max(ep_cross_jumps))})

        if verbose and (ep < 5 or ep % 10 == 0):
            print(f"  Ep {ep:3d}: max_reward={max_reward:.4f}", flush=True)

    mean_max = float(np.mean([r["max_reward"] for r in results]))
    std_max = float(np.std([r["max_reward"] for r in results]))
    cross_jump_mean = float(np.mean([c["mean"] for c in all_cross_jumps])) if all_cross_jumps else 0

    return {"mean_max_reward": mean_max, "std_max_reward": std_max,
            "cross_jump_mean": cross_jump_mean,
            "action_delta_mean": float(np.mean([d["mean"] for d in all_deltas])),
            "ep_gt_05": sum(1 for r in results if r["max_reward"] > 0.5)}


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Step D1: Diffusion sampling audit")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/stepB_retrain/B2_minmax_clip/latest.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/stepD1")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["d11", "d12", "d13", "d14", "all"])
    parser.add_argument("--n-eps", type=int, default=10, help="Episodes for eval (10=sanity, 50=final)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # ===== D1-1: Same-state variance =====
    if args.mode in ("d11", "all"):
        model, norm_state, unnorm_action, norm_action, norm_type = load_model(
            args.checkpoint, device, num_inference_steps=10)
        d11_res = d11_same_state_variance(model, norm_state, unnorm_action, device, args.checkpoint)
        all_results["d11"] = d11_res
        with open(os.path.join(args.output_dir, "d11_variance.json"), "w") as f:
            json.dump(d11_res, f, indent=2)

    # ===== D1-2 & D1-3: Sampler ablation + temporal consistency =====
    if args.mode in ("d12", "d13", "all"):
        print("\n" + "=" * 70)
        print("D1-2 + D1-3: SAMPLER ABLATION + TEMPORAL CONSISTENCY")
        print("=" * 70)

        configs = [
            {"label": "baseline", "infer_steps": 10, "fixed_noise": False},
            {"label": "more_steps_50", "infer_steps": 50, "fixed_noise": False},
            {"label": "more_steps_100", "infer_steps": 100, "fixed_noise": False},
            {"label": "fixed_noise", "infer_steps": 10, "fixed_noise": True},
        ]

        sampler_results = {}
        for cfg in configs:
            print(f"\n  --- {cfg['label']}: infer_steps={cfg['infer_steps']} fixed_noise={cfg['fixed_noise']} ---")
            model, norm_state, unnorm_action, norm_action, norm_type = load_model(
                args.checkpoint, device, num_inference_steps=cfg["infer_steps"])
            t0 = time.time()
            res = run_eval_fixed_buffer_with_model(
                model, norm_state, unnorm_action, device,
                n_eps=args.n_eps, n_action_exec=8,
                use_fixed_noise=cfg["fixed_noise"],
                label=cfg["label"])
            elapsed = time.time() - t0
            res["elapsed"] = elapsed
            print(f"  Elapsed: {elapsed:.1f}s  score={res['mean_max_reward']:.4f}  "
                  f"cross_jump_mean={res['cross_jump_mean']:.1f}  "
                  f"delta_mean={res['action_delta_mean']:.1f}")
            sampler_results[cfg["label"]] = res

        all_results["d12_d13"] = sampler_results

        # Print summary table
        print(f"\n  {'Run':<18s} {'Steps':>6s} {'FixNoise':>8s} {'Score':>8s} {'CrossJump':>10s} {'Delta':>8s}")
        print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
        for cfg in configs:
            r = sampler_results[cfg["label"]]
            print(f"  {cfg['label']:<18s} {cfg['infer_steps']:>6d} {str(cfg['fixed_noise']):>8s} "
                  f"{r['mean_max_reward']:>8.4f} {r['cross_jump_mean']:>10.1f} {r['action_delta_mean']:>8.1f}")

        with open(os.path.join(args.output_dir, "d12_sampler_results.json"), "w") as f:
            json.dump(sampler_results, f, indent=2)

    # ===== D1-4: Temporal ensembling =====
    if args.mode in ("d14", "all"):
        print("\n" + "=" * 70)
        print("D1-4: TEMPORAL ENSEMBLING ABLATION")
        print("=" * 70)

        model, norm_state, unnorm_action, norm_action, norm_type = load_model(
            args.checkpoint, device, num_inference_steps=10)

        ensemble_configs = [
            {"label": "none", "variant": "none", "lambda_": None},
            {"label": "overlap_avg", "variant": "two_plan", "lambda_": 0.0},
            {"label": "overlap_lam0.1", "variant": "two_plan", "lambda_": 0.1},
            {"label": "overlap_lam0.25", "variant": "two_plan", "lambda_": 0.25},
            {"label": "overlap_lam0.5", "variant": "two_plan", "lambda_": 0.5},
        ]

        ensemble_results = {}
        for cfg in ensemble_configs:
            if cfg["variant"] == "none":
                print(f"\n  --- none (baseline) ---")
                t0 = time.time()
                res = run_eval_fixed_buffer_with_model(
                    model, norm_state, unnorm_action, device,
                    n_eps=args.n_eps, n_action_exec=8, label="none")
            else:
                print(f"\n  --- {cfg['label']}: variant={cfg['variant']} lambda={cfg['lambda_']} ---")
                t0 = time.time()
                res = run_eval_temporal_ensemble(
                    model, norm_state, unnorm_action, device,
                    n_eps=args.n_eps, n_action_exec=8,
                    lambda_=cfg["lambda_"], variant=cfg["variant"])
            elapsed = time.time() - t0
            res["elapsed"] = elapsed
            print(f"  Elapsed: {elapsed:.1f}s  score={res['mean_max_reward']:.4f}  "
                  f"cross_jump={res['cross_jump_mean']:.1f}")
            ensemble_results[cfg["label"]] = res

        all_results["d14"] = ensemble_results

        print(f"\n  {'Ensemble':<18s} {'Lambda':>8s} {'Score':>8s} {'CrossJump':>10s}")
        print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*10}")
        for cfg in ensemble_configs:
            r = ensemble_results[cfg["label"]]
            lam_str = str(cfg["lambda_"]) if cfg["lambda_"] is not None else "N/A"
            print(f"  {cfg['label']:<18s} {lam_str:>8s} {r['mean_max_reward']:>8.4f} "
                  f"{r['cross_jump_mean']:>10.1f}")

        with open(os.path.join(args.output_dir, "d14_ensemble_results.json"), "w") as f:
            json.dump(ensemble_results, f, indent=2)

    # Save full results
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        # Clean up non-serializable data
        clean = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                clean[k] = {kk: vv for kk, vv in v.items()
                           if not isinstance(vv, np.ndarray)}
            else:
                clean[k] = str(type(v))
        json.dump(clean, f, indent=2)

    print(f"\n  Full results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
