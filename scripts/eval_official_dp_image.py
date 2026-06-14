#!/usr/bin/env python3
"""Step 5: Run official DP image checkpoint (workspace format) on same 50 seeds
with deterministic seeding.

Loads the Hydra workspace checkpoint and evaluates DiffusionUnetHybridImagePolicy
on the same seeds as our deterministic paired eval.
"""
import argparse, json, os, sys, time, random
from pathlib import Path
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.common.pytorch_util import dict_apply


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


def load_official_model(checkpoint_path, device):
    """Load official DP image model from workspace checkpoint."""
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # Instantiate policy
    policy = hydra.utils.instantiate(cfg.policy)
    if cfg.training.use_ema:
        # Load EMA model state
        ema_state = payload["state_dicts"]["ema_model"]
        policy.load_state_dict(ema_state)
    else:
        policy.load_state_dict(payload["state_dicts"]["model"])

    policy.to(device)
    policy.eval()
    return policy, cfg


def obs_to_device(obs, device):
    """Move obs dict to device, handling numpy arrays."""
    result = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v).to(device)
        elif isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


def eval_episode(policy, device, seed):
    """Run one deterministic episode with official DP image model."""
    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
    env.seed(seed)
    obs = env.reset()
    rewards = []
    done = False
    step = 0

    while not done and step < 300:
        # Convert numpy to torch, add batch dim: env returns (T, C, H, W), model expects (B, T, C, H, W)
        obs_batched = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                obs_batched[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            else:
                obs_batched[k] = v.unsqueeze(0).to(device) if v.dim() > 0 else v.to(device)
        with torch.no_grad():
            action_dict = policy.predict_action(obs_batched)
        action = action_dict["action"].cpu().numpy().squeeze(0)

        obs, reward, done, info = env.step(action)
        rewards.append(float(reward))
        done = bool(np.all(done))
        step += 1

    return float(max(rewards)) if rewards else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="artifacts_keep/dp_50epoch/latest.ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-eps", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--output-dir", type=str,
                        default="outputs/deterministic_paired_eval/seeds_100000_100049")
    parser.add_argument("--inference-steps", type=int, default=None,
                        help="Override inference steps (default: use config value)")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_cudnn_deterministic()
    seeds = [args.seed_start + i for i in range(args.n_eps)]

    print("=" * 60)
    print("Step 5: Official DP Image Baseline (Deterministic)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)} episodes)")
    print("=" * 60)

    print("\nLoading official DP image model...", flush=True)
    policy, cfg = load_official_model(args.checkpoint, device)
    # Override inference steps if specified
    if args.inference_steps is not None:
        policy.num_inference_steps = args.inference_steps
        print(f"  Inference steps: {args.inference_steps} (overridden from {cfg.policy.num_inference_steps})")
    else:
        print(f"  Inference steps: {cfg.policy.num_inference_steps}")
    print(f"  Policy: {cfg.policy._target_}")
    print(f"  EMA: {cfg.training.use_ema}")
    print(f"  Clip sample: {cfg.policy.noise_scheduler.clip_sample}")

    print(f"\nEvaluating on {len(seeds)} seeds...", flush=True)
    scores = []
    t0 = time.time()

    for i, seed in enumerate(seeds):
        seed_everything(seed)
        score = eval_episode(policy, device, seed)
        scores.append(score)
        if i < 5 or i % 10 == 0 or i == len(seeds) - 1:
            print(f"  [Official DP] Ep {i:3d} (seed={seed}): max_reward={score:.4f}", flush=True)

    elapsed = time.time() - t0
    arr = np.array(scores)
    print(f"\n  [Official DP] {len(seeds)} eps in {elapsed:.0f}s: "
          f"mean={arr.mean():.4f} median={np.median(arr):.4f} "
          f"ep>0.5={(arr > 0.5).sum()}/{len(seeds)}", flush=True)

    # Save scores
    os.makedirs(args.output_dir, exist_ok=True)
    scores_path = os.path.join(args.output_dir, "official_dp_image_scores.csv")
    with open(scores_path, "w") as f:
        f.write("ep,seed,score\n")
        for i, seed in enumerate(seeds):
            f.write(f"{i},{seed},{scores[i]:.6f}\n")

    summary = {
        "checkpoint": args.checkpoint,
        "policy_type": cfg.policy._target_,
        "ema": cfg.training.use_ema,
        "inference_steps": int(cfg.policy.num_inference_steps),
        "clip_sample": bool(cfg.policy.noise_scheduler.clip_sample),
        "num_episodes": len(seeds),
        "seeds": seeds,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "ep_gt_05": int((arr > 0.5).sum()),
    }
    summary_path = os.path.join(args.output_dir, "official_dp_image_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {scores_path}")
    print(f"Saved: {summary_path}")

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON (same 50 seeds 100000-100049)")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Mean':>8} {'Median':>8} {'Std':>8} {'ep>0.5':>10}")
    print("-" * 58)
    print(f"  {'Official DP image':<23} {arr.mean():8.4f} {np.median(arr):8.4f} "
          f"{arr.std():8.4f} {(arr > 0.5).sum():>8}/{len(seeds)}")
    print(f"  {'UWM joint C':<23} {'0.4342':>8} {'0.4057':>8} "
          f"{'0.3850':>8} {'20/50':>10}")
    print(f"  {'DP-only C (ours)':<23} {'0.3912':>8} {'0.3909':>8} "
          f"{'0.3456':>8} {'17/50':>10}")

    print("\nDone.")


if __name__ == "__main__":
    main()
