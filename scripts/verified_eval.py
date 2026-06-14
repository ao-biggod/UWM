#!/usr/bin/env python3
"""
Verified deterministic evaluation with synchronous video recording.
Must be called with CUBLAS_WORKSPACE_CONFIG=:4096:8 set BEFORE Python starts.

Usage:
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONHASHSEED=0
  python scripts/verified_eval.py --run scratch_lambda005 --seeds 100000 100049

Output: outputs/final_visual_summary_verified/
"""

import argparse, json, os, sys, time, random
from pathlib import Path
import numpy as np
import torch
import imageio
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "unified-world-model-main"))
sys.path.insert(0, str(ROOT / "diffusion_policy-main"))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from models.uwm.uwm import UnifiedWorldModel
from models.uwm.obs_encoder import UWMObservationEncoder

OUT_DIR = ROOT / "outputs" / "final_visual_summary_verified"
VID_DIR = OUT_DIR / "videos"
FIG_DIR = OUT_DIR / "figures"
FRAME_DIR = FIG_DIR / "frames"
JSON_DIR = OUT_DIR / "json"
CSV_DIR = OUT_DIR / "csv"

for d in [VID_DIR, FIG_DIR, FRAME_DIR, JSON_DIR, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model definitions ────────────────────────────────────────────
MODELS = {
    "joint_lambda1": {
        "label": "joint λ=1.0",
        "checkpoint": "outputs/uwm_pusht_crossattn_joint_C/main_20k/latest.pt",
        "color": "#E53935",
    },
    "R1_lambda0": {
        "label": "R1 λ=0.0",
        "checkpoint": "outputs/uwm_pusht_r1_loss_off/main_20k/latest.pt",
        "color": "#2196F3",
    },
    "cont_lambda0": {
        "label": "cont λ=0.0",
        "checkpoint": "outputs/uwm_pusht_r1_ft/lambda_0.00/latest.pt",
        "color": "#FF9800",
    },
    "ft_lambda005": {
        "label": "ft λ=0.05",
        "checkpoint": "outputs/uwm_pusht_r1_ft/lambda_0.05/latest.pt",
        "color": "#4CAF50",
    },
    "scratch_lambda005": {
        "label": "Scratch λ=0.05",
        "checkpoint": "outputs/uwm_pusht_scratch_lambda005/latest.pt",
        "color": "#9C27B0",
    },
    "warmup_0_to_005": {
        "label": "Warmup 0→0.05",
        "checkpoint": "outputs/uwm_pusht_warmup_step2/latest.pt",
        "color": "#607D8B",
    },
}


def set_full_determinism(seed):
    """Call ONCE per episode before env creation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(device, ckpt):
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


def run_one_episode(model, device, action_scale, action_offset,
                    ap_scale, ap_offset, seed, run_name, ep_idx,
                    save_video=True):
    """Run one fully deterministic episode, save video+frames synchronously."""
    set_full_determinism(seed)

    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
    env.seed(seed)
    obs = env.reset()

    frames = []
    rewards = []
    actions = []
    step = 0

    while step < 300:
        frame = env.env.render(mode="rgb_array")
        frames.append(frame)

        img = obs["image"]
        ap_raw = obs["agent_pos"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(ap_raw, np.ndarray):
            ap_raw = torch.from_numpy(ap_raw)

        img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        ap = ap_raw.float().to(device)
        ap_norm = (ap - torch.tensor(ap_offset, device=device).float()) / torch.tensor(ap_scale, device=device).float()

        obs_model = {
            "image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
            "agent_pos": ap_norm.unsqueeze(0),
        }

        with torch.no_grad():
            action_norm = model.sample(obs_model)[0]

        action_raw = (action_norm
                      * torch.tensor(action_scale, device=device).float()
                      + torch.tensor(action_offset, device=device).float())
        action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
        actions.append(action_raw_np[:8].tolist())

        obs, reward, done, _ = env.step(action_raw_np[:8])
        rewards.append(float(reward))
        if np.all(done):
            break
        step += 1

    # Final frame
    frame = env.env.render(mode="rgb_array")
    frames.append(frame)
    env.close()

    max_r = float(max(rewards)) if rewards else 0.0
    peak_t = int(np.argmax(rewards)) if rewards else 0
    final_r = float(rewards[-1]) if rewards else 0.0
    num_steps = len(rewards)

    # Save video
    score_str = f"{max_r:.3f}"
    vid_name = f"{run_name}_seed{seed}_score{score_str}.mp4"
    vid_path = VID_DIR / vid_name
    imageio.mimsave(str(vid_path), frames, fps=10)

    # Save peak frame
    peak_name = f"{run_name}_seed{seed}_peak.png"
    peak_path = FRAME_DIR / peak_name
    if peak_t < len(frames):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(frames[peak_t])
        ax.set_title(f"{run_name} seed={seed} peak_t={peak_t} r={max_r:.3f}", fontsize=7)
        ax.axis("off")
        fig.savefig(peak_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Save final frame
    final_name = f"{run_name}_seed{seed}_final.png"
    final_path = FRAME_DIR / final_name
    if len(frames) > 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(frames[-1])
        ax.set_title(f"{run_name} seed={seed} final_t={num_steps-1} r={final_r:.3f}", fontsize=7)
        ax.axis("off")
        fig.savefig(final_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    record = {
        "run_name": run_name,
        "seed": seed,
        "episode_index": ep_idx,
        "max_reward": max_r,
        "final_reward": final_r,
        "peak_timestep": peak_t,
        "num_steps": num_steps,
        "video_path": str(vid_path.relative_to(OUT_DIR)),
        "peak_frame_path": str(peak_path.relative_to(OUT_DIR)),
        "final_frame_path": str(final_path.relative_to(OUT_DIR)),
        "actions": actions,
    }

    return record


def eval_model(run_id, seeds, device, save_video=True):
    """Evaluate one model on all seeds, returning all episode records."""
    run_cfg = MODELS[run_id]
    ckpt_path = ROOT / run_cfg["checkpoint"]

    print(f"\n{'='*60}")
    print(f"Evaluating: {run_id} ({run_cfg['label']})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)} episodes)")
    print(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'NOT SET!')}")
    print(f"{'='*60}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    action_scale = np.array(ckpt["action_normalizer"]["scale"])
    action_offset = np.array(ckpt["action_normalizer"]["offset"])
    ap_scale = np.array(ckpt["lowdim_normalizer"]["agent_pos"]["scale"])
    ap_offset = np.array(ckpt["lowdim_normalizer"]["agent_pos"]["offset"])

    model = build_model(device, ckpt)
    del ckpt
    model.eval()

    records = []
    t_start = time.time()

    for i, seed in enumerate(seeds):
        t_ep = time.time()
        record = run_one_episode(
            model, device, action_scale, action_offset,
            ap_scale, ap_offset, seed, run_id, i,
            save_video=save_video,
        )
        records.append(record)

        elapsed_ep = time.time() - t_ep
        elapsed_total = time.time() - t_start
        eta = (elapsed_total / (i + 1)) * (len(seeds) - i - 1) if i < len(seeds) - 1 else 0
        print(f"  [{i+1:3d}/{len(seeds)}] seed={seed:6d}  max_r={record['max_reward']:.4f}  "
              f"peak_t={record['peak_timestep']:3d}  steps={record['num_steps']:3d}  "
              f"ep={elapsed_ep:.0f}s  ETA={eta:.0f}s")

    elapsed = time.time() - t_start
    scores = [r["max_reward"] for r in records]
    arr = np.array(scores)
    summary = {
        "run_name": run_id,
        "label": run_cfg["label"],
        "checkpoint": run_cfg["checkpoint"],
        "n_episodes": len(records),
        "seeds": [r["seed"] for r in records],
        "scores": scores,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "ep_gt_0_5": int(np.sum(arr > 0.5)),
        "elapsed_s": elapsed,
    }
    print(f"  DONE: mean={summary['mean']:.3f} median={summary['median']:.3f} "
          f"ep>0.5={summary['ep_gt_0_5']}/{len(records)} elapsed={elapsed:.0f}s")

    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Run IDs to evaluate (default: all 5 core)")
    parser.add_argument("--seeds-start", type=int, default=100000)
    parser.add_argument("--seeds-end", type=int, default=100049)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--priority", action="store_true",
                        help="Only run priority-3: joint, R1, scratch")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording (faster, for testing)")
    args = parser.parse_args()

    # Check determinism
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    if not cublas:
        print("=" * 60)
        print("⚠ WARNING: CUBLAS_WORKSPACE_CONFIG is NOT SET!")
        print("  Set before Python starts: export CUBLAS_WORKSPACE_CONFIG=:4096:8")
        print("  Continuing anyway, but results may NOT be reproducible.")
        print("=" * 60)

    device = torch.device(args.device)

    if args.runs:
        run_ids = args.runs
    elif args.priority:
        run_ids = ["joint_lambda1", "R1_lambda0", "scratch_lambda005"]
    else:
        run_ids = ["joint_lambda1", "R1_lambda0", "cont_lambda0",
                   "ft_lambda005", "scratch_lambda005"]

    seeds = list(range(args.seeds_start, args.seeds_end + 1))

    all_records = []
    all_summaries = []

    for run_id in run_ids:
        records, summary = eval_model(run_id, seeds, device,
                                       save_video=not args.no_video)
        all_records.extend(records)
        all_summaries.append(summary)

        # Save per-model records immediately
        rec_path = JSON_DIR / f"episode_records_{run_id}.json"
        with open(rec_path, "w") as f:
            json.dump(records, f, indent=1)
        print(f"  Saved: {rec_path}")

    # Save combined records
    combined_path = JSON_DIR / "episode_records.json"
    with open(combined_path, "w") as f:
        json.dump(all_records, f, indent=1)

    # Save summaries
    summary_path = JSON_DIR / "eval_summary_verified.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Save CSV
    import csv
    csv_path = CSV_DIR / "per_episode_scores_verified.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "seed", "episode_index", "max_reward",
                         "peak_timestep", "num_steps", "final_reward"])
        for r in all_records:
            writer.writerow([r["run_name"], r["seed"], r["episode_index"],
                             r["max_reward"], r["peak_timestep"],
                             r["num_steps"], r["final_reward"]])

    print(f"\nSaved: {combined_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")
    print(f"Video dir: {VID_DIR}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("VERIFIED EVAL RESULTS")
    print("=" * 60)
    for s in all_summaries:
        print(f"  {s['run_name']:<20s} mean={s['mean']:.4f}  median={s['median']:.4f}  "
              f"std={s['std']:.4f}  ep>0.5={s['ep_gt_0_5']}/{s['n_episodes']}")

    return all_records, all_summaries


if __name__ == "__main__":
    main()
