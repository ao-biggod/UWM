#!/usr/bin/env python3
"""Steps 1-4: Statistical analysis, plots, typical episodes, rollout videos.

Reads per_episode_scores.csv from deterministic paired eval and produces:
  Step 1: Statistical significance tests
  Step 2: 3 plots (scatter, delta bar, histogram)
  Step 3: Typical episode tables
  Step 4: Rollout videos for top/bottom cases
"""
import argparse, csv, json, os, sys, time, random
from pathlib import Path
from collections import deque
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic utils (same as deterministic_paired_eval.py)
# ──────────────────────────────────────────────────────────────────────────────

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


def make_env(seed):
    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
    env.seed(seed)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Model builders (same as deterministic_paired_eval.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_model(device, model_type, clip_sample=False):
    if model_type == "dp":
        from models.dp import ImageDiffusionPolicy, ImageObservationEncoder
        from models.dp import TransformerNoisePredictionNet
        from functools import partial
        shape_meta = {"obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                               "agent_pos": {"shape": [2], "type": "low_dim"}},
                       "action": {"shape": [2]}}
        obs_encoder = ImageObservationEncoder(
            shape_meta=shape_meta, num_frames=2, embed_dim=768,
            resize_shape=None, crop_shape=None, random_crop=False,
            color_jitter=None, imagenet_norm=False,
            pretrained_weights=None, use_low_dim=True, use_language=False)
        model = ImageDiffusionPolicy(
            action_len=16, action_dim=2, obs_encoder=obs_encoder,
            noise_pred_net=partial(TransformerNoisePredictionNet, input_len=16, input_dim=2,
                                   timestep_embed_dim=256, embed_dim=768, depth=12,
                                   num_heads=12, mlp_ratio=4, qkv_bias=True),
            num_train_steps=100, num_inference_steps=10,
            beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample)
    else:  # uwm
        from models.uwm import UnifiedWorldModel
        from models.uwm.obs_encoder import UWMObservationEncoder
        sm = {"obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                      "agent_pos": {"shape": [2], "type": "low_dim"}},
              "action": {"shape": [2]}}
        oe = UWMObservationEncoder(
            shape_meta=sm, num_frames=2, embed_dim=768,
            resize_shape=None, crop_shape=None, random_crop=False,
            color_jitter=None, imagenet_norm=False,
            vision_backbone="resnet", use_low_dim=True, use_language=False)
        model = UnifiedWorldModel(
            action_len=16, action_dim=2, obs_encoder=oe,
            embed_dim=768, timestep_embed_dim=512,
            latent_patch_shape=[2, 4, 4], depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, num_registers=8,
            num_train_steps=100, num_inference_steps=10,
            beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample)
    return model.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Rollout recording
# ──────────────────────────────────────────────────────────────────────────────

def rollout_with_frames(model, device, action_scale, action_offset,
                        ap_scale, ap_offset, seed, model_type):
    """Run a deterministic episode and collect rendered frames. Returns frames + max_reward."""
    seed_everything(seed)
    env = make_env(seed)
    obs = env.reset()
    rewards = []; done = False; step = 0
    frames = []

    # Render initial frame
    frame = env.env.render(mode="rgb_array")
    frames.append(frame)

    while not done and step < 300:
        img = obs["image"]
        agent_pos = obs["agent_pos"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(agent_pos, np.ndarray):
            agent_pos = torch.from_numpy(agent_pos)

        img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        ap = agent_pos.float().to(device)
        # norm_agent_pos=True for config C
        ap = (ap - torch.tensor(ap_offset, device=device).float()) / torch.tensor(ap_scale, device=device).float()

        obs_model = {"image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
                     "agent_pos": ap.unsqueeze(0)}

        with torch.no_grad():
            action_norm = model.sample(obs_model)[0]

        action_raw = (action_norm * torch.tensor(action_scale, device=device).float()
                      + torch.tensor(action_offset, device=device).float())
        action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
        exec_actions = action_raw_np[:8]

        obs, reward, done, info = env.step(exec_actions)
        rewards.append(float(reward))
        done = bool(np.all(done))
        step += 1

        frame = env.env.render(mode="rgb_array")
        frames.append(frame)

    max_r = float(max(rewards)) if rewards else 0.0
    return frames, max_r


def save_video(frames, path, fps=24):
    """Save frames as mp4 video using imageio."""
    import imageio

    if not frames:
        print(f"  WARNING: no frames for {path}")
        return

    writer = imageio.get_writer(path, fps=fps, format="FFMPEG", codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Statistical tests
# ──────────────────────────────────────────────────────────────────────────────

def step1_tests(dp_scores, uwm_scores, output_dir):
    print("=" * 60)
    print("Step 1: Statistical Significance Tests")
    print("=" * 60)

    delta = np.array(uwm_scores) - np.array(dp_scores)
    n = len(delta)
    nontie_delta = delta[delta != 0]
    u_greater = (delta > 0).sum()
    d_greater = (delta < 0).sum()
    tie = (delta == 0).sum()

    results = []

    # 1. Paired t-test
    t_stat, t_p = stats.ttest_rel(uwm_scores, dp_scores)
    results.append(("Paired t-test", f"t={t_stat:.4f}", f"{t_p:.4f}",
                    "significant" if t_p < 0.05 else "not significant"))

    # 2. Wilcoxon signed-rank test
    w_stat, w_p = stats.wilcoxon(uwm_scores, dp_scores, zero_method="zsplit")
    results.append(("Wilcoxon signed-rank", f"W={w_stat:.1f}", f"{w_p:.4f}",
                    "significant" if w_p < 0.05 else "not significant"))

    # 3. Sign test (binomial test on non-tie cases)
    if len(nontie_delta) > 0:
        n_success = max(u_greater, d_greater)
        sign_p = stats.binomtest(n_success, len(nontie_delta), p=0.5, alternative="two-sided").pvalue
    else:
        sign_p = 1.0
    results.append(("Sign test (binomial)", f"wins={u_greater}/{d_greater} (U/D)", f"{sign_p:.4f}",
                    "significant" if sign_p < 0.05 else "not significant"))

    # 4. Bootstrap 95% CI for delta mean
    rng = np.random.RandomState(42)
    n_boot = 10000
    boot_means = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means.append(delta[idx].mean())
    boot_means = np.sort(boot_means)
    delta_mean_95ci = (boot_means[250], boot_means[9750])
    results.append(("Bootstrap Δ mean 95% CI", f"{delta_mean_95ci[0]:.4f} to {delta_mean_95ci[1]:.4f}", "-", "-"))

    # 5. Bootstrap 95% CI for delta median
    boot_meds = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_meds.append(np.median(delta[idx]))
    boot_meds = np.sort(boot_meds)
    delta_median_95ci = (boot_meds[250], boot_meds[9750])
    results.append(("Bootstrap Δ median 95% CI", f"{delta_median_95ci[0]:.4f} to {delta_median_95ci[1]:.4f}", "-", "-"))

    # Print table
    print(f"\n{'Test':<30} {'Statistic':<25} {'p-value':>10} {'Interpretation':<20}")
    print("-" * 85)
    for name, stat, pval, interp in results:
        print(f"  {name:<28} {stat:<25} {pval:>10} {interp:<20}")

    print(f"\n  delta_mean_95ci:  [{delta_mean_95ci[0]:.4f}, {delta_mean_95ci[1]:.4f}]")
    print(f"  delta_median_95ci: [{delta_median_95ci[0]:.4f}, {delta_median_95ci[1]:.4f}]")

    # Save
    tests_data = {
        "paired_t_test": {"statistic": float(t_stat), "p_value": float(t_p)},
        "wilcoxon": {"statistic": float(w_stat), "p_value": float(w_p)},
        "sign_test_binomial": {"uwm_wins": int(u_greater), "dp_wins": int(d_greater),
                                "ties": int(tie), "p_value": float(sign_p)},
        "delta_mean_95ci": [float(delta_mean_95ci[0]), float(delta_mean_95ci[1])],
        "delta_median_95ci": [float(delta_median_95ci[0]), float(delta_median_95ci[1])],
    }
    with open(os.path.join(output_dir, "statistical_tests.json"), "w") as f:
        json.dump(tests_data, f, indent=2)

    return delta


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Plots
# ──────────────────────────────────────────────────────────────────────────────

def step2_plots(dp_scores, uwm_scores, delta, output_dir):
    print("\n" + "=" * 60)
    print("Step 2: Generating plots")
    print("=" * 60)

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: DP score vs UWM score scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(dp_scores, uwm_scores, alpha=0.6, s=40, c="steelblue", edgecolors="white", linewidth=0.5)
    lims = [-0.05, 1.05]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="y=x (equality)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("DP-only C score", fontsize=12)
    ax.set_ylabel("UWM joint C score", fontsize=12)
    ax.set_title("Per-episode scores: UWM vs DP (paired)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "scatter_uwm_vs_dp.png"), dpi=150)
    plt.close(fig)
    print("  Saved: scatter_uwm_vs_dp.png")

    # Plot 2: Per-episode delta bar plot
    fig, ax = plt.subplots(figsize=(18, 5))
    colors = ["#d62728" if d < 0 else "#2ca02c" if d > 0 else "#7f7f7f" for d in delta]
    ax.bar(range(len(delta)), delta, color=colors, edgecolor="none", width=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Episode index", fontsize=12)
    ax.set_ylabel("UWM score - DP score", fontsize=12)
    ax.set_title("Per-episode delta (UWM - DP)", fontsize=13)
    ax.set_xlim(-1, len(delta))
    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label=f"UWM > DP ({int((delta > 0).sum())})"),
        Patch(facecolor="#d62728", label=f"DP > UWM ({int((delta < 0).sum())})"),
        Patch(facecolor="#7f7f7f", label=f"tie ({int((delta == 0).sum())})"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "delta_bar.png"), dpi=150)
    plt.close(fig)
    print("  Saved: delta_bar.png")

    # Plot 3: Score distribution histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 26)
    ax.hist(dp_scores, bins=bins, alpha=0.55, label=f"DP-only C (mean={np.mean(dp_scores):.3f})",
            color="#d62728", edgecolor="white")
    ax.hist(uwm_scores, bins=bins, alpha=0.55, label=f"UWM joint C (mean={np.mean(uwm_scores):.3f})",
            color="#1f77b4", edgecolor="white")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="ep>0.5 threshold")
    ax.set_xlabel("Max reward (score)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score distribution: UWM vs DP (50 episodes)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "score_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved: score_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Typical episodes
# ──────────────────────────────────────────────────────────────────────────────

def step3_typical_episodes(dp_scores, uwm_scores, delta, seeds, output_dir):
    print("\n" + "=" * 60)
    print("Step 3: Typical Episodes")
    print("=" * 60)

    records = []
    for i, (s, ds, us, d) in enumerate(zip(seeds, dp_scores, uwm_scores, delta)):
        records.append({"ep": i, "seed": s, "dp_score": ds, "uwm_score": us, "delta": d})

    # Sort by delta (descending for UWM wins, ascending for DP wins)
    sorted_by_delta = sorted(records, key=lambda r: r["delta"], reverse=True)
    nonzero_records = [r for r in records if r["delta"] != 0]

    # UWM top wins (positive delta)
    uwm_top = [r for r in sorted_by_delta if r["delta"] > 0][:5]
    # DP top wins (negative delta)
    dp_top = [r for r in sorted(sorted_by_delta, key=lambda r: r["delta"]) if r["delta"] < 0][:5]
    # Both fail (tie at low score)
    both_fail = [r for r in records if r["delta"] == 0 and r["dp_score"] < 0.1][:5]
    # Both succeed (tie at high score)
    both_succeed = [r for r in records if r["delta"] == 0 and r["dp_score"] > 0.9][:5]

    def print_category(name, rows):
        print(f"\n  --- {name} ---")
        print(f"  {'ep':>4} {'seed':>8} {'DP score':>10} {'UWM score':>11} {'delta':>10}")
        print(f"  {'-'*46}")
        for r in rows:
            print(f"  {r['ep']:4d} {r['seed']:8d} {r['dp_score']:10.4f} {r['uwm_score']:11.4f} {r['delta']:+10.4f}")

    print_category("UWM top-5 wins (UWM >> DP)", uwm_top)
    print_category("DP top-5 wins (DP >> UWM)", dp_top)
    print_category("Both fail (tie, score near 0)", both_fail)
    print_category("Both succeed (tie, score near 1)", both_succeed)

    typical = {"uwm_top_wins": uwm_top, "dp_top_wins": dp_top,
               "both_fail": both_fail, "both_succeed": both_succeed}
    with open(os.path.join(output_dir, "typical_episodes.json"), "w") as f:
        json.dump(typical, f, indent=2)

    return uwm_top, dp_top, both_fail, both_succeed


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Rollout videos
# ──────────────────────────────────────────────────────────────────────────────

def step4_videos(dp_ckpt_path, uwm_ckpt_path, device, dp_action_scale, dp_action_offset,
                 dp_ap_scale, dp_ap_offset, uwm_action_scale, uwm_action_offset,
                 uwm_ap_scale, uwm_ap_offset, uwm_top, dp_top, both_fail, both_succeed, output_dir):
    print("\n" + "=" * 60)
    print("Step 4: Generating Rollout Videos")
    print("=" * 60)

    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    # Load both models once
    set_cudnn_deterministic()
    print("  Loading DP-only model...", flush=True)
    dp_ckpt = torch.load(dp_ckpt_path, map_location=device)
    dp_model = build_model(device, "dp", clip_sample=False)
    dp_model.load_state_dict(dp_ckpt["model"])
    dp_model.eval()

    print("  Loading UWM joint model...", flush=True)
    uwm_ckpt = torch.load(uwm_ckpt_path, map_location=device)
    uwm_model = build_model(device, "uwm", clip_sample=False)
    uwm_model.load_state_dict(uwm_ckpt["model"], strict=False)
    uwm_model.eval()

    # Episodes to record
    cases = []
    for r in uwm_top[:3]:
        cases.append(("uwm_top_win", r, uwm_top))
    for r in dp_top[:3]:
        cases.append(("dp_top_win", r, dp_top))
    for r in both_fail[:3]:
        cases.append(("both_fail", r, both_fail))
    for r in both_succeed[:3]:
        cases.append(("both_succeed", r, both_succeed))

    for category, record, group in cases:
        seed = record["seed"]
        ep = record["ep"]

        # UWM rollout
        print(f"  [{category}] seed={seed} ep={ep}: UWM rollout...", flush=True)
        uwm_frames, uwm_score = rollout_with_frames(
            uwm_model, device, uwm_action_scale, uwm_action_offset,
            uwm_ap_scale, uwm_ap_offset, seed, "uwm")
        save_video(uwm_frames, os.path.join(videos_dir, f"ep{ep:03d}_seed{seed:06d}_uwm.mp4"))

        # DP rollout
        print(f"  [{category}] seed={seed} ep={ep}: DP rollout...", flush=True)
        dp_frames, dp_score = rollout_with_frames(
            dp_model, device, dp_action_scale, dp_action_offset,
            dp_ap_scale, dp_ap_offset, seed, "dp")
        save_video(dp_frames, os.path.join(videos_dir, f"ep{ep:03d}_seed{seed:06d}_dp.mp4"))

    print(f"\n  Videos saved to: {videos_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default="outputs/deterministic_paired_eval/seeds_100000_100049/per_episode_scores.csv")
    parser.add_argument("--dp-ckpt", type=str,
                        default="outputs/dp_pusht/run_20k_bs64/latest.pt")
    parser.add_argument("--uwm-ckpt", type=str,
                        default="artifacts_keep/uwm_20k/checkpoint_20k_latest.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/deterministic_paired_eval/seeds_100000_100049")
    parser.add_argument("--skip-videos", action="store_true",
                        help="Skip video generation (slow)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load CSV
    seeds = []
    dp_scores = []
    uwm_scores = []
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seeds.append(int(row["seed"]))
            dp_scores.append(float(row["dp_score"]))
            uwm_scores.append(float(row["uwm_score"]))

    print(f"Loaded {len(seeds)} episodes from {args.csv}")

    # Step 1
    delta = step1_tests(dp_scores, uwm_scores, args.output_dir)

    # Step 2
    step2_plots(dp_scores, uwm_scores, delta, args.output_dir)

    # Step 3
    uwm_top, dp_top, both_fail, both_succeed = step3_typical_episodes(
        dp_scores, uwm_scores, delta, seeds, args.output_dir)

    # Step 4
    if not args.skip_videos:
        # Load normalizer params from checkpoints
        def get_dp_normalizers(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            an = ckpt["action_normalizer"]
            action_scale = np.array(an["scale"]); action_offset = np.array(an["offset"])
            import zarr
            z = zarr.open("diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr", "r")
            ep_ends = z["meta/episode_ends"][:]
            state = z["data/state"][:ep_ends[89], :2]
            ap_scale = (state.max(axis=0) - state.min(axis=0)) / 2.0
            ap_offset = (state.max(axis=0) + state.min(axis=0)) / 2.0
            return action_scale, action_offset, ap_scale, ap_offset

        def get_uwm_normalizers(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            an = ckpt["action_normalizer"]
            action_scale = np.array(an["scale"]); action_offset = np.array(an["offset"])
            ln = ckpt["lowdim_normalizer"]["agent_pos"]
            ap_scale = np.array(ln["scale"]); ap_offset = np.array(ln["offset"])
            return action_scale, action_offset, ap_scale, ap_offset

        dp_as, dp_ao, dp_aps, dp_apo = get_dp_normalizers(args.dp_ckpt)
        uwm_as, uwm_ao, uwm_aps, uwm_apo = get_uwm_normalizers(args.uwm_ckpt)

        step4_videos(args.dp_ckpt, args.uwm_ckpt, device,
                     dp_as, dp_ao, dp_aps, dp_apo,
                     uwm_as, uwm_ao, uwm_aps, uwm_apo,
                     uwm_top, dp_top, both_fail, both_succeed, args.output_dir)

    print("\n" + "=" * 60)
    print("Steps 1-4 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
