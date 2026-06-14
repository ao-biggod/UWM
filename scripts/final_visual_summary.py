#!/usr/bin/env python3
"""Generate complete visual summary for UWM PushT Phase A+B results.

Outputs go to outputs/final_visual_summary/
- figures/: score_summary_main.png, per_episode_scores_main.png,
            paired_delta_scratch005_vs_R1.png, keyframe_grid_seed100000.png,
            reward_curves_seed100000.png, gradient_cosine_histogram.png
- videos/: best/median/failure rollouts for scratch λ=0.05,
           seed=100000 videos for all runs, side-by-side comparison
- json/summary_main.json
- csv/per_episode_scores_main.csv

No retraining. Reads existing eval JSONs + checkpoints.
Only re-rollouts when necessary (seed=100000 keyframes/videos).
"""

import json, os, sys, argparse, csv, random, time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import imageio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "unified-world-model-main"))
sys.path.insert(0, str(ROOT / "diffusion_policy-main"))

OUT = ROOT / "outputs" / "final_visual_summary"
FIG = OUT / "figures"
VID = OUT / "videos"
JSON_DIR = OUT / "json"
CSV_DIR = OUT / "csv"

for d in [FIG, VID, JSON_DIR, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Run definitions
# ──────────────────────────────────────────────────────────────────────────────

RUNS = OrderedDict({
    "joint_lambda1": {
        "label": "joint λ=1.0",
        "eval_json": None,  # uses paired CSV
        "checkpoint": "outputs/uwm_pusht_crossattn_joint_C/main_20k/latest.pt",
        "color": "#E53935",
        "key": "joint λ=1.0",
    },
    "R1_lambda0": {
        "label": "R1 λ=0.0",
        "eval_json": "outputs/uwm_pusht_r1_loss_off/eval_det_50ep.json",
        "checkpoint": "outputs/uwm_pusht_r1_loss_off/main_20k/latest.pt",
        "color": "#2196F3",
        "key": "R1 λ=0.0",
    },
    "cont_lambda0": {
        "label": "cont λ=0.0",
        "eval_json": "outputs/uwm_pusht_r1_ft/lambda_0.00/eval_det_50ep.json",
        "checkpoint": "outputs/uwm_pusht_r1_ft/lambda_0.00/latest.pt",
        "color": "#FF9800",
        "key": "cont λ=0.0",
    },
    "ft_lambda005": {
        "label": "ft λ=0.05",
        "eval_json": "outputs/uwm_pusht_r1_ft/lambda_0.05/eval_det_50ep.json",
        "checkpoint": "outputs/uwm_pusht_r1_ft/lambda_0.05/latest.pt",
        "color": "#4CAF50",
        "key": "ft λ=0.05",
    },
    "scratch_lambda005": {
        "label": "Scratch λ=0.05",
        "eval_json": "outputs/uwm_pusht_scratch_lambda005/eval_det_50ep.json",
        "checkpoint": "outputs/uwm_pusht_scratch_lambda005/latest.pt",
        "color": "#9C27B0",
        "key": "Scratch λ=0.05",
    },
    "warmup_0_to_005": {
        "label": "Warmup 0→0.05",
        "eval_json": "outputs/uwm_pusht_warmup_step2/eval_det_10ep.json",
        "checkpoint": "outputs/uwm_pusht_warmup_step2/latest.pt",
        "color": "#607D8B",
        "key": "Warmup 0→0.05",
    },
})

# Load joint C scores from paired CSV
def load_joint_scores():
    csv_path = ROOT / "outputs/uwm_pusht_crossattn_joint_C/main_20k/paired_comparison.csv"
    scores = []
    seeds = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seeds.append(int(row["seed"]))
            scores.append(float(row["crossattn_score"]))
    # Sort by seed for consistency
    paired = sorted(zip(seeds, scores), key=lambda x: x[0])
    seeds, scores = zip(*paired)
    return list(seeds), list(scores)

def load_eval(path):
    with open(ROOT / path) as f:
        return json.load(f)

def get_scores(run_id):
    """Return (seeds, scores, stats_dict) for a run."""
    run = RUNS[run_id]
    if run_id == "joint_lambda1":
        seeds, scores = load_joint_scores()
        arr = np.array(scores)
        return seeds, scores, {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "ep_gt_0_5": int(np.sum(arr > 0.5)),
        }
    else:
        d = load_eval(run["eval_json"])
        return d["seeds"], d["scores"], {
            "mean": d["mean"],
            "median": d["median"],
            "std": d["std"],
            "ep_gt_0_5": d["ep_gt_0_5"],
        }

# Preload all data
DATA = {}
for run_id in RUNS:
    seeds, scores, stats = get_scores(run_id)
    DATA[run_id] = {"seeds": seeds, "scores": scores, "stats": stats}
    print(f"Loaded {run_id}: mean={stats['mean']:.3f}, n={len(scores)}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Score Summary Bar Chart
# ──────────────────────────────────────────────────────────────────────────────

def fig1_score_summary():
    run_ids_6 = list(RUNS.keys())
    means = [DATA[r]["stats"]["mean"] for r in run_ids_6]
    stds = [DATA[r]["stats"]["std"] for r in run_ids_6]
    labels = [RUNS[r]["key"] for r in run_ids_6]
    colors = [RUNS[r]["color"] for r in run_ids_6]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(run_ids_6))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5, alpha=0.9)

    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Mean Score (50 episodes)", fontsize=11)
    ax.set_title("Effect of Video/Dynamics Loss Weight on PushT Policy", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="threshold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # Annotate "SOTA"
    best_idx = len(run_ids_6) - 2  # scratch is second to last before warmup
    ax.annotate("SOTA", xy=(best_idx, means[best_idx] + stds[best_idx] + 0.03),
                ha="center", fontsize=9, color="#9C27B0", fontweight="bold")

    fig.tight_layout()
    path = FIG / "score_summary_main.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-episode Score Curves
# ──────────────────────────────────────────────────────────────────────────────

def fig2_per_episode_scores():
    run_ids_curve = ["joint_lambda1", "R1_lambda0", "cont_lambda0",
                     "ft_lambda005", "scratch_lambda005"]
    # Warmup is 10ep only, we handle separately

    fig, (ax_main, ax_warmup) = plt.subplots(1, 2, figsize=(16, 6),
                                              gridspec_kw={"width_ratios": [3, 1]})

    for run_id in run_ids_curve:
        scores = DATA[run_id]["scores"]
        n = len(scores)
        ax_main.plot(range(n), sorted(scores), color=RUNS[run_id]["color"],
                     label=RUNS[run_id]["key"], linewidth=1.5, alpha=0.85)

    ax_main.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="threshold")
    ax_main.set_xlabel("Episode Index (sorted by score)", fontsize=10)
    ax_main.set_ylabel("Score", fontsize=10)
    ax_main.set_title("Per-Episode Score Distribution", fontsize=12, fontweight="bold")
    ax_main.legend(fontsize=8, loc="lower right")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(-0.05, 1.05)

    # Warmup subplot
    warmup_scores = DATA["warmup_0_to_005"]["scores"]
    ax_warmup.plot(range(len(warmup_scores)), warmup_scores,
                   color=RUNS["warmup_0_to_005"]["color"], linewidth=2, marker="o", markersize=6)
    ax_warmup.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax_warmup.set_xlabel("Episode Index", fontsize=10)
    ax_warmup.set_ylabel("Score", fontsize=10)
    ax_warmup.set_title("Warmup 0→0.05 (10ep)", fontsize=11, fontweight="bold")
    ax_warmup.grid(True, alpha=0.3)
    ax_warmup.set_ylim(-0.05, 1.05)

    fig.suptitle("Per-Episode Scores — PushT UWM Cross-Attn", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIG / "per_episode_scores_main.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Paired Delta Scratch λ=0.05 vs R1
# ──────────────────────────────────────────────────────────────────────────────

def paired_ttest_rel(a, b):
    """Compute paired t-test without scipy."""
    a, b = np.array(a), np.array(b)
    d = a - b
    n = len(d)
    mean_d = np.mean(d)
    # Use ddof=1 for sample std (matching scipy default)
    std_d = np.std(d, ddof=1) if n > 1 else 1e-8
    if std_d == 0:
        return 0.0, 1.0
    t_stat = mean_d / (std_d / np.sqrt(n))
    # Two-sided p-value from t-distribution (approximate for n >= 30)
    # Use gaussian approximation
    from math import sqrt, pi, exp
    z = abs(t_stat)
    # simple Gaussian tail approximation
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + _erf_approx(z / sqrt(2))))
    return t_stat, max(0.0, min(1.0, p_val))

def _erf_approx(x):
    """Approximation of the error function."""
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    return sign * (1.0 - y * np.exp(-x * x))
    return sign * y

def fig3_paired_delta():
    # Align by seed
    r1_seeds = DATA["R1_lambda0"]["seeds"]
    r1_scores = DATA["R1_lambda0"]["scores"]
    r1_by_seed = dict(zip(r1_seeds, r1_scores))

    sc_seeds = DATA["scratch_lambda005"]["seeds"]
    sc_scores = DATA["scratch_lambda005"]["scores"]
    sc_by_seed = dict(zip(sc_seeds, sc_scores))

    common_seeds = sorted(set(r1_seeds) & set(sc_seeds))
    deltas = []
    scratch_wins = 0
    r1_wins = 0
    ties = 0
    for seed in common_seeds:
        d = sc_by_seed[seed] - r1_by_seed[seed]
        deltas.append(d)
        if d > 0:
            scratch_wins += 1
        elif d < 0:
            r1_wins += 1
        else:
            ties += 1

    deltas = np.array(deltas)
    t_stat, p_val = paired_ttest_rel([sc_by_seed[s] for s in common_seeds],
                                     [r1_by_seed[s] for s in common_seeds])

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ["#4CAF50" if d > 0 else "#E53935" for d in deltas]
    ax.bar(range(len(deltas)), deltas, color=colors, alpha=0.8, edgecolor="none", width=1.0)

    ax.axhline(y=0, color="black", linewidth=1.0)

    mean_delta = float(np.mean(deltas))
    ax.axhline(y=mean_delta, color="#9C27B0", linestyle="--", linewidth=1.5)

    # Annotations
    textstr = f"Mean Δ = {mean_delta:+.3f}\n"
    textstr += f"Scratch > R1: {scratch_wins}/{len(common_seeds)}\n"
    textstr += f"R1 > Scratch: {r1_wins}/{len(common_seeds)}\n"
    textstr += f"Ties: {ties}/{len(common_seeds)}\n"
    textstr += f"Paired t-test p = {p_val:.4f}"
    props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    ax.set_xlabel("Episode Seed Index", fontsize=10)
    ax.set_ylabel("Δ Score (Scratch λ=0.05 − R1 λ=0.0)", fontsize=10)
    ax.set_title("Paired Per-Episode Delta: Scratch λ=0.05 vs R1 λ=0.0", fontsize=12, fontweight="bold")
    ax.set_xlim(-1, len(deltas))
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = FIG / "paired_delta_scratch005_vs_R1.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Reward-over-time curves from trajectory JSONs
# ──────────────────────────────────────────────────────────────────────────────

def fig5_reward_curves():
    traj_dir = ROOT / "outputs/diag_traj"
    traj_files = {
        "R1_lambda0": traj_dir / "trajectory_R1_seed100000.json",
        "cont_lambda0": traj_dir / "trajectory_cont0_seed100000.json",
        "ft_lambda005": traj_dir / "trajectory_ft005_seed100000.json",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for run_id, fpath in traj_files.items():
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            traj = json.load(f)

        steps = [s["step"] for s in traj["steps"]]
        rewards = [s["reward"] for s in traj["steps"]]

        # Fill final reward to step 300
        final_r = rewards[-1]
        pad_steps = list(range(steps[-1] + 1, 301))
        pad_rewards = [final_r] * len(pad_steps)
        all_steps = steps + pad_steps
        all_rewards = rewards + pad_rewards

        ax.plot(all_steps, all_rewards, color=RUNS[run_id]["color"],
                label=RUNS[run_id]["key"], linewidth=1.5)

        # Peak annotation
        peak_idx = np.argmax(rewards)
        ax.scatter(steps[peak_idx], rewards[peak_idx], color=RUNS[run_id]["color"],
                   s=60, zorder=5, marker="o", edgecolors="black", linewidths=0.5)
        ax.annotate(f"peak={rewards[peak_idx]:.3f} @t={steps[peak_idx]}",
                    (steps[peak_idx], rewards[peak_idx]),
                    textcoords="offset points", xytext=(5, 10), fontsize=8,
                    color=RUNS[run_id]["color"])

        # Final annotation
        ax.scatter(steps[-1], rewards[-1], color=RUNS[run_id]["color"],
                   s=30, zorder=5, marker="s", edgecolors="black", linewidths=0.5)

    # Also load scratch and joint trajectory if available
    for run_id in ["scratch_lambda005", "joint_lambda1"]:
        traj_path = ROOT / "outputs" / "diag_traj" / f"trajectory_{run_id}_seed100000.json"
        if traj_path.exists():
            with open(traj_path) as f:
                traj = json.load(f)
            steps = [s["step"] for s in traj["steps"]]
            rewards = [s["reward"] for s in traj["steps"]]
            final_r = rewards[-1]
            pad_steps = list(range(steps[-1] + 1, 301))
            pad_rewards = [final_r] * len(pad_steps)
            all_steps = steps + pad_steps
            all_rewards = rewards + pad_rewards
            ax.plot(all_steps, all_rewards, color=RUNS[run_id]["color"],
                    label=RUNS[run_id]["key"], linewidth=1.5)
            peak_idx = np.argmax(rewards)
            ax.scatter(steps[peak_idx], rewards[peak_idx], color=RUNS[run_id]["color"],
                       s=60, zorder=5, marker="o", edgecolors="black", linewidths=0.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_ylabel("Reward", fontsize=10)
    ax.set_title("Reward over Time — seed=100000", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = FIG / "reward_curves_seed100000.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: Gradient Cosine Histogram
# ──────────────────────────────────────────────────────────────────────────────

def fig6_gradient_cosine():
    diag_path = ROOT / "outputs/diag_loss_grad_scale_main_20k.json"
    if not diag_path.exists():
        print("  WARNING: gradient diagnostic not found, skipping")
        return

    diag = json.load(open(diag_path))
    pl = diag["per_layer"]
    gs = diag["gradient_stats"]

    # Collect per-layer cosine values
    block_keys = [k for k in pl.keys() if k.startswith("blocks.")]
    block_keys.sort(key=lambda x: int(x.split(".")[1]))
    other_keys = [k for k in pl.keys() if not k.startswith("blocks.")]
    layer_names = block_keys + other_keys
    layer_cos = [pl[k]["cos"] for k in layer_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-layer cosine bar chart
    x = np.arange(len(layer_cos))
    colors = ["#E53935" if c < 0 else "#4CAF50" for c in layer_cos]
    ax1.bar(x, layer_cos, color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax1.axhline(y=gs["cosine_similarity_mean"], color="#9C27B0", linestyle="--",
                linewidth=1.5, label=f"mean={gs['cosine_similarity_mean']:.4f}")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_xlabel("Transformer Block", fontsize=10)
    ax1.set_ylabel("Cosine Similarity (action vs dynamics grad)", fontsize=10)
    ax1.set_title("Per-Layer Gradient Cosine Similarity", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: summary statistics panel
    ax2.axis("off")
    stats_text = (
        f"Gradient Conflict Summary\n"
        f"{'─' * 35}\n\n"
        f"Mean cosine similarity: {gs['cosine_similarity_mean']:.4f}\n"
        f"Median cosine similarity: {gs['cosine_similarity_median']:.4f}\n\n"
        f"cos < 0 fraction: {gs['cosine_negative_fraction']:.1%}\n"
        f"cos < 0.3 fraction: {gs['cosine_below_0_3_fraction']:.1%}\n\n"
        f"‖grad_dyn‖ / ‖grad_act‖ (median): {gs['grad_norm_ratio_median']:.2f}\n"
        f"‖grad_action‖ mean: {gs['grad_norm_action_mean']:.6f}\n"
        f"‖grad_dynamics‖ mean: {gs['grad_norm_dynamics_mean']:.6f}\n\n"
        f"Conclusion: Gradients are nearly\n"
        f"orthogonal — video/dynamics loss\n"
        f"injects noise, not useful signal."
    )
    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="gray", alpha=0.9))

    fig.suptitle("Gradient Conflict Between Action Loss and Video/Dynamics Loss", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIG / "gradient_cosine_histogram.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary JSON + CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_summary_files():
    # JSON
    summary = {
        "main_claim": "Small video/dynamics loss (λ=0.05) improves PushT policy as a regularizer, while large video loss (λ=1.0) hurts policy via gradient conflict.",
        "runs": {
            "joint_lambda1": {"mean": DATA["joint_lambda1"]["stats"]["mean"],
                              "median": DATA["joint_lambda1"]["stats"]["median"],
                              "std": DATA["joint_lambda1"]["stats"]["std"],
                              "ep_gt_0_5": DATA["joint_lambda1"]["stats"]["ep_gt_0_5"],
                              "setting": "Cross-Attn joint, λ=1.0"},
            "R1_lambda0": {"mean": DATA["R1_lambda0"]["stats"]["mean"],
                           "median": DATA["R1_lambda0"]["stats"]["median"],
                           "std": DATA["R1_lambda0"]["stats"]["std"],
                           "ep_gt_0_5": DATA["R1_lambda0"]["stats"]["ep_gt_0_5"],
                           "setting": "Cross-Attn joint, λ=0.0 (action-only)"},
            "cont_lambda0": {"mean": DATA["cont_lambda0"]["stats"]["mean"],
                             "median": DATA["cont_lambda0"]["stats"]["median"],
                             "std": DATA["cont_lambda0"]["stats"]["std"],
                             "ep_gt_0_5": DATA["cont_lambda0"]["stats"]["ep_gt_0_5"],
                             "setting": "R1 +5k steps, λ=0.0 (pure action continue)"},
            "ft_lambda005": {"mean": DATA["ft_lambda005"]["stats"]["mean"],
                             "median": DATA["ft_lambda005"]["stats"]["median"],
                             "std": DATA["ft_lambda005"]["stats"]["std"],
                             "ep_gt_0_5": DATA["ft_lambda005"]["stats"]["ep_gt_0_5"],
                             "setting": "R1 +5k steps, λ=0.05 (small video regularizer)"},
            "scratch_lambda005": {"mean": DATA["scratch_lambda005"]["stats"]["mean"],
                                  "median": DATA["scratch_lambda005"]["stats"]["median"],
                                  "std": DATA["scratch_lambda005"]["stats"]["std"],
                                  "ep_gt_0_5": DATA["scratch_lambda005"]["stats"]["ep_gt_0_5"],
                                  "setting": "Cross-Attn joint, λ=0.05 from scratch (SOTA)"},
            "warmup_0_to_005": {"mean": DATA["warmup_0_to_005"]["stats"]["mean"],
                                "median": DATA["warmup_0_to_005"]["stats"]["median"],
                                "std": DATA["warmup_0_to_005"]["stats"]["std"],
                                "ep_gt_0_5": DATA["warmup_0_to_005"]["stats"]["ep_gt_0_5"],
                                "setting": "0-10k λ=0, 10-20k λ=0.05 (abrupt switch)"},
        },
        "interpretation": [
            "λ=1.0 joint training hurts policy (0.354) vs action-only (0.534) due to gradient conflict (cos≈0.016).",
            "R1 λ=0.0 (0.534) is a strong anchor but pure action continuation degrades to 0.450.",
            "λ=0.05 finetune (0.537) prevents degradation — small video loss as regularizer.",
            "Scratch λ=0.05 (0.614) is the new SOTA, +0.080 over R1, +0.260 over joint λ=1.0.",
            "Abrupt warmup 0→0.05 fails (0.241) — introducing dynamics loss mid-training is destabilizing.",
            "Gradient cos ≈ 0.016 explains joint λ=1.0 failure: action/dynamics gradients are nearly orthogonal.",
        ],
    }

    with open(JSON_DIR / "summary_main.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV: per_episode_scores for all 6 runs
    csv_path = CSV_DIR / "per_episode_scores_main.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["ep_index"]
        for run_id in RUNS:
            header.append(f"{run_id}_seed")
            header.append(f"{run_id}_score")
        writer.writerow(header)

        max_n = max(len(DATA[r]["seeds"]) for r in RUNS)
        for i in range(max_n):
            row = [i]
            for run_id in RUNS:
                if i < len(DATA[run_id]["seeds"]):
                    row.append(DATA[run_id]["seeds"][i])
                    row.append(DATA[run_id]["scores"][i])
                else:
                    row.append("")
                    row.append("")
            writer.writerow(row)

    print(f"Saved: {JSON_DIR / 'summary_main.json'}")
    print(f"Saved: {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main: Run all figure generators
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true",
                        help="Only generate figures (no rollouts/videos)")
    parser.add_argument("--videos-only", action="store_true",
                        help="Only generate videos (no figures)")
    args = parser.parse_args()

    if not args.videos_only:
        print("=" * 60)
        print("Generating Figures...")
        print("=" * 60)
        try:
            fig1_score_summary()
            fig2_per_episode_scores()
            fig3_paired_delta()
            fig5_reward_curves()
            fig6_gradient_cosine()
            save_summary_files()
        except Exception as e:
            print(f"ERROR in figures: {e}")
            import traceback
            traceback.print_exc()

    if not args.figures_only:
        print()
        print("=" * 60)
        print("Recording Rollout Videos + Keyframe Grid (Figure 4)...")
        print("=" * 60)
        record_all_videos()
        # Figure 4 runs after videos since it reuses rendered frames where possible
        try:
            fig4_keyframe_grid()
        except Exception as e:
            print(f"ERROR in keyframe grid: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("File Manifest")
    print("=" * 60)
    print_final_manifest()


def print_final_manifest():
    figs = sorted(FIG.glob("*.png"))
    vids = sorted(VID.glob("*.mp4"))
    jsons = sorted(JSON_DIR.glob("*.json"))
    csvs = sorted(CSV_DIR.glob("*.csv"))

    print("\nFigures:")
    for p in figs: print(f"  {p}")
    print("\nVideos:")
    for p in vids: print(f"  {p}")
    print("\nJSON:")
    for p in jsons: print(f"  {p}")
    print("\nCSV:")
    for p in csvs: print(f"  {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Video / Rollout Section
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_cudnn_deterministic():
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

def build_model(device, ckpt):
    import torch
    from models.uwm.uwm import UnifiedWorldModel
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

def rollout_one(ckpt_path, seed, device):
    """Run deterministic rollout from checkpoint, return (frames, rewards, steps, score)."""
    import torch
    import numpy as np
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    ckpt = torch.load(ckpt_path, map_location="cpu")
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

    frames = []
    rewards = []
    step = 0

    while step < 300:
        frame = inner_env.render(mode="rgb_array")
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

        obs_model = {"image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
                     "agent_pos": ap_norm.unsqueeze(0)}

        with torch.no_grad():
            action_norm = model.sample(obs_model)[0]

        action_raw = (action_norm
                      * torch.tensor(action_scale, device=device).float()
                      + torch.tensor(action_offset, device=device).float())
        action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

        obs, reward, done, _ = env.step(action_raw_np[:8])
        rewards.append(float(reward))
        if np.all(done):
            break
        step += 1

    frame = inner_env.render(mode="rgb_array")
    frames.append(frame)
    env.close()

    score = float(max(rewards)) if rewards else 0.0
    return frames, rewards, step + 1, score


def record_all_videos():
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_cudnn_deterministic()

    # ── 1. Scratch λ=0.05: best / median / failure ──
    run_id = "scratch_lambda005"
    ckpt_path = str(ROOT / RUNS[run_id]["checkpoint"])
    seeds = DATA[run_id]["seeds"]
    scores = DATA[run_id]["scores"]

    # Find best, median, failure
    sorted_idx = np.argsort(scores)
    best_idx = sorted_idx[-1]
    median_idx = sorted_idx[len(sorted_idx) // 2]
    failure_idx = sorted_idx[0]

    targets = [
        ("best", seeds[best_idx], scores[best_idx]),
        ("median", seeds[median_idx], scores[median_idx]),
        ("failure", seeds[failure_idx], scores[failure_idx]),
    ]

    for label, seed, score in targets:
        vid_path = VID / f"scratch_lambda005_{label}_seed{seed}_score{score:.3f}.mp4"
        if vid_path.exists():
            print(f"  SKIP (exists): {vid_path.name}")
            continue
        print(f"  Recording: scratch_lambda005 {label} seed={seed} score={score:.3f}")
        frames, rewards, steps, sc = rollout_one(ckpt_path, seed, device)
        imageio.mimsave(str(vid_path), frames, fps=10)
        sz = os.path.getsize(str(vid_path)) / 1024 / 1024
        print(f"    saved ({sz:.1f} MB), steps={steps}")

    # ── 2. Seed=100000 videos for all runs ──
    seed_100000 = 100000
    for run_id, run in RUNS.items():
        vid_path = VID / f"{run_id}_seed100000.mp4"
        if vid_path.exists():
            print(f"  SKIP (exists): {vid_path.name}")
            continue
        ckpt_path = str(ROOT / run["checkpoint"])
        if not os.path.exists(ckpt_path):
            print(f"  MISSING checkpoint: {ckpt_path}")
            continue
        score_in_eval = None
        if seed_100000 in DATA[run_id]["seeds"]:
            idx = DATA[run_id]["seeds"].index(seed_100000)
            score_in_eval = DATA[run_id]["scores"][idx]

        print(f"  Recording: {run_id} seed={seed_100000}" +
              (f" score={score_in_eval:.3f}" if score_in_eval is not None else ""))
        frames, rewards, steps, sc = rollout_one(ckpt_path, seed_100000, device)
        imageio.mimsave(str(vid_path), frames, fps=10)
        sz = os.path.getsize(str(vid_path)) / 1024 / 1024
        print(f"    saved ({sz:.1f} MB), steps={steps}, max_reward={sc:.3f}")

    # ── 3. Side-by-side comparison video for seed=100000 ──
    _make_comparison_video()


def _make_comparison_video():
    """Combine 4 models into side-by-side grid for seed=100000."""
    run_ids = ["joint_lambda1", "R1_lambda0", "ft_lambda005", "scratch_lambda005"]

    # Collect frames
    vid_data = []
    for run_id in run_ids:
        vid_path = VID / f"{run_id}_seed100000.mp4"
        if not vid_path.exists():
            print(f"  Comparison: missing {vid_path.name}, skipping side-by-side")
            return
        frames = imageio.mimread(str(vid_path))
        vid_data.append((run_id, frames, RUNS[run_id]["key"]))

    # Pad to same length
    max_len = max(len(f) for _, f, _ in vid_data)
    for i, (rid, fr, lab) in enumerate(vid_data):
        if len(fr) < max_len:
            fr.extend([fr[-1]] * (max_len - len(fr)))
            vid_data[i] = (rid, fr, lab)

    # Grid: 2x2
    output_frames = []
    for t in range(max_len):
        # Build 2x2 grid
        rows = []
        for row in range(2):
            cols = []
            for col in range(2):
                idx = row * 2 + col
                if idx < len(vid_data):
                    f = vid_data[idx][1][t]
                    # Add label at top
                    f_labeled = f.copy()
                    # Add colored bar at top
                    color = RUNS[vid_data[idx][0]]["color"]
                    color_rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                    # Convert to BGR for imageio
                    bar_h = 25
                    bar = np.zeros((bar_h, f.shape[1], 3), dtype=np.uint8)
                    bar[:, :, 0] = color_rgb[2]
                    bar[:, :, 1] = color_rgb[1]
                    bar[:, :, 2] = color_rgb[0]
                    f_labeled = np.vstack([bar, f])
                    cols.append(f_labeled)
                else:
                    cols.append(np.zeros_like(cols[0]))
            rows.append(np.hstack(cols))
        output_frames.append(np.vstack(rows))

    out_path = VID / "comparison_main_seed100000_side_by_side.mp4"
    imageio.mimsave(str(out_path), output_frames, fps=10)
    sz = os.path.getsize(str(out_path)) / 1024 / 1024
    print(f"  Saved comparison video ({sz:.1f} MB): {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Keyframe Grid (Figure 4) — requires running rollout to specific steps
# ──────────────────────────────────────────────────────────────────────────────

def fig4_keyframe_grid():
    """Render keyframes for seed=100000 across 5 runs."""
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_cudnn_deterministic()

    run_ids = ["joint_lambda1", "R1_lambda0", "cont_lambda0", "ft_lambda005", "scratch_lambda005"]
    seed = 100000

    # First, run quick rollouts to identify key timesteps
    keyframes_meta = {}
    for run_id in run_ids:
        ckpt_path = str(ROOT / RUNS[run_id]["checkpoint"])
        print(f"  Keyframe rollout: {run_id}")
        frames, rewards, steps, score = rollout_one(ckpt_path, seed, device)

        peak_idx = int(np.argmax(rewards))
        peak_step = peak_idx  # step index
        mid_step = min(peak_step + 50, len(rewards) - 1)
        final_step = len(rewards) - 1

        keyframes_meta[run_id] = {
            "frames": frames,
            "rewards": rewards,
            "score": score,
            "t0": 0,
            "peak": peak_step,
            "mid": mid_step,
            "final": final_step,
        }
        print(f"    score={score:.3f} t0=0 peak={peak_step} mid={mid_step} final={final_step}")

    # Build grid: 5 rows x 4 cols
    n_rows = len(run_ids)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    col_labels = ["t=0", "t=peak reward", "t=peak+50 / mid", "t=final"]

    for row, run_id in enumerate(run_ids):
        meta = keyframes_meta[run_id]
        run_label = RUNS[run_id]["key"]
        score = meta["score"]
        axes[row, 0].set_ylabel(f"{run_label}\nscore={score:.3f}", fontsize=9, fontweight="bold",
                                rotation=0, ha="right", va="center", labelpad=60)

        for col, key in enumerate(["t0", "peak", "mid", "final"]):
            ax = axes[row, col]
            step_idx = meta[key]
            frame = meta["frames"][step_idx]
            ax.imshow(frame)
            ax.set_title(f"{col_labels[col]}\n(step {step_idx})", fontsize=8)
            ax.axis("off")

    for col in range(n_cols):
        axes[0, col].set_title(col_labels[col], fontsize=10, fontweight="bold")

    fig.suptitle(f"Keyframe Comparison — seed={seed} (Cross-Attn UWM PushT)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIG / "keyframe_grid_seed100000.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
