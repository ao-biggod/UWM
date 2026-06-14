#!/usr/bin/env python3
"""Full consistency audit of all videos in outputs/final_visual_summary/videos/"""

import os, re, json, csv, sys, time, datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
VID_DIR = ROOT / "outputs" / "final_visual_summary" / "videos"
FIG_DIR = ROOT / "outputs" / "final_visual_summary" / "figures"
JSON_DIR = ROOT / "outputs" / "final_visual_summary" / "json"
FIG_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# ── Load all eval JSONs ──────────────────────────────────────────
EVAL_DB = {}

# Phase A: joint C scores from paired CSV
ca_joint_scores = {}
paired_csv = ROOT / "outputs/uwm_pusht_crossattn_joint_C/main_20k/paired_comparison.csv"
if paired_csv.exists():
    with open(paired_csv) as f:
        for row in csv.DictReader(f):
            ca_joint_scores[int(row["seed"])] = float(row["crossattn_score"])

# Phase B: eval JSONs
eval_paths = {
    "joint_lambda1": (ca_joint_scores, "outputs/uwm_pusht_crossattn_joint_C/main_20k/latest.pt"),
    "R1_lambda0": ("outputs/uwm_pusht_r1_loss_off/eval_det_50ep.json", None),
    "cont_lambda0": ("outputs/uwm_pusht_r1_ft/lambda_0.00/eval_det_50ep.json", None),
    "ft_lambda005": ("outputs/uwm_pusht_r1_ft/lambda_0.05/eval_det_50ep.json", None),
    "scratch_lambda005": ("outputs/uwm_pusht_scratch_lambda005/eval_det_50ep.json", None),
    "warmup_0_to_005": ("outputs/uwm_pusht_warmup_step2/eval_det_10ep.json", None),
}

for run_id, (path_or_dict, _) in eval_paths.items():
    if isinstance(path_or_dict, dict):
        # ca_joint_scores dict
        EVAL_DB[run_id] = {
            "seeds": list(path_or_dict.keys()),
            "scores": list(path_or_dict.values()),
        }
    elif os.path.exists(ROOT / path_or_dict):
        d = json.load(open(ROOT / path_or_dict))
        EVAL_DB[run_id] = {"seeds": d["seeds"], "scores": d["scores"]}
    else:
        print(f"WARNING: eval JSON not found: {path_or_dict}")

# ── Parse video filenames ────────────────────────────────────────
def parse_filename(fname):
    """Parse video filename into (run_id, seed, score_in_name)."""
    base = fname.replace(".mp4", "")

    # Pattern 1: scratch_lambda005_best_seed100049_score1.000
    m = re.match(r"scratch_lambda005_(best|median|failure)_seed(\d+)_score([\d.]+)", base)
    if m:
        return "scratch_lambda005", int(m.group(2)), float(m.group(3)), m.group(1)

    # Pattern 2: joint_lambda1_seed100000 (no score)
    m = re.match(r"(\w+_lambda[\d.]+)_seed(\d+)", base)
    if m:
        return m.group(1), int(m.group(2)), None, None

    # Pattern 3: Old names like 01_joint_lambda1_seed100000_score0390
    m = re.match(r"\d+_(\w+_lambda[\d.]+)_seed(\d+)_score([\d.]+)", base)
    if m:
        return m.group(1), int(m.group(2)), float(m.group(3)), None

    # Pattern 4: comparison_main_seed100000_side_by_side
    if "comparison" in base or "side_by_side" in base:
        return "comparison", 100000, None, None

    return None, None, None, None


videos = []
for fpath in sorted(VID_DIR.glob("*.mp4")):
    run_id, seed, score_name, variant = parse_filename(fpath.name)

    # Check against eval DB
    eval_score = None
    eval_match = False
    if run_id and run_id in EVAL_DB and seed is not None:
        db = EVAL_DB[run_id]
        if seed in db["seeds"]:
            idx = db["seeds"].index(seed)
            eval_score = db["scores"][idx]
            eval_match = True

    # Read video metadata
    try:
        vid = imageio.mimread(str(fpath))
        n_frames = len(vid)
        sz_mb = os.path.getsize(str(fpath)) / 1024 / 1024
    except Exception:
        n_frames = 0
        sz_mb = 0

    videos.append({
        "filename": fpath.name,
        "run_id": run_id,
        "seed": seed,
        "variant": variant,
        "score_in_name": score_name,
        "score_in_eval": eval_score,
        "eval_match": eval_match,
        "n_frames": n_frames,
        "size_mb": sz_mb,
        "path": str(fpath),
    })

# ── Task 1: Print audit table ────────────────────────────────────
print("=" * 100)
print("TASK 1: Video Inventory")
print("=" * 100)
print(f"{'filename':<55s} {'run':<22s} {'seed':>7s} {'name_score':>10s} {'eval_score':>10s} {'match':>6s} {'frames':>6s} {'MB':>5s}")
print("-" * 100)
for v in videos:
    run_str = v["run_id"] or "???"
    seed_str = str(v["seed"]) if v["seed"] is not None else "???"
    name_sc = f"{v['score_in_name']:.3f}" if v["score_in_name"] is not None else "N/A"
    eval_sc = f"{v['score_in_eval']:.3f}" if v["score_in_eval"] is not None else "N/A"

    # Match check
    if v["score_in_name"] is not None and v["score_in_eval"] is not None:
        diff = abs(v["score_in_name"] - v["score_in_eval"])
        if diff < 0.001:
            match_str = "OK"
        elif diff < 0.01:
            match_str = f"±{diff:.4f}"
        else:
            match_str = "MISMATCH"
    elif v["score_in_name"] is not None and v["score_in_eval"] is None:
        match_str = "NO_EVAL"
    else:
        match_str = "N/A"

    print(f"{v['filename']:<55s} {run_str:<22s} {seed_str:>7s} {name_sc:>10s} {eval_sc:>10s} {match_str:>6s} {v['n_frames']:>6d} {v['size_mb']:>5.1f}")

# ── Task 2: Generate contact sheet ───────────────────────────────
print("\n" + "=" * 100)
print("TASK 2: Generating Contact Sheet")
print("=" * 100)

N_COLS = 5  # t=0, 25%, 50%, 75%, final
n_rows = len(videos)

fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 3.5, n_rows * 3))
if n_rows == 1:
    axes = axes.reshape(1, -1)

col_pcts = [0, 0.25, 0.50, 0.75, 1.0]
col_labels = ["t=0%", "t=25%", "t=50%", "t=75%", "t=final"]

for col in range(N_COLS):
    axes[0, col].set_title(col_labels[col], fontsize=8, fontweight="bold")

for row, v in enumerate(videos):
    # Build label
    label = v["filename"].replace(".mp4", "").replace("scratch_lambda005_", "scr_")
    if v["score_in_name"] is not None:
        label += f"\nscore={v['score_in_name']:.3f}"
    if v["score_in_eval"] is not None:
        label += f" (eval:{v['score_in_eval']:.3f})"
    if not v["eval_match"]:
        label += " [NO EVAL]"

    # Color-code row label
    color = "black"
    if v["score_in_name"] is not None and v["score_in_eval"] is not None:
        if abs(v["score_in_name"] - v["score_in_eval"]) < 0.001:
            color = "green"
        else:
            color = "red"

    axes[row, 0].set_ylabel(label, fontsize=6, color=color, rotation=0, ha="right", va="center",
                             labelpad=60, fontfamily="monospace")

    if v["n_frames"] == 0:
        for col in range(N_COLS):
            axes[row, col].text(0.5, 0.5, "NO VIDEO", ha="center", va="center", fontsize=10)
            axes[row, col].axis("off")
        continue

    try:
        vid = imageio.mimread(v["path"])
        for col, pct in enumerate(col_pcts):
            idx = min(int((len(vid) - 1) * pct), len(vid) - 1)
            axes[row, col].imshow(vid[idx])
            axes[row, col].set_title(f"t={idx}", fontsize=7)
            axes[row, col].axis("off")
    except Exception as e:
        for col in range(N_COLS):
            axes[row, col].text(0.5, 0.5, f"ERROR: {e}", ha="center", va="center", fontsize=6)
            axes[row, col].axis("off")

fig.suptitle("Video Audit Contact Sheet — All Videos in final_visual_summary/videos/",
             fontsize=11, fontweight="bold")
fig.tight_layout()
contact_path = FIG_DIR / "video_audit_contact_sheet.png"
fig.savefig(contact_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {contact_path}")

# ── Task 3: Deep-dive on best/median/failure ─────────────────────
print("\n" + "=" * 100)
print("TASK 3: Deep-dive on best/median/failure videos")
print("=" * 100)

focus_videos = [v for v in videos if v["variant"] in ("best", "median", "failure")]

for v in focus_videos:
    print(f"\n--- {v['filename']} ---")
    print(f"  seed: {v['seed']}")
    print(f"  score_in_filename: {v['score_in_name']:.3f}")
    print(f"  score_in_eval_json: {v['score_in_eval']:.3f}" if v['score_in_eval'] is not None else "  score_in_eval_json: N/A")
    print(f"  frames: {v['n_frames']}")

    # Assess visual status
    try:
        vid = imageio.mimread(v["path"])
        # Quick heuristic: check if the last frame is different from the first
        # (if the agent moved the T-block)
        first_frame = vid[0].astype(float)
        last_frame = vid[-1].astype(float)
        diff = np.mean(np.abs(last_frame - first_frame))

        if v["score_in_name"] is not None and v["score_in_name"] > 0.9:
            visual_status = "likely_success (score>0.9)"
        elif v["score_in_name"] is not None and v["score_in_name"] > 0.5:
            visual_status = "likely_partial (0.5<score<0.9)"
        elif v["score_in_name"] is not None and v["score_in_name"] < 0.1:
            visual_status = "likely_failure (score<0.1)"
        else:
            visual_status = "uncertain"

        print(f"  frame_diff_metric: {diff:.1f} (pixel-level change)")
        print(f"  visual_final_status: {visual_status}")

        # Generate per-video keyframe figure
        n = len(vid)
        key_idxs = [0, n//4, n//2, 3*n//4, n-1]
        fig2, axes2 = plt.subplots(1, 5, figsize=(18, 4.5))
        for ax, idx in zip(axes2, key_idxs):
            ax.imshow(vid[min(idx, n-1)])
            ax.set_title(f"t={min(idx, n-1)}", fontsize=9)
            ax.axis("off")
        fig2.suptitle(f"{v['filename']}  |  score={v['score_in_name']:.3f}  |  eval={v['score_in_eval']:.3f}" if v['score_in_eval'] else "",
                      fontsize=10, fontweight="bold")
        fig2.tight_layout()
        audit_path = FIG_DIR / f"audit_scratch_{v['variant']}_seed{v['seed']}.png"
        fig2.savefig(audit_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  saved: {audit_path}")

    except Exception as e:
        print(f"  ERROR reading video: {e}")
        visual_status = "error"

# ── Task 4: Check rollout generation method ─────────────────────
print("\n" + "=" * 100)
print("TASK 4: Rollout Generation Check")
print("=" * 100)

# Check the video generation script
video_script = ROOT / "scripts" / "final_visual_summary.py"
if video_script.exists():
    content = open(video_script).read()

    uses_cached = "traj_dir" in content or "diag_traj" in content
    rerolls = "rollout_one" in content
    has_cublas = "CUBLAS_WORKSPACE_CONFIG" in content
    has_deterministic = "use_deterministic_algorithms" in content
    has_cudnn = "cudnn.deterministic" in content

    print(f"  Script: {video_script}")
    print(f"  Uses cached trajectories: {uses_cached}")
    print(f"  Re-rolls from checkpoint: {rerolls}")
    print(f"  Sets CUBLAS_WORKSPACE_CONFIG: {has_cublas}")
    print(f"  torch.use_deterministic_algorithms: {has_deterministic}")
    print(f"  torch.backends.cudnn.deterministic: {has_cudnn}")

    if rerolls and not has_cublas:
        print()
        print("  ⚠ WARNING: Videos are re-generated rollouts WITHOUT CUBLAS_WORKSPACE_CONFIG.")
        print("    Same seed + same checkpoint may produce DIFFERENT results on re-run.")
        print("    Score in filename = eval JSON score, NOT necessarily the video's actual score.")

    # Check if the old 01-04 videos came from a different source
    old_vids = [v for v in videos if v["filename"].startswith("0")]
    new_vids = [v for v in videos if not v["filename"].startswith("0") and v["run_id"] != "comparison"]
    print(f"\n  Old videos (01-04 prefix): {len(old_vids)} (from previous session)")
    print(f"  New videos (regenerated): {len(new_vids)} (from --videos-only run)")

# ── Save audit JSON ──────────────────────────────────────────────
audit_data = {
    "audit_timestamp": datetime.datetime.now().isoformat(),
    "cublas_warning": "Videos are regenerated rollouts without CUBLAS_WORKSPACE_CONFIG. Same seed may produce different results on re-run.",
    "videos": [],
}
for v in videos:
    # Check if score matches
    score_match = None
    if v["score_in_name"] is not None and v["score_in_eval"] is not None:
        score_match = abs(v["score_in_name"] - v["score_in_eval"]) < 0.001

    audit_data["videos"].append({
        "filename": v["filename"],
        "run_id": v["run_id"],
        "seed": v["seed"],
        "variant": v["variant"],
        "score_in_name": v["score_in_name"],
        "score_in_eval": v["score_in_eval"],
        "score_match": score_match,
        "eval_entry_exists": v["eval_match"],
        "n_frames": v["n_frames"],
        "size_mb": round(v["size_mb"], 2),
    })

audit_json_path = JSON_DIR / "video_audit_report.json"
with open(audit_json_path, "w") as f:
    json.dump(audit_data, f, indent=2)
print(f"\nSaved: {audit_json_path}")

print("\n" + "=" * 100)
print("AUDIT COMPLETE")
print("=" * 100)
