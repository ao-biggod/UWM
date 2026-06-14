#!/usr/bin/env python3
"""Deterministic paired evaluation: DP-only vs UWM joint on same 50 seeds.

Key difference from previous evaluations:
  - random.seed(seed), np.random.seed(seed), torch.manual_seed(seed),
    torch.cuda.manual_seed_all(seed) at the start of EVERY episode
  - Same env seed for both models (re-seeded between models)
  - Config C only: norm_agent_pos=True, clip_sample=False
  - Seeds: 100000-100049

Output: per-episode scores, paired statistics.
"""
import argparse, json, os, sys, time, random
from pathlib import Path
from functools import partial

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


# ──────────────────────────────────────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────────────────────────────────────

def build_dp_only(device, clip_sample=False):
    from models.dp import ImageDiffusionPolicy, ImageObservationEncoder
    from models.dp import TransformerNoisePredictionNet

    shape_meta = {
        "obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                "agent_pos": {"shape": [2], "type": "low_dim"}},
        "action": {"shape": [2]},
    }
    obs_encoder = ImageObservationEncoder(
        shape_meta=shape_meta, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False,
        pretrained_weights=None,
        use_low_dim=True, use_language=False,
    )
    model = ImageDiffusionPolicy(
        action_len=16, action_dim=2,
        obs_encoder=obs_encoder,
        noise_pred_net=partial(
            TransformerNoisePredictionNet,
            input_len=16, input_dim=2,
            timestep_embed_dim=256, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        ),
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample,
    )
    return model.to(device)


def build_uwm(device, clip_sample=False, conditioning_type="adaln",
              dp_only=False, dynamics_loss_weight=1.0, self_attn_mask=None):
    from models.uwm import UnifiedWorldModel
    from models.uwm.obs_encoder import UWMObservationEncoder

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
        beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample,
        conditioning_type=conditioning_type,
        dp_only=dp_only,
        dynamics_loss_weight=dynamics_loss_weight,
        self_attn_mask=self_attn_mask,
    )
    return m.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Normalizer helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_lowdim_normalizer_dp(ckpt):
    """Reconstruct agent_pos normalizer for DP-only ckpt (not stored)."""
    if "lowdim_normalizer" in ckpt:
        ln = ckpt["lowdim_normalizer"]
        return np.array(ln["agent_pos"]["scale"]), np.array(ln["agent_pos"]["offset"])
    import zarr
    z = zarr.open("diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr", "r")
    ep_ends = z["meta/episode_ends"][:]
    train_end = ep_ends[89]
    state = z["data/state"][:train_end, :2]
    scale = (state.max(axis=0) - state.min(axis=0)) / 2.0
    offset = (state.max(axis=0) + state.min(axis=0)) / 2.0
    return scale, offset


def get_lowdim_normalizer_uwm(ckpt):
    ln = ckpt["lowdim_normalizer"]["agent_pos"]
    return np.array(ln["scale"]), np.array(ln["offset"])


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic episode evaluation
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


def eval_one_episode(model, device, action_scale, action_offset,
                     ap_scale, ap_offset, norm_agent_pos, seed):
    """Run one episode with deterministic seeding. Returns max_reward."""
    model.eval()
    env = make_env(seed)
    obs = env.reset()
    rewards = []
    done = False
    step = 0

    while not done and step < 300:
        # Preprocess observation
        img = obs["image"]
        agent_pos = obs["agent_pos"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(agent_pos, np.ndarray):
            agent_pos = torch.from_numpy(agent_pos)

        img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        ap = agent_pos.float().to(device)

        if norm_agent_pos:
            ap = (ap - torch.tensor(ap_offset, device=device).float()) / torch.tensor(ap_scale, device=device).float()

        obs_model = {
            "image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
            "agent_pos": ap.unsqueeze(0),
        }

        with torch.no_grad():
            action_norm = model.sample(obs_model)[0]

        action_raw = (action_norm
                      * torch.tensor(action_scale, device=device).float()
                      + torch.tensor(action_offset, device=device).float())
        action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
        exec_actions = action_raw_np[:8]

        obs, reward, done, info = env.step(exec_actions)
        rewards.append(float(reward))
        done = bool(np.all(done))
        step += 1

    return float(max(rewards)) if rewards else 0.0


def eval_all_episodes(model, device, action_scale, action_offset,
                      ap_scale, ap_offset, norm_agent_pos,
                      seeds, label):
    """Evaluate model on all seeds, seeding everything per episode."""
    scores = []
    t0 = time.time()

    for i, seed in enumerate(seeds):
        seed_everything(seed)
        score = eval_one_episode(model, device, action_scale, action_offset,
                                 ap_scale, ap_offset, norm_agent_pos, seed)
        scores.append(score)
        if i < 5 or i % 10 == 0 or i == len(seeds) - 1:
            print(f"  [{label}] Ep {i:3d} (seed={seed}): max_reward={score:.4f}", flush=True)

    elapsed = time.time() - t0
    arr = np.array(scores)
    print(f"\n  [{label}] {len(seeds)} eps in {elapsed:.0f}s: "
          f"mean={arr.mean():.4f} median={np.median(arr):.4f} "
          f"ep>0.5={(arr > 0.5).sum()}/{len(seeds)}", flush=True)
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deterministic paired eval: DP-only vs UWM joint")
    parser.add_argument("--dp-ckpt", type=str,
                        default="outputs/dp_pusht/run_20k_bs64/latest.pt")
    parser.add_argument("--uwm-ckpt", type=str,
                        default="artifacts_keep/uwm_20k/checkpoint_20k_latest.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-eps", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--output-dir", type=str,
                        default="outputs/deterministic_paired_eval")
    parser.add_argument("--conditioning-type", type=str, default=None,
                        choices=["adaln", "cross_attn"],
                        help="Override conditioning type (default: auto-detect from checkpoint)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    set_cudnn_deterministic()
    seeds = [args.seed_start + i for i in range(args.n_eps)]

    # Detect UWM conditioning_type from checkpoint early (for header)
    uwm_cond_type = args.conditioning_type
    if uwm_cond_type is None and args.uwm_ckpt:
        uwm_ckpt_temp = torch.load(args.uwm_ckpt, map_location="cpu")
        uwm_cond_type = uwm_ckpt_temp.get("conditioning_type", "adaln")
        del uwm_ckpt_temp

    print("=" * 60)
    print("Deterministic Paired Evaluation: DP-only C vs UWM joint C")
    print(f"  Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)} episodes)")
    print(f"  DP ckpt: {args.dp_ckpt}")
    print(f"  UWM ckpt: {args.uwm_ckpt}")
    print(f"  Config C: norm_agent_pos=True, clip_sample=False")
    print(f"  UWM conditioning_type: {uwm_cond_type}")
    print("=" * 60)

    # ── Load DP-only ──────────────────────────────────────────────────────────

    print("\n[1/4] Loading DP-only model...", flush=True)
    dp_ckpt = torch.load(args.dp_ckpt, map_location=device)
    dp_an = dp_ckpt["action_normalizer"]
    dp_action_scale = np.array(dp_an["scale"])
    dp_action_offset = np.array(dp_an["offset"])
    dp_ap_scale, dp_ap_offset = get_lowdim_normalizer_dp(dp_ckpt)

    dp_model = build_dp_only(device, clip_sample=False)
    dp_model.load_state_dict(dp_ckpt["model"])
    dp_model.eval()

    print(f"  DP action_normalizer:  scale={dp_action_scale}  offset={dp_action_offset}")
    print(f"  DP agent_pos_normalizer: scale={dp_ap_scale}  offset={dp_ap_offset}")

    # ── Load UWM joint ────────────────────────────────────────────────────────

    print("\n[2/4] Loading UWM joint model...", flush=True)
    uwm_ckpt = torch.load(args.uwm_ckpt, map_location=device)
    uwm_an = uwm_ckpt["action_normalizer"]
    uwm_action_scale = np.array(uwm_an["scale"])
    uwm_action_offset = np.array(uwm_an["offset"])
    uwm_ap_scale, uwm_ap_offset = get_lowdim_normalizer_uwm(uwm_ckpt)

    # Auto-detect conditioning_type from checkpoint, allow override
    uwm_cond_type = args.conditioning_type
    if uwm_cond_type is None:
        uwm_cond_type = uwm_ckpt.get("conditioning_type", "adaln")

    uwm_cfg = uwm_ckpt.get("config", {})
    uwm_dp_only = uwm_ckpt.get("dp_only", uwm_cfg.get("dp_only", False))
    uwm_dyn_weight = uwm_cfg.get("dynamics_loss_weight", 1.0)
    uwm_sa_mask = uwm_cfg.get("self_attn_mask", None)

    uwm_model = build_uwm(device, clip_sample=False, conditioning_type=uwm_cond_type,
                          dp_only=uwm_dp_only, dynamics_loss_weight=uwm_dyn_weight,
                          self_attn_mask=uwm_sa_mask)
    uwm_model.load_state_dict(uwm_ckpt["model"], strict=False)
    uwm_model.eval()

    print(f"  UWM conditioning_type: {uwm_cond_type}")
    print(f"  UWM dp_only: {uwm_dp_only}  dyn_weight: {uwm_dyn_weight}  sa_mask: {uwm_sa_mask}")
    print(f"  UWM action_normalizer:  scale={uwm_action_scale}  offset={uwm_action_offset}")
    print(f"  UWM agent_pos_normalizer: scale={uwm_ap_scale}  offset={uwm_ap_offset}")

    # ── Evaluate DP-only ──────────────────────────────────────────────────────

    print("\n[3/4] Evaluating DP-only (Config C)...", flush=True)
    dp_scores = eval_all_episodes(dp_model, device, dp_action_scale, dp_action_offset,
                                  dp_ap_scale, dp_ap_offset, norm_agent_pos=True,
                                  seeds=seeds, label="DP-only")
    dp_arr = np.array(dp_scores)

    # ── Evaluate UWM joint ────────────────────────────────────────────────────

    print("\n[4/4] Evaluating UWM joint (Config C)...", flush=True)
    uwm_scores = eval_all_episodes(uwm_model, device, uwm_action_scale, uwm_action_offset,
                                   uwm_ap_scale, uwm_ap_offset, norm_agent_pos=True,
                                   seeds=seeds, label="UWM")
    uwm_arr = np.array(uwm_scores)

    # ── Paired statistics ─────────────────────────────────────────────────────

    delta = uwm_arr - dp_arr
    uwm_gt_dp = (delta > 0).sum()
    dp_gt_uwm = (delta < 0).sum()
    tie = (delta == 0).sum()

    print("\n" + "=" * 60)
    print("PAIRED RESULTS (deterministic, same 50 seeds)")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'ep>0.5':>10}")
    print("-" * 52)
    print(f"  {'DP-only C':<18} {dp_arr.mean():8.4f} {np.median(dp_arr):8.4f} "
          f"{dp_arr.std():8.4f} {(dp_arr > 0.5).sum():>8}/{len(seeds)}")
    print(f"  {'UWM joint C':<18} {uwm_arr.mean():8.4f} {np.median(uwm_arr):8.4f} "
          f"{uwm_arr.std():8.4f} {(uwm_arr > 0.5).sum():>8}/{len(seeds)}")

    print(f"\n  Paired Δ (UWM - DP):")
    print(f"    Δ mean:    {delta.mean():+.4f}")
    print(f"    Δ median:  {np.median(delta):+.4f}")
    print(f"    Δ std:     {delta.std():.4f}")
    print(f"    UWM > DP:  {uwm_gt_dp}/{len(seeds)}")
    print(f"    DP > UWM:  {dp_gt_uwm}/{len(seeds)}")
    if tie > 0:
        print(f"    tie:       {tie}/{len(seeds)}")

    # ── Save results ──────────────────────────────────────────────────────────

    # Determine winner per episode
    winners = []
    for d in delta:
        if d > 0:
            winners.append("UWM")
        elif d < 0:
            winners.append("DP")
        else:
            winners.append("tie")

    summary = {
        "config": "C (norm_agent_pos=True, clip_sample=False)",
        "uwm_conditioning_type": uwm_cond_type,
        "dp_ckpt": args.dp_ckpt,
        "uwm_ckpt": args.uwm_ckpt,
        "seeds": seeds,
        "uwm_mean": float(uwm_arr.mean()),
        "uwm_median": float(np.median(uwm_arr)),
        "uwm_std": float(uwm_arr.std()),
        "uwm_ep_gt_05": int((uwm_arr > 0.5).sum()),
        "uwm_ep_total": len(seeds),
        "dp_mean": float(dp_arr.mean()),
        "dp_median": float(np.median(dp_arr)),
        "dp_std": float(dp_arr.std()),
        "dp_ep_gt_05": int((dp_arr > 0.5).sum()),
        "dp_ep_total": len(seeds),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_std": float(delta.std()),
        "uwm_gt_dp_count": int(uwm_gt_dp),
        "dp_gt_uwm_count": int(dp_gt_uwm),
        "tie_count": int(tie),
        "per_episode_scores": {
            str(int(s)): {"dp_score": float(dp_scores[i]), "uwm_score": float(uwm_scores[i]),
                          "delta": float(delta[i]), "winner": winners[i]}
            for i, s in enumerate(seeds)
        },
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.output_dir, "per_episode_scores.csv")
    with open(csv_path, "w") as f:
        f.write("ep,seed,dp_score,uwm_score,delta_uwm_minus_dp,winner\n")
        for i, seed in enumerate(seeds):
            f.write(f"{i},{seed},{dp_scores[i]:.6f},{uwm_scores[i]:.6f},{delta[i]:.6f},{winners[i]}\n")

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
