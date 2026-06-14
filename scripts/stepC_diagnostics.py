#!/usr/bin/env python3
"""Step C: Diagnostic tasks 1-4 for lowdim oracle bottleneck.

Task 1: Normalizer audit (expert action stats, normalized stats, sampled stats)
Task 2: Action slicing review (documented inline, see report below)
Task 3: EMA comparison (documented inline, see report below)
Task 4: Offline action prediction sanity check (pred vs GT on val batch)
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for 'scripts' imports


# ---- Data loading ----
def load_zarr_data(zarr_path):
    import zarr
    z = zarr.open(zarr_path, "r")
    state = z["data/state"][:]
    action = z["data/action"][:]
    ep_ends = z["meta/episode_ends"][:]
    episodes = []
    start = 0
    for end in ep_ends:
        episodes.append((state[start:end], action[start:end]))
        start = end
    return episodes


# ---- Task 1: Normalizer Audit ----
def task1_normalizer_audit(args, device):
    print("=" * 70)
    print("TASK 1: NORMALIZER AUDIT")
    print("=" * 70)

    episodes = load_zarr_data(args.zarr_path)
    all_action = np.concatenate([a for _, a in episodes[:90]], axis=0)
    all_state = np.concatenate([s for s, _ in episodes[:90]], axis=0)

    # 1. Raw expert action stats
    print("\n--- 1. Raw expert action stats ---")
    print(f"  Shape:        {all_action.shape}")
    print(f"  Min:          {all_action.min(axis=0)}")
    print(f"  Max:          {all_action.max(axis=0)}")
    print(f"  Mean:         {all_action.mean(axis=0)}")
    print(f"  Std:          {all_action.std(axis=0)}")

    # 2. Mean/std normalizer
    a_mean = all_action.mean(axis=0)
    a_std = all_action.std(axis=0) + 1e-6
    act_meanstd = (all_action - a_mean) / a_std
    print("\n--- 2. Mean/std normalizer ---")
    print(f"  mean (offset): {a_mean}")
    print(f"  std  (scale):  {a_std}")
    print(f"  After norm: min={act_meanstd.min(axis=0)}, max={act_meanstd.max(axis=0)}, "
          f"mean={act_meanstd.mean(axis=0)}, std={act_meanstd.std(axis=0)}")

    # 3. DP official LinearNormalizer (min/max limits)
    a_min = all_action.min(axis=0)
    a_max = all_action.max(axis=0)
    a_offset = (a_max + a_min) / 2.0
    a_scale = (a_max - a_min) / 2.0
    a_scale[a_scale < 1e-6] = 1.0
    act_minmax = (all_action - a_offset) / a_scale
    print("\n--- 3. Min/max normalizer (DP official) ---")
    print(f"  offset:        {a_offset}")
    print(f"  scale:         {a_scale}")
    print(f"  After norm: min={act_minmax.min(axis=0)}, max={act_minmax.max(axis=0)}, "
          f"mean={act_minmax.mean(axis=0)}")

    # 4. Expert action sample
    print("\n--- 6. Expert action first 20 values ---")
    print(f"  {all_action[:20].flatten()}")

    # Load model and sample
    ckpt_path = args.checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"\n--- Loading model from {ckpt_path} ---")
        ckpt = torch.load(ckpt_path, map_location=device)

        # Instantiate model (inline import to avoid package issues)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "stepB_retrain_lowdim",
            str(Path(__file__).resolve().parent / "stepB_retrain_lowdim.py"))
        stepB = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stepB)
        LowdimStatePolicyV2 = stepB.LowdimStatePolicyV2
        clip_sample_flag = ckpt.get("clip_sample", True)
        model = LowdimStatePolicyV2(clip_sample=clip_sample_flag, num_inference_steps=10).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        norm_type = ckpt.get("norm_type", "meanstd")

        # Build normalizers from checkpoint
        an = ckpt["action_normalizer"]
        sn = ckpt["state_normalizer"]
        if norm_type == "minmax":
            a_offset_t = torch.tensor(an["offset"], device=device).float()
            a_scale_t = torch.tensor(an["scale"], device=device).float()
            s_offset_t = torch.tensor(sn["offset"], device=device).float()
            s_scale_t = torch.tensor(sn["scale"], device=device).float()
            def norm_action(x):
                return (x - a_offset_t) / a_scale_t
            def unnorm_action(x):
                return x * a_scale_t + a_offset_t
            def norm_state(x):
                return (x - s_offset_t) / s_scale_t
        else:
            a_mean_t = torch.tensor(an["mean"], device=device).float()
            a_std_t = torch.tensor(an["std"], device=device).float()
            s_mean_t = torch.tensor(sn["mean"], device=device).float()
            s_std_t = torch.tensor(sn["std"], device=device).float()
            def norm_action(x):
                return (x - a_mean_t) / a_std_t
            def unnorm_action(x):
                return x * a_std_t + a_mean_t
            def norm_state(x):
                return (x - s_mean_t) / s_std_t

        print(f"  Normalizer type: {norm_type}")
        print(f"  clip_sample: {clip_sample_flag}")

        # Task 1.4 + 1.5: Sample from model on validation states
        build_dataset = stepB.build_dataset
        obs_horizon, action_horizon, seq_len = 2, 16, 19
        dataset = build_dataset(episodes, 90, seq_len, obs_horizon, action_horizon)

        # Use first 64 samples as validation
        val_obs, val_act = dataset[:64]
        val_obs_np = val_obs.numpy()
        val_act_np = val_act.numpy()

        # Normalize state
        val_obs_t = norm_state(torch.from_numpy(val_obs_np).float().to(device))

        # Sample
        all_pred_norm = []
        with torch.no_grad():
            for i in range(0, len(val_obs_t), 16):
                batch = val_obs_t[i:i+16]
                pred_norm = model.sample(batch)
                all_pred_norm.append(pred_norm.cpu().numpy())
        all_pred_norm = np.concatenate(all_pred_norm, axis=0)

        # 4. Model sampled normalized action stats
        print("\n--- 4. Model sampled NORMALIZED action stats ---")
        print(f"  Shape: {all_pred_norm.shape}")
        print(f"  Min:   {all_pred_norm.min(axis=(0,1))}")
        print(f"  Max:   {all_pred_norm.max(axis=(0,1))}")
        print(f"  Mean:  {all_pred_norm.mean(axis=(0,1))}")
        print(f"  Std:   {all_pred_norm.std(axis=(0,1))}")

        # 5. Model sampled unnormalized action stats
        all_pred_t = torch.from_numpy(all_pred_norm).float().to(device)
        all_pred_raw = unnorm_action(all_pred_t).cpu().numpy()

        print("\n--- 5. Model sampled UNNORMALIZED action stats ---")
        print(f"  Shape:  {all_pred_raw.shape}")
        print(f"  Min:    {all_pred_raw.min(axis=(0,1))}")
        print(f"  Max:    {all_pred_raw.max(axis=(0,1))}")
        print(f"  Mean:   {all_pred_raw.mean(axis=(0,1))}")
        print(f"  Std:    {all_pred_raw.std(axis=(0,1))}")

        # 7. Model unnormalized action first 20 values
        gt_actions = val_act_np[:, obs_horizon - 1: obs_horizon - 1 + action_horizon]
        print("\n--- 7a. Expert GT action (first 20 values, first sample) ---")
        print(f"  {gt_actions[0].flatten()[:20]}")
        print("\n--- 7b. Model unnormalized action (first 20 values, first sample) ---")
        print(f"  {all_pred_raw[0].flatten()[:20]}")

        # Clamp fraction estimate
        n_clamped = np.sum((all_pred_raw <= 0.0) | (all_pred_raw >= 512.0))
        frac = n_clamped / all_pred_raw.size
        print(f"\n--- Clamp fraction ---")
        print(f"  Values at boundary (<=0 or >=512): {n_clamped}/{all_pred_raw.size} = {frac:.4f}")
        print(f"  Values < 0:   {(all_pred_raw < 0).sum()}")
        print(f"  Values > 512: {(all_pred_raw > 512).sum()}")

        # ---- Task 4: Offline action prediction sanity check ----
        print("\n\n" + "=" * 70)
        print("TASK 4: OFFLINE ACTION PREDICTION SANITY CHECK")
        print("=" * 70)

        # Normalized MSE
        act_target_norm = norm_action(torch.from_numpy(
            val_act_np[:, obs_horizon - 1: obs_horizon - 1 + action_horizon]
        ).float().to(device))
        act_target_raw = torch.from_numpy(
            val_act_np[:, obs_horizon - 1: obs_horizon - 1 + action_horizon]
        ).float().to(device)

        pred_norm_t = torch.from_numpy(all_pred_norm).float().to(device)
        pred_raw_t = torch.from_numpy(all_pred_raw).float().to(device)

        norm_mse = F.mse_loss(pred_norm_t, act_target_norm).item()
        raw_mse = F.mse_loss(pred_raw_t, act_target_raw).item()

        # Per-dimension MSE
        raw_mse_dim0 = F.mse_loss(pred_raw_t[:, :, 0], act_target_raw[:, :, 0]).item()
        raw_mse_dim1 = F.mse_loss(pred_raw_t[:, :, 1], act_target_raw[:, :, 1]).item()

        print(f"\n  Normalized MSE:     {norm_mse:.6f}")
        print(f"  Unnormalized MSE:   {raw_mse:.2f}")
        print(f"  Unnorm MSE (dim 0): {raw_mse_dim0:.2f}")
        print(f"  Unnorm MSE (dim 1): {raw_mse_dim1:.2f}")

        # Action-wise MSE (MSE per action step)
        print(f"\n  Per-timestep unnorm MSE:")
        for t in range(16):
            mse_t = F.mse_loss(pred_raw_t[:, t, :], act_target_raw[:, t, :]).item()
            print(f"    t={t:2d}: {mse_t:.1f}")

        # Print 3 sample comparisons
        print(f"\n  --- 3 Sample Comparisons (pred vs GT) ---")
        for idx in [0, 1, 2]:
            print(f"\n  Sample {idx}:")
            gt = act_target_raw[idx].cpu().numpy()
            pred = pred_raw_t[idx].cpu().numpy()
            print(f"    GT action range: [{gt.min(axis=0)}, {gt.max(axis=0)}]")
            print(f"    Pred range:      [{pred.min(axis=0)}, {pred.max(axis=0)}]")
            print(f"    GT first 5 steps:    {gt[:5]}")
            print(f"    Pred first 5 steps:  {pred[:5]}")
            # Per-timestep error
            abs_err = np.abs(gt - pred)
            print(f"    Mean abs error:      {abs_err.mean():.2f}")
            print(f"    Max abs error:       {abs_err.max():.2f}")
            print(f"    Mean abs err (dim0): {abs_err[:, 0].mean():.2f}")
            print(f"    Mean abs err (dim1): {abs_err[:, 1].mean():.2f}")

        # Check: is the first predicted action closer to GT than later ones?
        print(f"\n  --- Checking prediction quality decay ---")
        abs_err_full = np.abs(act_target_raw.cpu().numpy() - pred_raw_t.cpu().numpy())
        for t_range in [(0, 4), (4, 8), (8, 12), (12, 16)]:
            seg = abs_err_full[:, t_range[0]:t_range[1], :]
            print(f"    Steps {t_range[0]:2d}-{t_range[1]:2d}: mean_abs_err={seg.mean():.2f}")

        # Expert action std for comparison
        print(f"\n  --- Expert action std (for reference) ---")
        print(f"  Expert action std: {all_action.std(axis=0)}")
        print(f"  Prediction RMSE / Expert std: {np.sqrt(raw_mse) / all_action.std(axis=0)}")

        return {
            "norm_type": norm_type,
            "clip_sample": clip_sample_flag,
            "norm_mse": float(norm_mse),
            "raw_mse": float(raw_mse),
            "clamp_fraction": float(frac),
            "expert_action_stats": {
                "min": all_action.min(axis=0).tolist(),
                "max": all_action.max(axis=0).tolist(),
                "mean": all_action.mean(axis=0).tolist(),
                "std": all_action.std(axis=0).tolist(),
            },
            "pred_raw_stats": {
                "min": all_pred_raw.min(axis=(0, 1)).tolist(),
                "max": all_pred_raw.max(axis=(0, 1)).tolist(),
                "mean": all_pred_raw.mean(axis=(0, 1)).tolist(),
                "std": all_pred_raw.std(axis=(0, 1)).tolist(),
            },
        }
    else:
        print(f"\n  [SKIP] No checkpoint at {ckpt_path}, Task 4 skipped")
        return None


# ---- Task 2 & 3: Documented inline ----
def print_task2_findings():
    print("\n\n" + "=" * 70)
    print("TASK 2: ACTION SLICING REVIEW")
    print("=" * 70)
    print("""
  Files checked:
    - scripts/stepB_retrain_lowdim.py line 217
    - scripts/stepA_eval_noclip.py line 70
    - scripts/diag_lowdim_dp.py line 181
    - diffusion_policy-main/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py line 147-148

  ALL our eval scripts use:  action_exec = action_raw[:8]

  DP official uses:
    start = self.n_obs_steps - 1   # = 1
    end = start + self.n_action_steps  # = 1 + 8 = 9
    action = action_pred[:, start:end]  # action_pred[:, 1:9]

  KEY DIFFERENCE in model architecture:
    DP official: UNet conditions on BOTH obs AND noisy action.
      → action_pred[0] = action taken after obs[0] (FIRST obs in window)
      → Must skip to index 1 (=n_obs_steps-1) for action after CURRENT obs

    Our DiT: conditions ONLY on obs (no action input).
      → Training: given state[i:i+2], predict action[i+1:i+17]
      → action_pred[0] = action[i+1] = action after CURRENT obs (obs[1])
      → action_pred[:8] = correct 8 actions after current state

  VERDICT: action[:8] is CORRECT for our obs-conditioned DiT model.
  It is equivalent to DP official's action[1:9] for their obs+action-conditioned UNet.

  BUT NOTE: State spacing issue in eval loop:
    During eval, state_buffer accumulates states 8 physical steps apart
    (because each env.step executes 8 actions). Training uses states 1 step apart.
    This creates a distribution mismatch in state pairs.
    States 8 steps apart → velocity information is less precise.
""")


def print_task3_findings():
    print("\n\n" + "=" * 70)
    print("TASK 3: EMA COMPARISON")
    print("=" * 70)
    print("""
  DP official training config (artifacts_keep/dp_50epoch/hydra_config.yaml):
    use_ema: true
    ema:
      _target_: diffusion_policy.model.diffusion.ema_model.EMAModel
      inv_gamma: 1.0
      max_value: 0.9999
      min_value: 0.0
      power: 0.75
      update_after_step: 0

  Config files checked (ALL have use_ema: true):
    - train_diffusion_unet_hybrid_workspace.yaml (used by DP image baseline)
    - train_diffusion_transformer_lowdim_pusht_workspace.yaml
    - train_diffusion_unet_image_workspace.yaml
    - train_diffusion_unet_lowdim_workspace.yaml

  Our models:
    - DP-only same-backbone: NO EMA
    - Lowdim oracle (diag_lowdim_dp.py):    NO EMA
    - Lowdim oracle (stepB_retrain_lowdim.py): NO EMA
    - UWM: NO EMA

  Comparison:
  | Item                | DP official  | Our DP-only | Our lowdim | Same? |
  |---------------------|-------------|-------------|------------|-------|
  | use_ema             | True        | False       | False      | NO    |
  | EMA power           | 0.75        | N/A         | N/A        | NO    |
  | EMA max_value       | 0.9999      | N/A         | N/A        | NO    |
  | EMA for eval        | Yes         | No          | No         | NO    |
  | LR schedule         | cosine      | constant    | constant   | NO    |
  | Weight decay         | 1e-6        | 1e-6        | 1e-6       | Yes   |
  | Gradient clip        | 1.0         | 1.0         | 1.0        | Yes   |

  IMPACT: EMA typically improves diffusion policy performance by 10-30%.
  This alone doesn't explain the 0.18 vs 0.7 gap, but it's a meaningful factor.
""")


def main():
    parser = argparse.ArgumentParser(description="Step C: Diagnostics for lowdim oracle bottleneck")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--zarr-path", type=str,
                        default="diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/stepB_retrain/B2_minmax_clip/latest.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/stepC_diag")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip model loading (Tasks 1+4 require model)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Task 1 + 4 (interleaved - needs model)
    if not args.skip_model:
        results = task1_normalizer_audit(args, device)
    else:
        results = None

    # Task 2 + 3 (documented)
    print_task2_findings()
    print_task3_findings()

    # Summary
    print("\n\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    if results:
        print(f"""
  1. NORMALIZER AUDIT:
     - Raw expert action range: [{results['expert_action_stats']['min']}, {results['expert_action_stats']['max']}]
     - Pred raw action range:  [{results['pred_raw_stats']['min']}, {results['pred_raw_stats']['max']}]
     - Clamp fraction: {results['clamp_fraction']:.4f}
     - Coordinate range match: {'YES' if results['pred_raw_stats']['min'][0] > 0 else 'BOUNDARY HIT'}
     → Normalizer: {'SUSPICIOUS' if results['clamp_fraction'] > 0.01 else 'LIKELY OK'}

  2. ACTION SLICING:
     - action[:8] is correct for obs-conditioned DiT (see analysis above)
     - State spacing (8-step gap vs 1-step training) is a potential issue

  3. EMA / LR RECIPE:
     - DP official: EMA=True, LR=cosine
     - Our models:   EMA=False, LR=constant
     → EMA: SUSPICIOUS (missing optimization feature)
     → LR schedule: MODERATELY SUSPICIOUS

  4. OFFLINE MSE:
     - Norm MSE: {results['norm_mse']:.6f}
     - Raw MSE:  {results['raw_mse']:.1f}
     - Expert action std: {results['expert_action_stats']['std']}
""")

    print("  Ready for manual interpretation.")


if __name__ == "__main__":
    main()
