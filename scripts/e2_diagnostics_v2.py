#!/usr/bin/env python3
"""
E2 diagnostics v2: B3 vs UWM-DP-KP root-cause analysis.
Focus: conditioning method (obs-as-token vs AdaLN modulation).

E2-0: unified offline MSE
E2-1: same-obs action parity
E2-2: resampling variance
E2-2.5: obs sensitivity
E2-3: corrected contract diff table

Key correction: both B3 and UWM-DP-KP use bidirectional (non-causal) self-attention.
The primary difference is conditioning: B3 uses obs-as-token concat, UWM uses AdaLN.
"""
import argparse, json, os, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stepB_retrain_lowdim import LowdimStatePolicyV2, MinMaxNormalizer


def load_zarr_data_keypoint(zarr_path):
    """Load 20D keypoint obs = [keypoint(9,2)_flat(18), agent_pos(2)]"""
    import zarr
    z = zarr.open(zarr_path, "r")
    keypoint = z["data/keypoint"][:]  # (N, 9, 2)
    state = z["data/state"][:]  # (N, 5)
    action = z["data/action"][:]  # (N, 2)
    agent_pos = state[:, :2]
    obs_20d = np.concatenate([keypoint.reshape(keypoint.shape[0], -1), agent_pos], axis=-1)
    ep_ends = z["meta/episode_ends"][:]
    episodes = []
    start = 0
    for end in ep_ends:
        episodes.append((obs_20d[start:end], action[start:end]))
        start = end
    return episodes


def build_dataset(episodes, n_episodes, seq_len, obs_horizon=2, action_horizon=16):
    all_obs, all_act = [], []
    for ep_obs, ep_action in episodes[:n_episodes]:
        T = len(ep_obs)
        for i in range(T - seq_len):
            all_obs.append(ep_obs[i:i + obs_horizon])
            all_act.append(ep_action[i:i + seq_len])
    ds = torch.utils.data.TensorDataset(
        torch.tensor(np.stack(all_obs), dtype=torch.float32),
        torch.tensor(np.stack(all_act), dtype=torch.float32))
    return ds


def build_uwm_dit_policy(obs_dim, action_dim, action_len, device):
    """Replicate uwm_dp_only_keypoint20.py model construction."""
    from models.dp.transformer import TransformerNoisePredictionNet
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    class LowdimObsEncoder(torch.nn.Module):
        def __init__(self, obs_dim, num_frames, embed_dim):
            super().__init__()
            in_dim = obs_dim * num_frames
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, embed_dim),
                torch.nn.Mish(),
                torch.nn.Linear(embed_dim, embed_dim))
        def forward(self, obs):
            B, T, D = obs.shape
            return self.net(obs.reshape(B, T * D))

    class LowdimDiTPolicy(torch.nn.Module):
        def __init__(self, obs_dim, action_len, action_dim, embed_dim=768, depth=12, num_heads=12):
            super().__init__()
            self.action_len = action_len
            self.action_dim = action_dim
            self.obs_encoder = LowdimObsEncoder(obs_dim, 2, embed_dim)
            self.noise_pred_net = TransformerNoisePredictionNet(
                input_len=action_len, input_dim=action_dim,
                global_cond_dim=embed_dim,
                timestep_embed_dim=256, embed_dim=embed_dim,
                depth=depth, num_heads=num_heads, mlp_ratio=4, qkv_bias=True)
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                clip_sample=True, prediction_type="epsilon")
            self.num_inference_steps = 10

        def forward(self, obs, action):
            B = action.shape[0]
            obs_embed = self.obs_encoder(obs)
            noise = torch.randn_like(action)
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=action.device).long()
            noisy_action = self.noise_scheduler.add_noise(action, noise, t)
            noise_pred = self.noise_pred_net(noisy_action, t, global_cond=obs_embed)
            return F.mse_loss(noise_pred, noise)

        @torch.no_grad()
        def sample(self, obs, seed=None):
            B, device = obs.shape[0], obs.device
            obs_embed = self.obs_encoder(obs)
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)
                action = torch.randn(B, self.action_len, self.action_dim, device=device, generator=generator)
            else:
                action = torch.randn(B, self.action_len, self.action_dim, device=device)
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for t_step in self.noise_scheduler.timesteps:
                t = torch.full((B,), t_step, device=device, dtype=torch.long)
                noise_pred = self.noise_pred_net(action, t, global_cond=obs_embed)
                action = self.noise_scheduler.step(noise_pred, t_step, action).prev_sample
            return action

    return LowdimDiTPolicy(obs_dim=obs_dim, action_len=action_len, action_dim=action_dim).to(device)


def load_models(device):
    """Load B3 and UWM-DP-KP checkpoints."""
    obs_dim, action_dim, action_len = 20, 2, 16

    # B3
    b3_model = LowdimStatePolicyV2(obs_dim=obs_dim, action_len=action_len, action_dim=action_dim)
    b3_ckpt = torch.load("artifacts_keep/B3_keypoint20_local_dit_20k/latest.pt", map_location=device)
    b3_model.load_state_dict(b3_ckpt["model"])
    b3_model.to(device)
    b3_model.eval()
    b3_n_state = MinMaxNormalizer()
    b3_n_state.offset = np.array(b3_ckpt["state_normalizer"]["offset"])
    b3_n_state.scale = np.array(b3_ckpt["state_normalizer"]["scale"])
    b3_n_action = MinMaxNormalizer()
    b3_n_action.offset = np.array(b3_ckpt["action_normalizer"]["offset"])
    b3_n_action.scale = np.array(b3_ckpt["action_normalizer"]["scale"])

    # UWM-DP-KP
    uwm_model = build_uwm_dit_policy(obs_dim=obs_dim, action_dim=action_dim, action_len=action_len, device=device)
    uwm_ckpt = torch.load("artifacts_keep/uwm_dp_only_keypoint20_20k/latest.pt", map_location=device)
    uwm_model.load_state_dict(uwm_ckpt["model"])
    uwm_model.eval()
    uwm_n_state = MinMaxNormalizer()
    uwm_n_state.offset = np.array(uwm_ckpt["state_normalizer"]["offset"])
    uwm_n_state.scale = np.array(uwm_ckpt["state_normalizer"]["scale"])
    uwm_n_action = MinMaxNormalizer()
    uwm_n_action.offset = np.array(uwm_ckpt["action_normalizer"]["offset"])
    uwm_n_action.scale = np.array(uwm_ckpt["action_normalizer"]["scale"])

    return {
        "b3": (b3_model, b3_n_state, b3_n_action),
        "uwm": (uwm_model, uwm_n_state, uwm_n_action),
    }


def normalize_obs(obs, norm):
    return (obs - torch.tensor(norm.offset, device=obs.device).float()) / torch.tensor(norm.scale, device=obs.device).float()


def unnormalize_action(act, norm):
    return act * torch.tensor(norm.scale, device=act.device).float() + torch.tensor(norm.offset, device=act.device).float()


def sample_seeded(model, obs_norm, seed, model_name):
    """Sample with fixed seed. model_name is 'b3' or 'uwm'."""
    B, device = obs_norm.shape[0], obs_norm.device
    if model_name == "uwm":
        # UWM: our custom sample() already supports seed
        return model.sample(obs_norm, seed=seed)
    elif model_name == "b3":
        # B3: manual re-implementation with fixed noise seed
        obs_feat = model.obs_proj(obs_norm.reshape(B, -1)).unsqueeze(1)
        generator = torch.Generator(device=device).manual_seed(seed)
        action = torch.randn(B, model.action_len, model.action_dim, device=device, generator=generator)
        model.noise_scheduler.set_timesteps(model.num_inference_steps)
        for t_step in model.noise_scheduler.timesteps:
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            temb = model._time_emb(t, B, device)
            act_emb = model.action_embed(action) + model.pos_embed
            x = torch.cat([obs_feat, act_emb], dim=1) + temb
            x = model.transformer(x)
            noise_pred = model.action_decoder(x[:, 1:])
            action = model.noise_scheduler.step(noise_pred, t_step, action).prev_sample
        return action
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def e2_0_offline_mse(models, device):
    """Unified offline MSE. Both evaluate on same 64 samples (first 64 of val set)."""
    print("=" * 60)
    print("E2-0: Unified Offline MSE")
    print("=" * 60)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    all_obs = np.concatenate([s for s, _ in episodes], axis=0)
    all_action = np.concatenate([a for _, a in episodes], axis=0)
    expert_action_std = float(all_action.std())
    print(f"  expert_action_std: {expert_action_std:.2f}")
    print(f"  expert_action_mean: {float(all_action.mean()):.2f}")
    print(f"  N_expert: {len(all_action)}")

    obs_horizon, action_horizon, seq_len = 2, 16, 19
    dataset = build_dataset(episodes, 90, seq_len, obs_horizon, action_horizon)

    # Use last 64 samples as test (avoid training samples)
    test_offset = max(0, len(dataset) - 64)
    obs_test, act_test = dataset[test_offset:test_offset + 64]
    obs_test = obs_test.to(device)
    act_test = act_test.to(device)
    gt_raw = act_test[:, obs_horizon - 1: obs_horizon - 1 + action_horizon]  # [B, 16, 2]

    results = {}
    for name in ["b3", "uwm"]:
        model, n_state, n_action = models[name]
        obs_norm = normalize_obs(obs_test, n_state).float()

        with torch.no_grad():
            act_pred_norm = model.sample(obs_norm)
        act_pred_raw = unnormalize_action(act_pred_norm, n_action)

        mse_all = F.mse_loss(act_pred_raw, gt_raw).item()
        mse_first8 = F.mse_loss(act_pred_raw[:, :8], gt_raw[:, :8]).item()
        exec_start = obs_horizon - 1  # 1
        exec_end = exec_start + 8  # 9
        mse_exec8 = F.mse_loss(act_pred_raw[:, exec_start:exec_end], gt_raw[:, exec_start:exec_end]).item()
        mse_t0 = F.mse_loss(act_pred_raw[:, 0], gt_raw[:, 0]).item()
        mse_t7 = F.mse_loss(act_pred_raw[:, 7], gt_raw[:, 7]).item()
        mse_t15 = F.mse_loss(act_pred_raw[:, 15], gt_raw[:, 15]).item()
        rmse_all = np.sqrt(mse_all)
        rmse_all_div_std = rmse_all / expert_action_std

        print(f"\n  [{name.upper()}]")
        print(f"    all16_raw_mse:          {mse_all:.1f}")
        print(f"    first8_raw_mse:         {mse_first8:.1f}")
        print(f"    exec8_raw_mse (t1:t9):  {mse_exec8:.1f}")
        print(f"    t0_raw_mse:             {mse_t0:.1f}")
        print(f"    t7_raw_mse:             {mse_t7:.1f}")
        print(f"    t15_raw_mse:            {mse_t15:.1f}")
        print(f"    all16_raw_rmse:         {rmse_all:.1f}")
        print(f"    rmse/expert_std:        {rmse_all_div_std:.4f}")

        results[name] = {
            "all16_raw_mse": mse_all,
            "first8_raw_mse": mse_first8,
            "exec8_raw_mse": mse_exec8,
            "t0_raw_mse": mse_t0,
            "t7_raw_mse": mse_t7,
            "t15_raw_mse": mse_t15,
            "all16_raw_rmse": rmse_all,
            "rmse_over_expert_std": rmse_all_div_std,
        }

    with open("outputs/e2_b3_vs_uwm/e2_offline_mse.json", "w") as f:
        json.dump({"expert_action_std": expert_action_std, "results": results}, f, indent=2)
    print(f"\n  saved: outputs/e2_b3_vs_uwm/e2_offline_mse.json")
    return results


def e2_1_action_parity(models, device):
    """Same-obs action parity: 128 val obs, feed to both models, compare."""
    print("=" * 60)
    print("E2-1: Same-Obs Action Parity")
    print("=" * 60)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    obs_horizon, action_horizon, seq_len = 2, 16, 19
    dataset = build_dataset(episodes, 90, seq_len, obs_horizon, action_horizon)

    N = 128
    obs_test, act_test = dataset[:N]
    obs_test = obs_test.to(device)
    act_test = act_test.to(device)
    gt_raw = act_test[:, obs_horizon - 1: obs_horizon - 1 + action_horizon]

    exec_start = obs_horizon - 1  # 1
    exec_end = exec_start + 8  # 9

    preds = {}
    for name in ["b3", "uwm"]:
        model, n_state, n_action = models[name]
        obs_norm = normalize_obs(obs_test, n_state).float()
        with torch.no_grad():
            act_pred_norm = model.sample(obs_norm)
        act_pred_raw = unnormalize_action(act_pred_norm, n_action)
        preds[name] = act_pred_raw

    # First executable action
    b3_first = preds["b3"][:, exec_start]
    uwm_first = preds["uwm"][:, exec_start]
    gt_first = gt_raw[:, exec_start]

    b3_vs_gt = torch.norm(b3_first - gt_first, dim=-1)
    uwm_vs_gt = torch.norm(uwm_first - gt_first, dim=-1)
    b3_vs_uwm = torch.norm(b3_first - uwm_first, dim=-1)

    print(f"  exec_start={exec_start}, exec_end={exec_end}  (rollout executes t1:t9)")
    print(f"\n  First executable action L2:")
    print(f"    B3-vs-GT mean/median/max:  {b3_vs_gt.mean():.1f} / {b3_vs_gt.median():.1f} / {b3_vs_gt.max():.1f}")
    print(f"    UWM-vs-GT mean/median/max: {uwm_vs_gt.mean():.1f} / {uwm_vs_gt.median():.1f} / {uwm_vs_gt.max():.1f}")
    print(f"    B3-vs-UWM mean/median/max: {b3_vs_uwm.mean():.1f} / {b3_vs_uwm.median():.1f} / {b3_vs_uwm.max():.1f}")

    # Full exec8 L2
    b3_exec8 = preds["b3"][:, exec_start:exec_end]
    uwm_exec8 = preds["uwm"][:, exec_start:exec_end]
    gt_exec8 = gt_raw[:, exec_start:exec_end]
    l2_all_b3 = torch.norm(b3_exec8.reshape(N, -1) - gt_exec8.reshape(N, -1), dim=-1)
    l2_all_uwm = torch.norm(uwm_exec8.reshape(N, -1) - gt_exec8.reshape(N, -1), dim=-1)
    l2_b3_uwm = torch.norm(b3_exec8.reshape(N, -1) - uwm_exec8.reshape(N, -1), dim=-1)
    print(f"\n  Exec8 action L2:")
    print(f"    B3-vs-GT mean/median/max:  {l2_all_b3.mean():.1f} / {l2_all_b3.median():.1f} / {l2_all_b3.max():.1f}")
    print(f"    UWM-vs-GT mean/median/max: {l2_all_uwm.mean():.1f} / {l2_all_uwm.median():.1f} / {l2_all_uwm.max():.1f}")
    print(f"    B3-vs-UWM mean/median/max: {l2_b3_uwm.mean():.1f} / {l2_b3_uwm.median():.1f} / {l2_b3_uwm.max():.1f}")

    # Summary of B3-UWM agreement
    agreement = (b3_vs_uwm < b3_vs_gt).float().mean()
    print(f"\n  Agreement: B3-UWM < B3-GT in {agreement*100:.1f}% of cases")

    # Print first 20 rows
    print(f"\n  {'idx':<5} {'gt_exec_t1':<24} {'B3_exec_t1':<24} {'UWM_exec_t1':<24} {'B3-GT L2':<10} {'UWM-GT L2':<10} {'B3-UWM L2'}")
    print(f"  {'-'*117}")
    for i in range(min(20, N)):
        g0, g1 = gt_first[i, 0].item(), gt_first[i, 1].item()
        b0, b1 = b3_first[i, 0].item(), b3_first[i, 1].item()
        u0, u1 = uwm_first[i, 0].item(), uwm_first[i, 1].item()
        print(f"  {i:<5} [{g0:7.1f}, {g1:7.1f}]     [{b0:7.1f}, {b1:7.1f}]     "
              f"[{u0:7.1f}, {u1:7.1f}]     {b3_vs_gt[i]:5.1f}       "
              f"{uwm_vs_gt[i]:5.1f}       {b3_vs_uwm[i]:5.1f}")

    results = {
        "exec_slice": [exec_start, exec_end],
        "N": N,
        "first_action_b3_vs_gt": {"mean": float(b3_vs_gt.mean()), "median": float(b3_vs_gt.median()), "max": float(b3_vs_gt.max())},
        "first_action_uwm_vs_gt": {"mean": float(uwm_vs_gt.mean()), "median": float(uwm_vs_gt.median()), "max": float(uwm_vs_gt.max())},
        "first_action_b3_vs_uwm": {"mean": float(b3_vs_uwm.mean()), "median": float(b3_vs_uwm.median()), "max": float(b3_vs_uwm.max())},
        "exec8_b3_vs_gt": {"mean": float(l2_all_b3.mean()), "median": float(l2_all_b3.median())},
        "exec8_uwm_vs_gt": {"mean": float(l2_all_uwm.mean()), "median": float(l2_all_uwm.median())},
        "exec8_b3_vs_uwm": {"mean": float(l2_b3_uwm.mean()), "median": float(l2_b3_uwm.median())},
        "agreement_frac": float(agreement),
    }

    with open("outputs/e2_b3_vs_uwm/e2_same_obs_parity.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def e2_2_resampling_variance(models, device):
    """K=32 resamples per obs: measure per-state sampling variance."""
    print("=" * 60)
    print("E2-2: Resampling Variance (K=32)")
    print("=" * 60)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    obs_horizon, action_horizon, seq_len = 2, 16, 19
    dataset = build_dataset(episodes, 90, seq_len, obs_horizon, action_horizon)

    N = 128
    K = 32
    obs_test, act_test = dataset[:N]
    obs_test = obs_test.to(device)

    exec_start = obs_horizon - 1

    all_results = {}
    for name in ["b3", "uwm"]:
        model, n_state, n_action = models[name]
        obs_norm = normalize_obs(obs_test, n_state).float()

        all_actions = []
        for k in range(K):
            seed = 10000 + k
            with torch.no_grad():
                act_pred_norm = sample_seeded(model, obs_norm, seed, name)
            act_pred_raw = unnormalize_action(act_pred_norm, n_action)
            all_actions.append(act_pred_raw.cpu().numpy())

        all_actions = np.stack(all_actions, axis=0)  # (K, N, 16, 2)

        # Per-sample std, then mean across samples
        first_action_std = np.stack([all_actions[:, i, exec_start, 0].std() for i in range(N)])
        first_action_std_l2 = np.array([np.sqrt(all_actions[:, i, exec_start, 0].var() + all_actions[:, i, exec_start, 1].var()) for i in range(N)])
        exec8_std = np.array([all_actions[:, i, exec_start:exec_start+8].std(axis=0).mean() for i in range(N)])
        all16_std = np.array([all_actions[:, i].std(axis=0).mean() for i in range(N)])

        # Plan delta: L2 between consecutive resamples
        plan_deltas = []
        for i in range(N):
            deltas = []
            for k in range(K - 1):
                d = np.sqrt(np.sum((all_actions[k+1, i, exec_start] - all_actions[k, i, exec_start])**2))
                deltas.append(d)
            plan_deltas.append(np.mean(deltas))
        plan_delta_mean = np.mean(plan_deltas)

        print(f"\n  [{name.upper()}] K={K}")
        print(f"    first_action_std (dim0):      {first_action_std.mean():.2f}")
        print(f"    first_action_std_L2:          {first_action_std_l2.mean():.2f}")
        print(f"    exec8_action_std (mean):      {exec8_std.mean():.2f}")
        print(f"    all16_action_std (mean):      {all16_std.mean():.2f}")
        print(f"    plan_delta (consecutive):     {plan_delta_mean:.2f}")

        all_results[name] = {
            "first_action_std_dim0": float(first_action_std.mean()),
            "first_action_std_L2": float(first_action_std_l2.mean()),
            "exec8_action_std": float(exec8_std.mean()),
            "all16_action_std": float(all16_std.mean()),
            "plan_delta": float(plan_delta_mean),
        }

    # Ratio
    ratio_first_std = all_results["uwm"]["first_action_std_L2"] / max(all_results["b3"]["first_action_std_L2"], 1e-8)
    ratio_plan_delta = all_results["uwm"]["plan_delta"] / max(all_results["b3"]["plan_delta"], 1e-8)
    all_results["uwm_vs_b3_ratio"] = {"first_action_std_L2": float(ratio_first_std), "plan_delta": float(ratio_plan_delta)}
    print(f"\n  UWM/B3 ratio: first_action_std_L2={ratio_first_std:.2f}, plan_delta={ratio_plan_delta:.2f}")

    with open("outputs/e2_b3_vs_uwm/e2_resampling_variance.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def e2_25_obs_sensitivity(models, device):
    """Obs sensitivity: shuffle obs / zero obs, measure action change."""
    print("=" * 60)
    print("E2-2.5: Obs Sensitivity (conditioning diagnostic)")
    print("=" * 60)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    obs_horizon, action_horizon, seq_len = 2, 16, 19
    dataset = build_dataset(episodes, 90, seq_len, obs_horizon, action_horizon)

    N = 128
    obs_test, act_test = dataset[:N]
    obs_test = obs_test.to(device)

    exec_start = obs_horizon - 1
    exec_end = exec_start + 8

    results = {}

    for name in ["b3", "uwm"]:
        model, n_state, n_action = models[name]
        obs_norm = normalize_obs(obs_test, n_state).float()

        # 1) Normal obs
        with torch.no_grad():
            act_norm = model.sample(obs_norm)
        act_normal = unnormalize_action(act_norm, n_action)

        # 2) Shuffled obs (within batch)
        shuffle_idx = torch.randperm(N)
        obs_shuffled = obs_norm[shuffle_idx]
        with torch.no_grad():
            act_norm_shuffled = model.sample(obs_shuffled)
        act_shuffled = unnormalize_action(act_norm_shuffled, n_action)

        # 3) Zero obs (normalized zero = center of [-1,1] range → ~0)
        # MinMax normalizer: (x - offset)/scale. center_x = offset gives normalized 0.
        obs_zero_norm = torch.zeros_like(obs_norm)
        with torch.no_grad():
            act_norm_zero = model.sample(obs_zero_norm)
        act_zero = unnormalize_action(act_norm_zero, n_action)

        # L2 differences
        normal_vs_shuffled = torch.norm((act_normal - act_shuffled).reshape(N, -1), dim=-1)
        normal_vs_zero = torch.norm((act_normal - act_zero).reshape(N, -1), dim=-1)
        normal_vs_shuffled_exec8 = torch.norm((act_normal[:, exec_start:exec_end] - act_shuffled[:, exec_start:exec_end]).reshape(N, -1), dim=-1)
        normal_vs_zero_exec8 = torch.norm((act_normal[:, exec_start:exec_end] - act_zero[:, exec_start:exec_end]).reshape(N, -1), dim=-1)

        print(f"\n  [{name.upper()}]")
        print(f"    normal_vs_shuffled all16 L2:   {normal_vs_shuffled.mean():.1f} (median={normal_vs_shuffled.median():.1f})")
        print(f"    normal_vs_shuffled exec8 L2:    {normal_vs_shuffled_exec8.mean():.1f} (median={normal_vs_shuffled_exec8.median():.1f})")
        print(f"    normal_vs_zero all16 L2:        {normal_vs_zero.mean():.1f} (median={normal_vs_zero.median():.1f})")
        print(f"    normal_vs_zero exec8 L2:        {normal_vs_zero_exec8.mean():.1f} (median={normal_vs_zero_exec8.median():.1f})")

        # Obs sensitivity via finite-difference perturbation
        # Perturb normalized obs by epsilon * noise, measure action change
        eps = 0.01
        torch.manual_seed(42)
        noise_vec = torch.randn_like(obs_norm[:32])
        noise_vec = noise_vec / noise_vec.reshape(32, -1).norm(dim=-1).view(32, 1, 1)  # unit norm per sample
        obs_perturbed = obs_norm[:32] + eps * noise_vec
        with torch.no_grad():
            act_base = model.sample(obs_norm[:32])[:32]
            act_pert = model.sample(obs_perturbed)[:32]
        delta_per_unit = torch.norm(act_pert - act_base, dim=-1).mean() / eps
        print(f"    obs sensitivity (||Δact||/||Δobs|| at eps=0.01): {delta_per_unit:.4f} (action unit per norm obs unit)")

        results[name] = {
            "normal_vs_shuffled_all16_l2": {"mean": float(normal_vs_shuffled.mean()), "median": float(normal_vs_shuffled.median())},
            "normal_vs_shuffled_exec8_l2": {"mean": float(normal_vs_shuffled_exec8.mean()), "median": float(normal_vs_shuffled_exec8.median())},
            "normal_vs_zero_all16_l2": {"mean": float(normal_vs_zero.mean()), "median": float(normal_vs_zero.median())},
            "normal_vs_zero_exec8_l2": {"mean": float(normal_vs_zero_exec8.mean()), "median": float(normal_vs_zero_exec8.median())},
            "obs_sensitivity_per_unit": float(delta_per_unit),
        }

    # Comparative analysis
    b3_sens_shuffled = results["b3"]["normal_vs_shuffled_exec8_l2"]["mean"]
    uwm_sens_shuffled = results["uwm"]["normal_vs_shuffled_exec8_l2"]["mean"]
    b3_sens_zero = results["b3"]["normal_vs_zero_exec8_l2"]["mean"]
    uwm_sens_zero = results["uwm"]["normal_vs_zero_exec8_l2"]["mean"]
    b3_sens = results["b3"]["obs_sensitivity_per_unit"]
    uwm_sens = results["uwm"]["obs_sensitivity_per_unit"]

    print(f"\n  Analysis:")
    print(f"    shuffled_sensitivity ratio (UWM/B3): {uwm_sens_shuffled / max(b3_sens_shuffled, 1e-8):.2f}")
    print(f"    zero_sensitivity ratio (UWM/B3):     {uwm_sens_zero / max(b3_sens_zero, 1e-8):.2f}")
    print(f"    FD sensitivity ratio (UWM/B3):        {uwm_sens / max(b3_sens, 1e-8):.2f}")

    with open("outputs/e2_b3_vs_uwm/e2_obs_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def e2_3_contract_diff(models):
    """Corrected policy contract diff table."""
    print("=" * 60)
    print("E2-3: Corrected Policy Contract Diff")
    print("=" * 60)

    b3_model, _, _ = models["b3"]
    uwm_model, _, _ = models["uwm"]

    b3_params = sum(p.numel() for p in b3_model.parameters())
    uwm_params = sum(p.numel() for p in uwm_model.parameters())

    table = {
        "observations": [
            {"item": "backbone", "b3": "nn.TransformerEncoder (6L/256E/8H)", "uwm": "TransformerNoisePredictionNet (12L/768E/12H)", "same": False, "risk": "High"},
            {"item": "parameters", "b3": f"{b3_params/1e6:.1f}M", "uwm": f"{uwm_params/1e6:.1f}M", "same": False, "risk": "Medium"},
            {"item": "obs_dim", "b3": "20", "uwm": "20", "same": True, "risk": "Low"},
            {"item": "action_dim", "b3": "2", "uwm": "2", "same": True, "risk": "Low"},
            {"item": "pred_horizon", "b3": "16", "uwm": "16", "same": True, "risk": "Low"},
            {"item": "obs_horizon", "b3": "2", "uwm": "2", "same": True, "risk": "Low"},
            {"item": "eval_exec_steps", "b3": "8", "uwm": "8", "same": True, "risk": "Low"},
            {"item": "attention causal?", "b3": "False", "uwm": "False (verified)", "same": True, "risk": "RULED OUT"},
            {"item": "conditioning type", "b3": "obs-as-token concat (shared self-attention)", "uwm": "AdaLN scale+shift modulation (obs → global_cond)", "same": False, "risk": "HIGH"},
            {"item": "token layout", "b3": "[obs_token, a1..a16] (17 tokens)", "uwm": "[a1..a16] (16 tokens), obs as modulation", "same": False, "risk": "HIGH"},
            {"item": "obs encoder output", "b3": "Linear(40→256→256) + unsqueeze", "uwm": "Linear(40→768→768), no unsqueeze → global_cond", "same": False, "risk": "Medium"},
            {"item": "global cond dim", "b3": "256 (concatenated as token, not separate)", "uwm": "768 (separate modulation pathway)", "same": False, "risk": "Medium"},
            {"item": "positional embedding", "b3": "randn(1, 16, 256) on action tokens only", "uwm": "normal_(1, 16, 768) on action tokens only", "same": "Similar", "risk": "Low"},
            {"item": "registers", "b3": "None", "uwm": "None", "same": True, "risk": "Low"},
            {"item": "time embedding", "b3": "hand-coded sinusoidal(256) + MLP(256→256)", "uwm": "SinusoidalPosEmb(256) + MLP(256→1024→256)", "same": "Similar", "risk": "Low"},
            {"item": "scheduler type", "b3": "DDPMScheduler", "uwm": "DDPMScheduler", "same": True, "risk": "Low"},
            {"item": "train timesteps", "b3": "100", "uwm": "100", "same": True, "risk": "Low"},
            {"item": "inf steps", "b3": "10", "uwm": "10", "same": True, "risk": "Low"},
            {"item": "beta_schedule", "b3": "squaredcos_cap_v2", "uwm": "squaredcos_cap_v2", "same": True, "risk": "Low"},
            {"item": "clip_sample", "b3": "True", "uwm": "True", "same": True, "risk": "Low"},
            {"item": "eval action slicing", "b3": "t1:t9 (exec_start=1)", "uwm": "t1:t9 (exec_start=1)", "same": True, "risk": "Low"},
            {"item": "action normalizer", "b3": "MinMax [-1,1]", "uwm": "MinMax [-1,1]", "same": True, "risk": "Low"},
            {"item": "obs normalizer", "b3": "MinMax [-1,1]", "uwm": "MinMax [-1,1]", "same": True, "risk": "Low"},
        ]
    }

    # Print
    print(f"\n{'Item':<28} {'B3 DiT':<45} {'UWM-DP-KP':<45} {'Same?':<7} {'Risk':<12}")
    print("-" * 145)
    for obs in table["observations"]:
        same_str = "Yes" if obs["same"] else "No"
        print(f"  {obs['item']:<26}  {obs['b3']:<43}  {obs['uwm']:<43}  {same_str:<5}  {obs['risk']:<10}")

    # Count
    n_same = sum(1 for o in table["observations"] if o["same"])
    n_diff = sum(1 for o in table["observations"] if not o["same"])
    n_high = sum(1 for o in table["observations"] if o["risk"] == "HIGH")
    print(f"\n  Summary: {n_same} identical, {n_diff} different, {n_high} HIGH-risk differences")

    with open("outputs/e2_b3_vs_uwm/e2_contract_diff.md", "w") as f:
        f.write("# E2-3: Corrected Policy Contract Diff Table\n\n")
        f.write("**Correction**: Previous \"causal attention\" diagnosis was WRONG. ")
        f.write("Both B3 and UWM-DP-KP use bidirectional (non-causal) self-attention ")
        f.write("(verified: TransformerNoisePredictionNet has is_causal=False in all 12 blocks; ")
        f.write("nn.TransformerEncoder defaults to is_causal=False).\n\n")
        f.write(f"**Summary**: {n_same} identical, {n_diff} different, {n_high} HIGH-risk differences.\n\n")
        f.write(f"| Item | B3 DiT | UWM-DP-KP | Same? | Risk |\n")
        f.write(f"|------|--------|------------|-------|------|\n")
        for obs in table["observations"]:
            same_str = "Yes" if obs["same"] else "No"
            f.write(f"| {obs['item']} | {obs['b3']} | {obs['uwm']} | {same_str} | {obs['risk']} |\n")
        f.write("\n## HIGH Risk Items\n\n")
        f.write("1. **Conditioning type**: B3 uses obs-as-token concat (obs participates in self-attention), ")
        f.write("UWM uses AdaLN modulation (obs controls via scale+shift). This is the most fundamental architectural difference.\n")
        f.write("2. **Token layout**: B3 has 17 tokens [obs, a1..a16] sharing attention; UWM has 16 tokens ")
        f.write("[a1..a16] with obs providing external modulation.\n")

    print(f"\n  saved: outputs/e2_b3_vs_uwm/e2_contract_diff.md")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip", nargs="+", default=[])
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs("outputs/e2_b3_vs_uwm", exist_ok=True)

    print("=" * 60)
    print("E2 Diagnostics v2: B3 vs UWM-DP-KP")
    print("  Key correction: causal attention IS NOT the gap (both bidirectional)")
    print("  Focus: conditioning method (obs-as-token vs AdaLN)")
    print("=" * 60)

    # Load models
    print("\n[Loading models...]")
    models = load_models(device)
    print("  B3 loaded")
    print(f"  UWM-DP-KP loaded")
    print(f"  B3 causal? nn.TransformerEncoder (PyTorch builtin, no is_causal param)")
    print(f"  UWM blocks[0].attn.is_causal = {models['uwm'][0].noise_pred_net.blocks[0].attn.is_causal}")

    # E2-0
    if "e20" not in args.skip:
        e2_0_offline_mse(models, device)
    else:
        print("  [skip E2-0]")

    # E2-1
    if "e21" not in args.skip:
        e2_1_action_parity(models, device)
    else:
        print("  [skip E2-1]")

    # E2-2
    if "e22" not in args.skip:
        e2_2_resampling_variance(models, device)
    else:
        print("  [skip E2-2]")

    # E2-2.5
    if "e225" not in args.skip:
        e2_25_obs_sensitivity(models, device)
    else:
        print("  [skip E2-2.5]")

    # E2-3
    if "e23" not in args.skip:
        e2_3_contract_diff(models)
    else:
        print("  [skip E2-3]")

    print(f"\n{'='*60}")
    print("All E2 diagnostics complete.")
    print(f"  outputs/e2_b3_vs_uwm/*")


if __name__ == "__main__":
    main()
