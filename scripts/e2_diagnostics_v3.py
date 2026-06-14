#!/usr/bin/env python3
"""
E2 diagnostics v3: B3 vs UWM-DP-KP root-cause analysis.
Focus: conditioning method (obs-as-token vs AdaLN modulation).

E2-0: unified offline MSE  (fixed GT convention, seeded)
E2-1: same-obs action parity  (fixed GT convention, seeded)
E2-2: resampling variance  (unchanged from v2)
E2-2.5: obs sensitivity (fixed: same diffusion seed for all variants)
E2-3: corrected contract diff table

Key corrections from v2:
  1. GT alignment: gt16 = act_test[:, :16], no double-shift. exec_start=1.
  2. Obs sensitivity: same seed for normal/shuffled/zero/fd, removing sampling confound.
  3. E2-0/E2-1: fixed seed for reproducible action predictions.

Previously ruled out: causal attention. Both models are bidirectional (is_causal=False).
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

# --- Constants ---
OBS_HORIZON = 2
PRED_HORIZON = 16
N_ACTION_STEPS = 8
SEQ_LEN = 19  # dataset sliding window
EXEC_START = OBS_HORIZON - 1   # = 1
EXEC_END = EXEC_START + N_ACTION_STEPS  # = 9
DIAG_SEED = 202506  # base seed for diagnostic reproducibility

# --- Data loading ---
def load_zarr_data_keypoint(zarr_path):
    """Load 20D keypoint obs = [keypoint(9,2)_flat(18), agent_pos(2)]"""
    import zarr
    z = zarr.open(zarr_path, "r")
    keypoint = z["data/keypoint"][:]
    state = z["data/state"][:]
    action = z["data/action"][:]
    agent_pos = state[:, :2]
    obs_20d = np.concatenate([keypoint.reshape(keypoint.shape[0], -1), agent_pos], axis=-1)
    ep_ends = z["meta/episode_ends"][:]
    episodes = []
    start = 0
    for end in ep_ends:
        episodes.append((obs_20d[start:end], action[start:end]))
        start = end
    return episodes


def build_dataset(episodes, n_episodes, seq_len, obs_horizon=2):
    """Build dataset. Returns obs[T:2,20] and act[T:19,2] (sliding window)."""
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


# --- GT convention ---
def get_gt16(act_seq):
    """Extract 16-step ground truth target.

    act_seq: [B, 19, 2] = actions at timesteps [i ... i+18]
    Returns: gt16 = act_seq[:, :16] = [B, 16, 2] = actions [i ... i+15]

    Convention:
      gt16[:, 0]  = act[i]   — action at first obs frame (overlap, not executed)
      gt16[:, 1]  = act[i+1] — first EXECUTABLE action (after last obs frame)
      gt16[:, 1:9]= act[i+1 ... i+8] = 8 executed actions in rollout
      gt16[:, 1:17] would be the full 16-step future action horizon
    """
    return act_seq[:, :PRED_HORIZON]


# --- Normalization ---
def normalize_obs(obs, norm):
    return (obs - torch.tensor(norm.offset, device=obs.device).float()) / torch.tensor(norm.scale, device=obs.device).float()


def unnormalize_action(act, norm):
    return act * torch.tensor(norm.scale, device=act.device).float() + torch.tensor(norm.offset, device=act.device).float()


# --- Model building ---
def build_uwm_dit_policy(obs_dim, action_dim, action_len, device):
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


@torch.no_grad()
def sample_seeded(model, obs_norm, seed, model_name):
    """Sample with fixed diffusion noise seed. model_name is 'b3' or 'uwm'."""
    B, device = obs_norm.shape[0], obs_norm.device
    if model_name == "uwm":
        return model.sample(obs_norm, seed=seed)
    elif model_name == "b3":
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


# ======================================================================
# E2-0: Unified Offline MSE
# ======================================================================
def e2_0_offline_mse(models, device):
    print("=" * 60)
    print("E2-0: Unified Offline MSE  (seeded, gt16 = act[:,:16])")
    print("=" * 60)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    all_obs = np.concatenate([s for s, _ in episodes], axis=0)
    all_action = np.concatenate([a for _, a in episodes], axis=0)
    expert_action_std = float(all_action.std())
    print(f"  expert_action_std: {expert_action_std:.2f}")
    print(f"  N_expert: {len(all_action)}")

    dataset = build_dataset(episodes, 90, SEQ_LEN, OBS_HORIZON)
    test_offset = max(0, len(dataset) - 64)
    obs_test, act_test = dataset[test_offset:test_offset + 64]
    obs_test = obs_test.to(device)
    act_test = act_test.to(device)
    gt16 = get_gt16(act_test)  # [B, 16, 2]

    print(f"  GT convention: gt16 = act[:, :{PRED_HORIZON}]")
    print(f"  exec_start={EXEC_START}, exec_end={EXEC_END}")
    print(f"  gt16[:, 0] = act[i], gt16[:, 1:9] = act[i+1...i+8] (8 executed in rollout)")

    results = {}
    SEED = DIAG_SEED
    for name in ["b3", "uwm"]:
        model, n_state, n_action = models[name]
        obs_norm = normalize_obs(obs_test, n_state).float()

        act_pred_norm = sample_seeded(model, obs_norm, seed=SEED, model_name=name)
        act_pred_raw = unnormalize_action(act_pred_norm, n_action)

        all16 = F.mse_loss(act_pred_raw[:, :16], gt16[:, :16]).item()
        first8 = F.mse_loss(act_pred_raw[:, :8], gt16[:, :8]).item()
        exec8 = F.mse_loss(act_pred_raw[:, EXEC_START:EXEC_END], gt16[:, EXEC_START:EXEC_END]).item()
        t0 = F.mse_loss(act_pred_raw[:, 0], gt16[:, 0]).item()
        t7 = F.mse_loss(act_pred_raw[:, 7], gt16[:, 7]).item()
        t15 = F.mse_loss(act_pred_raw[:, 15], gt16[:, 15]).item()
        rmse_all = np.sqrt(all16)

        print(f"\n  [{name.upper()}] seed={SEED}")
        print(f"    all16_raw_mse (pred[:16] vs gt16):     {all16:.1f}")
        print(f"    first8_raw_mse (pred[:8] vs gt16[:8]): {first8:.1f}")
        print(f"    exec8_raw_mse  (pred[1:9] vs gt16[1:9]): {exec8:.1f}")
        print(f"    t0_raw_mse  (pred[0] vs act[i]):        {t0:.1f}")
        print(f"    t7_raw_mse  (pred[7] vs act[i+7]):      {t7:.1f}")
        print(f"    t15_raw_mse (pred[15] vs act[i+15]):    {t15:.1f}")
        print(f"    all16_raw_rmse:                          {rmse_all:.1f}")
        print(f"    rmse/expert_std:                         {rmse_all / expert_action_std:.4f}")

        results[name] = {
            "all16_raw_mse": all16,
            "first8_raw_mse": first8,
            "exec8_raw_mse": exec8,
            "t0_raw_mse": t0,
            "t7_raw_mse": t7,
            "t15_raw_mse": t15,
            "all16_raw_rmse": rmse_all,
            "rmse_over_expert_std": rmse_all / expert_action_std,
        }

    with open("outputs/e2_b3_vs_uwm/e2_offline_mse.json", "w") as f:
        json.dump({"expert_action_std": expert_action_std, "gt_convention": "gt16 = act[:, :16]",
                   "exec_start": EXEC_START, "exec_end": EXEC_END, "seed": SEED, "results": results}, f, indent=2)
    print(f"\n  saved: outputs/e2_b3_vs_uwm/e2_offline_mse.json")
    return results


# ======================================================================
# E2-1: Same-Obs Action Parity
# ======================================================================
def e2_1_action_parity(models, device):
    print("=" * 60)
    print("E2-1: Same-Obs Action Parity  (seeded, gt16 = act[:,:16])")
    print("=" * 60)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    dataset = build_dataset(episodes, 90, SEQ_LEN, OBS_HORIZON)

    N = 128
    SEED = DIAG_SEED
    obs_test, act_test = dataset[:N]
    obs_test = obs_test.to(device)
    act_test = act_test.to(device)
    gt16 = get_gt16(act_test)

    print(f"  GT: gt16 = act[:, :16]  (N=128, seed={SEED})")
    print(f"  exec_start={EXEC_START}, exec_end={EXEC_END}")
    print(f"  first executable action = pred[:, {EXEC_START}] vs gt16[:, {EXEC_START}] = act[i+{EXEC_START}]")
    print(f"  exec8 = pred[:, {EXEC_START}:{EXEC_END}] vs gt16[:, {EXEC_START}:{EXEC_END}]")

    preds = {}
    for name in ["b3", "uwm"]:
        model, n_state, n_action = models[name]
        obs_norm = normalize_obs(obs_test, n_state).float()
        act_pred_raw = unnormalize_action(
            sample_seeded(model, obs_norm, seed=SEED, model_name=name), n_action)
        preds[name] = act_pred_raw

    # First executable action
    b3_first = preds["b3"][:, EXEC_START]
    uwm_first = preds["uwm"][:, EXEC_START]
    gt_first = gt16[:, EXEC_START]

    b3_vs_gt = torch.norm(b3_first - gt_first, dim=-1)
    uwm_vs_gt = torch.norm(uwm_first - gt_first, dim=-1)
    b3_vs_uwm = torch.norm(b3_first - uwm_first, dim=-1)

    print(f"\n  First executable action L2 (t={EXEC_START}):")
    print(f"    B3-vs-GT  mean/median/max:  {b3_vs_gt.mean():.1f} / {b3_vs_gt.median():.1f} / {b3_vs_gt.max():.1f}")
    print(f"    UWM-vs-GT mean/median/max:  {uwm_vs_gt.mean():.1f} / {uwm_vs_gt.median():.1f} / {uwm_vs_gt.max():.1f}")
    print(f"    B3-vs-UWM mean/median/max:  {b3_vs_uwm.mean():.1f} / {b3_vs_uwm.median():.1f} / {b3_vs_uwm.max():.1f}")

    # Exec8 L2
    b3_exec8 = preds["b3"][:, EXEC_START:EXEC_END]
    uwm_exec8 = preds["uwm"][:, EXEC_START:EXEC_END]
    gt_exec8 = gt16[:, EXEC_START:EXEC_END]
    l2_all_b3 = torch.norm(b3_exec8.reshape(N, -1) - gt_exec8.reshape(N, -1), dim=-1)
    l2_all_uwm = torch.norm(uwm_exec8.reshape(N, -1) - gt_exec8.reshape(N, -1), dim=-1)
    l2_b3_uwm = torch.norm(b3_exec8.reshape(N, -1) - uwm_exec8.reshape(N, -1), dim=-1)
    print(f"\n  Exec8 action L2 (t={EXEC_START}:{EXEC_END}):")
    print(f"    B3-vs-GT  mean/median/max:  {l2_all_b3.mean():.1f} / {l2_all_b3.median():.1f} / {l2_all_b3.max():.1f}")
    print(f"    UWM-vs-GT mean/median/max:  {l2_all_uwm.mean():.1f} / {l2_all_uwm.median():.1f} / {l2_all_uwm.max():.1f}")
    print(f"    B3-vs-UWM mean/median/max:  {l2_b3_uwm.mean():.1f} / {l2_b3_uwm.median():.1f} / {l2_b3_uwm.max():.1f}")

    agreement = (b3_vs_uwm < b3_vs_gt).float().mean()
    print(f"\n  Agreement (B3-UWM < B3-GT): {agreement*100:.1f}%")

    # Table: first 20
    print(f"\n  {'idx':<5} {'gt_exec_t1':<24} {'B3_exec_t1':<24} {'UWM_exec_t1':<24} {'B3-GT':<8} {'UWM-GT':<8} {'B3-UWM'}")
    print(f"  {'-'*113}")
    for i in range(min(20, N)):
        g0, g1 = gt_first[i, 0].item(), gt_first[i, 1].item()
        b0, b1 = b3_first[i, 0].item(), b3_first[i, 1].item()
        u0, u1 = uwm_first[i, 0].item(), uwm_first[i, 1].item()
        print(f"  {i:<5} [{g0:7.1f}, {g1:7.1f}]     [{b0:7.1f}, {b1:7.1f}]     "
              f"[{u0:7.1f}, {u1:7.1f}]     {b3_vs_gt[i]:4.1f}     {uwm_vs_gt[i]:4.1f}     {b3_vs_uwm[i]:4.1f}")

    results = {
        "gt_convention": "gt16 = act[:, :16]",
        "exec_slice": [EXEC_START, EXEC_END], "N": N, "seed": SEED,
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


# ======================================================================
# E2-2: Resampling Variance (unchanged logic, works fine)
# ======================================================================
def e2_2_resampling_variance(models, device):
    print("=" * 60)
    print("E2-2: Resampling Variance (K=32, multiple seeds)")
    print("=" * 60)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    dataset = build_dataset(episodes, 90, SEQ_LEN, OBS_HORIZON)

    N = 128
    K = 32
    obs_test, act_test = dataset[:N]
    obs_test = obs_test.to(device)

    all_results = {}
    for name in ["b3", "uwm"]:
        model, n_state, n_action = models[name]
        obs_norm = normalize_obs(obs_test, n_state).float()

        all_actions = []
        for k in range(K):
            seed = 10000 + k
            act_pred_raw = unnormalize_action(
                sample_seeded(model, obs_norm, seed=seed, model_name=name), n_action)
            all_actions.append(act_pred_raw.cpu().numpy())

        all_actions = np.stack(all_actions, axis=0)

        first_action_std_dim0 = np.stack([all_actions[:, i, EXEC_START, 0].std() for i in range(N)])
        first_action_std_L2 = np.array([np.sqrt(all_actions[:, i, EXEC_START, 0].var() + all_actions[:, i, EXEC_START, 1].var()) for i in range(N)])
        exec8_std = np.array([all_actions[:, i, EXEC_START:EXEC_END].std(axis=0).mean() for i in range(N)])
        all16_std = np.array([all_actions[:, i].std(axis=0).mean() for i in range(N)])

        plan_deltas = []
        for i in range(N):
            deltas = []
            for k in range(K - 1):
                d = np.sqrt(np.sum((all_actions[k+1, i, EXEC_START] - all_actions[k, i, EXEC_START])**2))
                deltas.append(d)
            plan_deltas.append(np.mean(deltas))
        plan_delta_mean = np.mean(plan_deltas)

        print(f"\n  [{name.upper()}] K={K}")
        print(f"    first_action_std (dim0):            {first_action_std_dim0.mean():.2f}")
        print(f"    first_action_std_L2 (t={EXEC_START}):   {first_action_std_L2.mean():.2f}")
        print(f"    exec8_action_std (mean):            {exec8_std.mean():.2f}")
        print(f"    all16_action_std (mean):            {all16_std.mean():.2f}")
        print(f"    plan_delta (consecutive seeds):     {plan_delta_mean:.2f}")

        all_results[name] = {
            "first_action_std_dim0": float(first_action_std_dim0.mean()),
            "first_action_std_L2": float(first_action_std_L2.mean()),
            "exec8_action_std": float(exec8_std.mean()),
            "all16_action_std": float(all16_std.mean()),
            "plan_delta": float(plan_delta_mean),
        }

    ratio_std = all_results["uwm"]["first_action_std_L2"] / max(all_results["b3"]["first_action_std_L2"], 1e-8)
    ratio_delta = all_results["uwm"]["plan_delta"] / max(all_results["b3"]["plan_delta"], 1e-8)
    all_results["uwm_vs_b3_ratio"] = {"first_action_std_L2": float(ratio_std), "plan_delta": float(ratio_delta)}
    print(f"\n  UWM/B3 ratio: std={ratio_std:.2f}, plan_delta={ratio_delta:.2f}")

    with open("outputs/e2_b3_vs_uwm/e2_resampling_variance.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


# ======================================================================
# E2-2.5: Obs Sensitivity (FIXED: same diffusion seed for all variants)
# ======================================================================
def e2_25_obs_sensitivity(models, device):
    print("=" * 60)
    print("E2-2.5: Obs Sensitivity (SAME diffusion seed for all obs variants)")
    print("=" * 60)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    dataset = build_dataset(episodes, 90, SEQ_LEN, OBS_HORIZON)

    N = 128
    SEED = DIAG_SEED
    obs_test, act_test = dataset[:N]
    obs_test = obs_test.to(device)

    results = {}

    for name in ["b3", "uwm"]:
        model, n_state, n_action = models[name]
        obs_norm = normalize_obs(obs_test, n_state).float()

        # 1) Normal obs
        act_normal = unnormalize_action(
            sample_seeded(model, obs_norm, seed=SEED, model_name=name), n_action)

        # 2) Shuffled obs (within batch, SAME seed)
        shuffle_idx = torch.randperm(N)
        obs_shuffled = obs_norm[shuffle_idx]
        act_shuffled = unnormalize_action(
            sample_seeded(model, obs_shuffled, seed=SEED, model_name=name), n_action)

        # 3) Zero obs (center of normalized range, SAME seed)
        obs_zero_norm = torch.zeros_like(obs_norm)
        act_zero = unnormalize_action(
            sample_seeded(model, obs_zero_norm, seed=SEED, model_name=name), n_action)

        # L2 differences (all16 + exec8)
        fmt = lambda t: float(t.reshape(N, -1).norm(dim=-1).mean())
        nv_s_all = fmt(act_normal - act_shuffled)
        nv_s_exec8 = fmt(act_normal[:, EXEC_START:EXEC_END] - act_shuffled[:, EXEC_START:EXEC_END])
        nv_z_all = fmt(act_normal - act_zero)
        nv_z_exec8 = fmt(act_normal[:, EXEC_START:EXEC_END] - act_zero[:, EXEC_START:EXEC_END])

        print(f"\n  [{name.upper()}] seed={SEED}")
        print(f"    normal_vs_shuffled all16 L2:   {nv_s_all:.1f}")
        print(f"    normal_vs_shuffled exec8 L2:    {nv_s_exec8:.1f}")
        print(f"    normal_vs_zero all16 L2:        {nv_z_all:.1f}")
        print(f"    normal_vs_zero exec8 L2:        {nv_z_exec8:.1f}")

        # FD sensitivity: small perturbation, SAME seed
        eps = 0.01
        torch.manual_seed(42)
        B_fd = 32
        noise_vec = torch.randn_like(obs_norm[:B_fd])
        noise_vec = noise_vec / noise_vec.reshape(B_fd, -1).norm(dim=-1).view(B_fd, 1, 1)
        obs_perturbed = obs_norm[:B_fd] + eps * noise_vec

        act_base = unnormalize_action(
            sample_seeded(model, obs_norm[:B_fd], seed=SEED, model_name=name), n_action)
        act_pert = unnormalize_action(
            sample_seeded(model, obs_perturbed, seed=SEED, model_name=name), n_action)
        delta_per_unit = torch.norm(act_pert - act_base, dim=-1).mean().item() / eps

        print(f"    FD sensitivity (||Δact||/||Δobs||): {delta_per_unit:.4f} (action unit per norm obs unit)")

        results[name] = {
            "normal_vs_shuffled_all16_l2": nv_s_all,
            "normal_vs_shuffled_exec8_l2": nv_s_exec8,
            "normal_vs_zero_all16_l2": nv_z_all,
            "normal_vs_zero_exec8_l2": nv_z_exec8,
            "obs_sensitivity_per_unit": delta_per_unit,
        }

    # Comparative analysis
    b3_s0 = results["b3"]["normal_vs_shuffled_exec8_l2"]
    uwm_s0 = results["uwm"]["normal_vs_shuffled_exec8_l2"]
    b3_sz = results["b3"]["normal_vs_zero_exec8_l2"]
    uwm_sz = results["uwm"]["normal_vs_zero_exec8_l2"]
    b3_fd = results["b3"]["obs_sensitivity_per_unit"]
    uwm_fd = results["uwm"]["obs_sensitivity_per_unit"]

    print(f"\n  Ratios (UWM/B3):")
    print(f"    shuffled_sensitivity: {uwm_s0 / max(b3_s0, 1e-8):.2f}")
    print(f"    zero_sensitivity:     {uwm_sz / max(b3_sz, 1e-8):.2f}")
    print(f"    FD sensitivity:       {uwm_fd / max(b3_fd, 1e-8):.2f}")

    with open("outputs/e2_b3_vs_uwm/e2_obs_sensitivity.json", "w") as f:
        json.dump({"seed": SEED, "eps": 0.01, "results": results}, f, indent=2)
    return results


# ======================================================================
# E2-3: Corrected Contract Diff Table
# ======================================================================
def e2_3_contract_diff(models):
    print("=" * 60)
    print("E2-3: Corrected Policy Contract Diff")
    print("=" * 60)

    b3_model, _, _ = models["b3"]
    uwm_model, _, _ = models["uwm"]
    b3_params = sum(p.numel() for p in b3_model.parameters())
    uwm_params = sum(p.numel() for p in uwm_model.parameters())

    rows = [
        ("backbone", "nn.TransformerEncoder (6L/256E/8H)", "TransformerNoisePredictionNet (12L/768E/12H)", False, "High"),
        ("parameters", f"{b3_params/1e6:.1f}M", f"{uwm_params/1e6:.1f}M", False, "Medium"),
        ("obs_dim", "20", "20", True, "Low"),
        ("action_dim", "2", "2", True, "Low"),
        ("pred_horizon", "16", "16", True, "Low"),
        ("obs_horizon", "2", "2", True, "Low"),
        ("eval_exec_steps", "8", "8", True, "Low"),
        ("attention causal?", "False (builtin default)", "False (verified in all 12 blocks)", True, "RULED OUT"),
        ("conditioning type", "obs-as-token concat", "AdaLN scale+shift modulation", False, "HIGH"),
        ("token layout", "[obs_token, a1..a16] (17 tokens)", "[a1..a16] (16 tokens), obs → AdaLN gate", False, "HIGH"),
        ("obs encoder output", "Linear(40→256→256) + unsqueeze → token", "Linear(40→768→768), no unsqueeze → global_cond", False, "Medium"),
        ("global cond dim", "256 (as token in seq)", "768 (separate modulation path)", False, "Medium"),
        ("pos embedding", "randn(1, 16, 256) on action", "normal_(1, 16, 768) on action", True, "Low"),
        ("registers", "None", "None", True, "Low"),
        ("time embedding", "sinusoidal(256)+MLP(256→256)", "SinusoidalPosEmb(256)+MLP(256→1024→256)", True, "Low"),
        ("scheduler type", "DDPMScheduler", "DDPMScheduler", True, "Low"),
        ("train timesteps", "100", "100", True, "Low"),
        ("inf steps", "10", "10", True, "Low"),
        ("beta_schedule", "squaredcos_cap_v2", "squaredcos_cap_v2", True, "Low"),
        ("clip_sample", "True", "True", True, "Low"),
        ("eval action slicing", f"pred[:, {EXEC_START}:{EXEC_END}] → 8 steps", f"pred[:, :8] → 8 steps", True, "Low"),
        ("action normalizer", "MinMax [-1,1]", "MinMax [-1,1]", True, "Low"),
        ("obs normalizer", "MinMax [-1,1]", "MinMax [-1,1]", True, "Low"),
    ]

    print(f"\n{'Item':<28} {'B3 DiT':<45} {'UWM-DP-KP':<45} {'Same?':<7} {'Risk':<12}")
    print("-" * 145)
    for item, b3_val, uwm_val, same, risk in rows:
        print(f"  {item:<26}  {b3_val:<43}  {uwm_val:<43}  {'Yes' if same else 'No':<5}  {risk:<10}")

    n_same = sum(1 for r in rows if r[3])
    n_diff = sum(1 for r in rows if not r[3])
    n_high = sum(1 for r in rows if r[4] == "HIGH")
    print(f"\n  Summary: {n_same} identical, {n_diff} different, {n_high} HIGH-risk")

    with open("outputs/e2_b3_vs_uwm/e2_contract_diff.md", "w") as f:
        f.write("# E2-3: Corrected Policy Contract Diff (v3)\n\n")
        f.write("**Correction**: causal attention ruled out. Both models use bidirectional (non-causal) self-attention.\n\n")
        f.write(f"**GT convention**: gt16 = act[:, :16]. exec_start={EXEC_START}, exec_end={EXEC_END}.\n\n")
        f.write(f"| Item | B3 DiT | UWM-DP-KP | Same? | Risk |\n")
        f.write(f"|------|--------|------------|-------|------|\n")
        for item, b3_val, uwm_val, same, risk in rows:
            f.write(f"| {item} | {b3_val} | {uwm_val} | {'Yes' if same else 'No'} | {risk} |\n")
        f.write(f"\n**Summary**: {n_same} identical, {n_diff} different, {n_high} HIGH-risk differences.\n")
        f.write("\n## HIGH Risk Items\n")
        f.write("1. **Conditioning type**: obs-as-token concat vs AdaLN modulation\n")
        f.write("2. **Token layout**: [obs, a1..a16] (17 tokens, obs in seq) vs [a1..a16] (16 tokens, obs external)\n")

    print(f"\n  saved: outputs/e2_b3_vs_uwm/e2_contract_diff.md")
    return rows


# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip", nargs="+", default=[])
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs("outputs/e2_b3_vs_uwm", exist_ok=True)

    print("=" * 60)
    print(f"E2 Diagnostics v3: B3 vs UWM-DP-KP")
    print(f"  GT convention: gt16 = act[:, :16], exec_start={EXEC_START}, exec_end={EXEC_END}")
    print(f"  All predictions: seeded (seed={DIAG_SEED})")
    print(f"  Obs sensitivity: SAME seed for normal/shuffled/zero/fd")
    print(f"  causal attention: RULED OUT (both bidirectional)")
    print("=" * 60)

    print("\n[Loading models...]")
    models = load_models(device)
    print("  B3 loaded")
    print("  UWM-DP-KP loaded")

    if "e20" not in args.skip:
        e2_0_offline_mse(models, device)
    else:
        print("  [skip E2-0]")

    if "e21" not in args.skip:
        e2_1_action_parity(models, device)
    else:
        print("  [skip E2-1]")

    if "e22" not in args.skip:
        e2_2_resampling_variance(models, device)
    else:
        print("  [skip E2-2]")

    if "e225" not in args.skip:
        e2_25_obs_sensitivity(models, device)
    else:
        print("  [skip E2-2.5]")

    if "e23" not in args.skip:
        e2_3_contract_diff(models)
    else:
        print("  [skip E2-3]")

    print(f"\n{'='*60}")
    print("All E2 diagnostics complete. outputs/e2_b3_vs_uwm/*")


if __name__ == "__main__":
    main()
