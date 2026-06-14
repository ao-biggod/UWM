#!/usr/bin/env python3
"""E2: Diagnose gap between B3 (0.790) and UWM-DP-KP (0.397) on same 20D keypoint obs.

E2-0: Standardized offline MSE
E2-1: Same-obs action parity (B3 vs UWM-DP-KP first-action comparison)
E2-2: UWM-DP-KP same-state resampling variance
E2-3: Architecture diff table
"""
import sys, os, torch, argparse, numpy as np, json, time
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_zarr_data_keypoint(zarr_path):
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


def build_dataset(episodes, n_ep, seq_len, obs_h=2, act_h=16):
    all_obs, all_act = [], []
    for ep_obs, ep_act in episodes[:n_ep]:
        T = len(ep_obs)
        for i in range(T - seq_len):
            all_obs.append(ep_obs[i:i + obs_h])
            all_act.append(ep_act[i:i + seq_len])
    return torch.utils.data.TensorDataset(
        torch.tensor(np.stack(all_obs), dtype=torch.float32),
        torch.tensor(np.stack(all_act), dtype=torch.float32))


class MinMaxNorm:
    def __init__(self): self.off = self.scl = None
    def fit(self, d):
        mn, mx = d.min(0), d.max(0)
        self.off = (mx + mn) / 2.0
        self.scl = (mx - mn) / 2.0
        self.scl[self.scl < 1e-6] = 1.0
    def norm(self, x): return (x - self.off) / self.scl
    def unnorm(self, x): return x * self.scl + self.off


def load_b3_model(ckpt_path, device):
    import importlib.util
    spec = importlib.util.spec_from_file_location("stepB",
        str(Path(__file__).resolve().parent / "stepB_retrain_lowdim.py"))
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    model = sb.LowdimStatePolicyV2(obs_dim=20, clip_sample=True, num_inference_steps=10).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    an = ckpt["action_normalizer"]
    sn = ckpt["state_normalizer"]
    return model, sn, an


def load_uwm_model(ckpt_path, device):
    from models.dp.transformer import TransformerNoisePredictionNet
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    class LowdimObsEncoder(torch.nn.Module):
        def __init__(self, obs_dim, num_frames, embed_dim):
            super().__init__()
            in_dim = obs_dim * num_frames
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, embed_dim), torch.nn.Mish(),
                torch.nn.Linear(embed_dim, embed_dim))
        def forward(self, obs):
            B, T, D = obs.shape
            return self.net(obs.reshape(B, T * D))

    class LowdimDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            embed_dim = 768
            self.obs_encoder = LowdimObsEncoder(20, 2, embed_dim)
            self.noise_pred_net = TransformerNoisePredictionNet(
                input_len=16, input_dim=2, global_cond_dim=embed_dim,
                timestep_embed_dim=256, embed_dim=embed_dim,
                depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
            self.noise_sch = DDPMScheduler(
                num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                clip_sample=True, prediction_type="epsilon")
            self.num_inf_steps = 10
        def sample(self, obs):
            B, d = obs.shape[0], obs.device
            oe = self.obs_encoder(obs)
            act = torch.randn(B, 16, 2, device=d)
            self.noise_sch.set_timesteps(self.num_inf_steps)
            for ts in self.noise_sch.timesteps:
                t = torch.full((B,), ts, device=d, dtype=torch.long)
                npred = self.noise_pred_net(act, t, global_cond=oe)
                act = self.noise_sch.step(npred, ts, act).prev_sample
            return act

    model = LowdimDiT().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sn = ckpt["state_normalizer"]
    an = ckpt["action_normalizer"]
    return model, sn, an


def standardized_offline_mse(model, sn, an, dataset, device, label, obs_h=2, act_h=16):
    """E2-0: Standardized offline MSE with consistent formula."""
    model.eval()
    obs_test, act_test = dataset[:64]
    s_off = torch.tensor(sn["offset"], device=device).float()
    s_scl = torch.tensor(sn["scale"], device=device).float()
    a_off = torch.tensor(an["offset"], device=device).float()
    a_scl = torch.tensor(an["scale"], device=device).float()

    obs_n = (obs_test.to(device) - s_off) / s_scl

    with torch.no_grad():
        pred_n = model.sample(obs_n)
    pred_raw = pred_n * a_scl + a_off
    gt_raw = act_test[:, obs_h - 1: obs_h - 1 + act_h].to(device)

    raw_mse_all = ((pred_raw - gt_raw)**2).mean().item()
    raw_mse_8 = ((pred_raw[:, :8] - gt_raw[:, :8])**2).mean().item()
    t0 = ((pred_raw[:, 0] - gt_raw[:, 0])**2).mean().item()
    t7 = ((pred_raw[:, 7] - gt_raw[:, 7])**2).mean().item()
    t15 = ((pred_raw[:, 15] - gt_raw[:, 15])**2).mean().item()

    # Expert action std for normalization
    all_act = torch.cat([d[1][:, obs_h-1:obs_h-1+act_h] for d in dataset], dim=0)
    act_std = all_act.std(dim=(0,1)).numpy()  # scalar per dim
    if act_std.ndim == 0:
        act_std = np.array([act_std, act_std])
    elif act_std.shape[0] < 2:
        act_std = np.array([act_std[0], act_std[0]])

    print(f"\n  [{label}] N=64 offline MSE:")
    print(f"    raw_mse_all:  {raw_mse_all:.1f}")
    print(f"    raw_mse_8:    {raw_mse_8:.1f}")
    print(f"    t=0:          {t0:.1f}")
    print(f"    t=7:          {t7:.1f}")
    print(f"    t=15:         {t15:.1f}")
    print(f"    RMSE/expert_std: {np.sqrt(raw_mse_all)/act_std.mean():.3f}")

    return {"raw_mse_all": raw_mse_all, "raw_mse_8": raw_mse_8, "t0": t0, "t7": t7, "t15": t15}


def same_obs_parity(b3_model, uwm_model, b3_sn, bw3_an, uwm_sn, uwm_an, dataset, device, n_samples=128):
    """E2-1: Feed same obs to both models, compare first-action predictions."""
    print(f"\n{'='*60}")
    print("E2-1: Same-obs action parity (B3 vs UWM-DP-KP)")
    print(f"{'='*60}")

    obs_test, _ = dataset[:n_samples]
    b3_s_off = torch.tensor(b3_sn["offset"], device=device).float()
    b3_s_scl = torch.tensor(b3_sn["scale"], device=device).float()
    b3_a_off = torch.tensor(bw3_an["offset"], device=device).float()
    b3_a_scl = torch.tensor(bw3_an["scale"], device=device).float()
    u_s_off = torch.tensor(uwm_sn["offset"], device=device).float()
    u_s_scl = torch.tensor(uwm_sn["scale"], device=device).float()
    u_a_off = torch.tensor(uwm_an["offset"], device=device).float()
    u_a_scl = torch.tensor(uwm_an["scale"], device=device).float()

    obs_b3 = (obs_test.to(device) - b3_s_off) / b3_s_scl
    obs_u = (obs_test.to(device) - u_s_off) / u_s_scl

    with torch.no_grad():
        b3_pred_n = b3_model.sample(obs_b3)
        u_pred_n = uwm_model.sample(obs_u)

    b3_raw = (b3_pred_n * b3_a_scl + b3_a_off).cpu().numpy()
    u_raw = (u_pred_n * u_a_scl + u_a_off).cpu().numpy()

    b3_first = b3_raw[:, 0, :]
    u_first = u_raw[:, 0, :]
    l2_diffs = np.linalg.norm(b3_first - u_first, axis=-1)

    print(f"  B3 first-action mean: [{b3_first[:,0].mean():.1f}, {b3_first[:,1].mean():.1f}]")
    print(f"  UWM first-action mean: [{u_first[:,0].mean():.1f}, {u_first[:,1].mean():.1f}]")
    print(f"  L2 diff: mean={l2_diffs.mean():.1f} max={l2_diffs.max():.1f} median={np.median(l2_diffs):.1f}")

    # Also compare full 16-step predictions
    l2_full = np.linalg.norm(b3_raw - u_raw, axis=-1).mean(axis=-1)
    print(f"  Full-seq L2 diff (per-timestep mean): mean={l2_full.mean():.1f} max={l2_full.max():.1f}")

    return {"first_action_l2_mean": float(l2_diffs.mean()), "first_action_l2_max": float(l2_diffs.max())}


def uwm_sampling_variance(uwm_model, uwm_sn, uwm_an, dataset, device, K=32):
    """E2-2: Same-state resampling variance for UWM-DP-KP."""
    print(f"\n{'='*60}")
    print("E2-2: UWM-DP-KP same-state resampling variance")
    print(f"{'='*60}")

    obs_test, _ = dataset[:128]
    s_off = torch.tensor(uwm_sn["offset"], device=device).float()
    s_scl = torch.tensor(uwm_sn["scale"], device=device).float()
    a_off = torch.tensor(uwm_an["offset"], device=device).float()
    a_scl = torch.tensor(uwm_an["scale"], device=device).float()

    obs_n = (obs_test.to(device) - s_off) / s_scl

    all_samples = []
    for k in range(K):
        with torch.no_grad():
            pred_n = uwm_model.sample(obs_n)
        pred_raw = (pred_n * a_scl + a_off).cpu().numpy()
        all_samples.append(pred_raw)

    all_raw = np.stack(all_samples)  # [K, 128, 16, 2]

    first_actions = all_raw[:, :, 0, :]  # [K, 128, 2]
    first_action_std = first_actions.std(axis=0).mean(axis=0)  # avg over 128 obs
    first_action_l2_std = np.linalg.norm(first_actions - first_actions.mean(axis=0), axis=-1).std(axis=0).mean()
    action_seq_std = all_raw.std(axis=0).mean()

    # Within-plan delta
    deltas = []
    for k in range(K):
        for n in range(128):
            d = all_raw[k, n, 1:8] - all_raw[k, n, 0:7]
            deltas.append(np.linalg.norm(d, axis=-1))
    deltas = np.concatenate(deltas)

    print(f"  first_action_std:        [{first_action_std[0]:.2f}, {first_action_std[1]:.2f}]")
    print(f"  first_action_l2_std:     {first_action_l2_std:.2f}")
    print(f"  action_seq_std_mean:     {action_seq_std:.2f}")
    print(f"  plan_delta_mean:         {deltas.mean():.2f}")
    print(f"  plan_delta_max:          {deltas.max():.2f}")

    # Compare with B3 (from D1: first_action_std=[4.09, 4.31], l2_std=2.93, seq_std_mean=11.58)
    print(f"\n  Reference (B3 from D1-1):")
    print(f"    first_action_std=[4.09, 4.31], l2_std=2.93, seq_std_mean=11.58, delta_mean=8.67")

    return {"first_action_std": first_action_std.tolist()}


def architecture_diff_table():
    """E2-3: Detailed architecture comparison."""
    print(f"\n{'='*60}")
    print("E2-3: Architecture / Policy Contract Diff Table")
    print(f"{'='*60}")

    table = """
| Item | B3 DiT | UWM-DP-KP | Same? |
|------|--------|-----------|-------|
| **Model class** | LowdimStatePolicyV2 | custom LowdimDiT | NO |
| **Backbone** | nn.TransformerEncoder (6L, 256E, 8H) | TransformerNoisePredictionNet (12L, 768E, 12H) | NO |
| **Params** | ~5M | ~150M | NO |
| **Obs encoder** | obs_proj: Linear(40→256) → Mish → Linear(256→256) | LowdimObsEncoder: Linear(40→768) → Mish → Linear(768→768) | NO |
| **Obs dim** | 20 (×2 frames = 40 in) | 20 (×2 frames = 40 in) | YES |
| **Action dim** | 2 | 2 | YES |
| **Action len** | 16 | 16 | YES |
| **Timestep embed** | sinusoidal (half=128, →256 via time_mlp) | sinusoidal → timestep_encoder (SinusoidalPosEmb → MLP) | DIFFERENT |
| **Conditioning** | obs_feat + action_embed + time_emb → concat as tokens | global_cond (obs_embed) + temb → cat as AdaLN condition | **DIFFERENT** |
| **Architecture** | Standard TransformerEncoder (self-attn + FFN) | AdaLN-Attention (DiT-style with scale+shift modulation) | **DIFFERENT** |
| **Attention** | full self-attention (no causal mask in nn.TransformerEncoder) | causal_attn=True (autoregressive) | **DIFFERENT** |
| **Pos embed** | Learned pos_embed (action_len=16, embed_dim=256) | Learned pos_embed (action_len=16, embed_dim=768) | DIFFERENT (size) |
| **Noise scheduler** | DDPMScheduler(100 steps, squaredcos_cap_v2, clip_sample=True) | DDPMScheduler(100 steps, squaredcos_cap_v2, clip_sample=True) | YES |
| **Inference steps** | 10 | 10 | YES |
| **Normalizer** | MinMaxNormalizer (custom, [-1,1]) | MinMaxNormalizer (custom, [-1,1]) | YES |
| **Weight init** | Default PyTorch | init_weights + DiT-specific zero-init for adaLN and output | DIFFERENT |
| **Action target** | action[:, obs_horizon-1 : obs_horizon-1+action_horizon] | Same | YES |
| **Obs input** | obs[:, :obs_horizon] (2 frames of 20D) | Same | YES |
| **Dropout** | None (TransformerEncoder default) | p_drop_attn=0.01 | DIFFERENT |

Key suspicions (ordered by likelihood):
1. **Conditioning method**: B3 concats obs as a token; UWM uses obs via AdaLN scale+shift.
   AdaLN may struggle without rich visual features.
2. **Causal attention**: UWM uses causal_attn=True; B3 uses full bidirectional attention.
   Causal may hurt action-only prediction where future context can inform past.
3. **Weight init**: UWM uses DiT-specific zero-init for stability; B3 uses default init.
4. **Model size**: UWM (150M) may need more training steps to converge than B3 (5M).
"""
    print(table)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--b3-ckpt", type=str,
                        default="artifacts_keep/B3_keypoint20_local_dit_20k/latest.pt")
    parser.add_argument("--uwm-ckpt", type=str,
                        default="artifacts_keep/uwm_dp_only_keypoint20_20k/latest.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/e2_diag")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    dataset = build_dataset(episodes, 90, 19)

    # E2-0: Standardized offline MSE
    print("=" * 60)
    print("E2-0: STANDARDIZED OFFLINE MSE")
    print("=" * 60)

    b3, b3_sn, b3_an = load_b3_model(args.b3_ckpt, device)
    mse_b3 = standardized_offline_mse(b3, b3_sn, b3_an, dataset, device, "B3")

    uwm, uwm_sn, uwm_an = load_uwm_model(args.uwm_ckpt, device)
    mse_uwm = standardized_offline_mse(uwm, uwm_sn, uwm_an, dataset, device, "UWM-DP-KP")

    # E2-1: Same-obs parity
    parity = same_obs_parity(b3, uwm, b3_sn, b3_an, uwm_sn, uwm_an, dataset, device)

    # E2-2: UWM sampling variance
    uwm_var = uwm_sampling_variance(uwm, uwm_sn, uwm_an, dataset, device)

    # E2-3: Architecture diff
    arch_diff = architecture_diff_table()

    # Save
    with open(os.path.join(args.output_dir, "e2_summary.json"), "w") as f:
        json.dump({"e20_mse": {"b3": mse_b3, "uwm": mse_uwm},
                   "e21_parity": parity, "e22_variance": uwm_var}, f, indent=2)

    print(f"\nSaved: {args.output_dir}/e2_summary.json")


if __name__ == "__main__":
    main()
