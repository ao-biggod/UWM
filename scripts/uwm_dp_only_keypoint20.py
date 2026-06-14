#!/usr/bin/env python3
"""UWM DP-only + 20D keypoint: keep UWM transformer backbone, use keypoint obs.

Matches B3's obs contract (parity with DP official PushTLowdimDataset).
Uses the UWM TransformerNoisePredictionNet backbone, no image encoder.
No video/dynamics loss. No EMA, no cosine.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


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


def build_dataset_kp(episodes, n_episodes, seq_len, obs_horizon=2, action_horizon=16):
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


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


class MinMaxNormalizer:
    def __init__(self):
        self.offset = None
        self.scale = None
    def fit(self, data):
        dmin = data.min(axis=0)
        dmax = data.max(axis=0)
        self.offset = (dmax + dmin) / 2.0
        self.scale = (dmax - dmin) / 2.0
        self.scale[self.scale < 1e-6] = 1.0
        return self
    def normalize(self, data):
        return (data - self.offset) / self.scale
    def unnormalize(self, data):
        return data * self.scale + self.offset


def build_uwm_dit_policy(obs_dim, action_dim, action_len, device):
    """UWM TransformerNoisePredictionNet based DiT policy, adapted for lowdim obs.

    Same backbone as UWM DP-only but using simple MLP obs encoder instead of ResNet."""
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
            obs_embed = self.obs_encoder(obs)  # (B, E)
            noise = torch.randn_like(action)
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=action.device).long()
            noisy_action = self.noise_scheduler.add_noise(action, noise, t)
            noise_pred = self.noise_pred_net(noisy_action, t, global_cond=obs_embed)
            return F.mse_loss(noise_pred, noise)

        @torch.no_grad()
        def sample(self, obs):
            B, device = obs.shape[0], obs.device
            obs_embed = self.obs_encoder(obs)  # (B, E)
            action = torch.randn(B, self.action_len, self.action_dim, device=device)
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for t_step in self.noise_scheduler.timesteps:
                t = torch.full((B,), t_step, device=device, dtype=torch.long)
                noise_pred = self.noise_pred_net(action, t, global_cond=obs_embed)
                action = self.noise_scheduler.step(noise_pred, t_step, action).prev_sample
            return action

    return LowdimDiTPolicy(obs_dim=obs_dim, action_len=action_len, action_dim=action_dim).to(device)


def eval_fixed_buffer(model, norm_state, norm_action, device, n_eps=50):
    """Fixed-buffer eval with PushTKeypointsEnv, 20D keypoint obs."""
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    model.eval()
    results = []
    for ep in range(n_eps):
        seed = 100000 + ep
        env = MultiStepWrapper(
            PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0),
            n_obs_steps=2, n_action_steps=8)
        env.seed(seed)
        raw_obs = env.reset()

        obs_buffer = deque(maxlen=2)
        Do = raw_obs.shape[-1] // 2
        obs_buffer.append(raw_obs[0, :Do])  # frame0 data
        obs_buffer.append(raw_obs[1, :Do])  # frame1 data
        rewards = []
        done = False
        step = 0

        while not done and step < 300:
            state_np = np.stack(list(obs_buffer))
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_norm = (state_t - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()

            with torch.no_grad():
                action_norm = model.sample(state_norm)[0]
            action_raw = action_norm * torch.tensor(norm_action.scale, device=device).float() + torch.tensor(norm_action.offset, device=device).float()
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
            exec_actions = action_raw[:8]

            raw_obs, reward, done, info = env.step(exec_actions)
            rewards.append(float(reward))
            done = bool(np.all(done))
            Do = raw_obs.shape[-1] // 2
            obs_buffer.append(raw_obs[1, :Do])
            step += 1

        max_r = float(max(rewards)) if rewards else 0.0
        results.append(max_r)
        if ep < 5 or ep % 10 == 0:
            print(f"  Ep {ep:3d}: max_reward={max_r:.4f}", flush=True)

    scores = np.array(results)
    print(f"\n  eval: mean={scores.mean():.4f} std={scores.std():.4f} median={np.median(scores):.4f} "
          f"ep>0.5={(scores>0.5).sum()}/{n_eps} ep>0.8={(scores>0.8).sum()}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str,
                        default="artifacts_keep/uwm_dp_only_keypoint20_20k")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    obs_horizon, action_horizon, seq_len = 2, 16, 19
    obs_dim = 20
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"UWM DP-only + 20D Keypoint")
    print(f"  Backbone: UWM TransformerNoisePredictionNet (12L/768E/12H)")
    print(f"  Obs: 20D keypoint (DP official aligned)")
    print(f"{'='*60}")

    # Data
    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_keypoint(zarr_path)
    all_obs = np.concatenate([s for s, _ in episodes[:90]], axis=0)
    all_action = np.concatenate([a for _, a in episodes[:90]], axis=0)

    norm_state = MinMaxNormalizer().fit(all_obs)
    norm_action = MinMaxNormalizer().fit(all_action)

    dataset = build_dataset_kp(episodes, 90, seq_len, obs_horizon, action_horizon)
    print(f"Training samples: {len(dataset)}")

    # Build model
    model = build_uwm_dit_policy(obs_dim=obs_dim, action_dim=2, action_len=16, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M")

    # Training
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    data_iter = iter(loader)
    t0 = time.time()

    for step in range(args.num_steps):
        try:
            obs_np, act_np = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            obs_np, act_np = next(data_iter)

        obs_state = (obs_np.to(device) - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()
        action_target = act_np[:, obs_horizon - 1: obs_horizon - 1 + action_horizon].to(device)
        action_norm = (action_target - torch.tensor(norm_action.offset, device=device).float()) / torch.tensor(norm_action.scale, device=device).float()

        model.train()
        loss = model(obs_state, action_norm)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0 or step < 5:
            elapsed = time.time() - t0
            print(f"  step {step:6d}: loss={loss.item():.6f}  {step / elapsed:.1f} s/s", flush=True)

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    # Offline MSE
    model.eval()
    obs_test, act_test = dataset[:64]
    obs_norm = (obs_test.to(device) - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()
    with torch.no_grad():
        act_pred_norm = model.sample(obs_norm)
    act_pred_raw = act_pred_norm * torch.tensor(norm_action.scale, device=device).float() + torch.tensor(norm_action.offset, device=device).float()
    act_gt_raw = act_test[:, obs_horizon - 1: obs_horizon - 1 + action_horizon].to(device)
    mse_all = F.mse_loss(act_pred_raw, act_gt_raw).item()
    mse_8 = F.mse_loss(act_pred_raw[:, :8], act_gt_raw[:, :8]).item()
    print(f"  offline MSE all16: {mse_all:.1f}  first8: {mse_8:.1f}")

    # Save
    torch.save({
        "model": model.state_dict(),
        "step": args.num_steps - 1,
        "action_normalizer": {"offset": norm_action.offset.tolist(), "scale": norm_action.scale.tolist()},
        "state_normalizer": {"offset": norm_state.offset.tolist(), "scale": norm_state.scale.tolist()},
        "config": {"obs_horizon": obs_horizon, "action_horizon": action_horizon, "obs_dim": obs_dim, "action_len": 16},
    }, os.path.join(args.output_dir, "latest.pt"))
    print(f"Saved: {args.output_dir}/latest.pt")

    # Eval
    if not args.skip_eval:
        print(f"\nEval: {args.n_eval_episodes} episodes")
        scores = eval_fixed_buffer(model, norm_state, norm_action, device, args.n_eval_episodes)
        summary = {"mean": float(scores.mean()), "std": float(scores.std()),
                   "median": float(np.median(scores)), "n_eps": args.n_eval_episodes,
                   "ep_gt_05": int((scores > 0.5).sum()), "ep_gt_08": int((scores > 0.8).sum()),
                   "offline_mse_all": mse_all, "offline_mse_first8": mse_8}
        with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Eval summary saved")


if __name__ == "__main__":
    main()
