#!/usr/bin/env python3
"""Combined Step 4 + Step 5: lowdim oracle + overfit sanity check.
- lowdim oracle: full 5D state → action diffusion → PushT env eval
- overfit: train on 1-5 demos, check if model can memorize
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))


class LowdimStatePolicy(torch.nn.Module):
    """Minimal DiT-based action diffusion with lowdim state input (5D)."""
    def __init__(self, obs_dim=5, action_len=16, action_dim=2, embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.obs_proj = torch.nn.Sequential(
            torch.nn.Linear(obs_dim * 2, embed_dim),  # 2 frames of full state
            torch.nn.Mish(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(256, embed_dim),
            torch.nn.Mish(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.action_embed = torch.nn.Linear(action_dim, embed_dim)
        self.action_decoder = torch.nn.Linear(embed_dim, action_dim)
        self.pos_embed = torch.nn.Parameter(torch.randn(1, action_len, embed_dim) * 0.02)
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon")
        self.num_inference_steps = 10

    def _time_emb(self, t, B, device):
        half = 256 // 2
        omega = torch.exp(-torch.arange(half, device=device).float() * (np.log(10000) / (half - 1)))
        arg = t.float().unsqueeze(1) * omega.unsqueeze(0)
        emb = torch.zeros(B, 256, device=device)
        emb[:, 0::2] = torch.sin(arg)
        emb[:, 1::2] = torch.cos(arg)
        return self.time_mlp(emb).unsqueeze(1)

    def forward(self, obs_state, action):
        B, device = action.shape[0], action.device
        obs_feat = self.obs_proj(obs_state.reshape(B, -1)).unsqueeze(1)  # (B,1,E)
        noise = torch.randn_like(action)
        t = torch.randint(0, 100, (B,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action, noise, t)
        temb = self._time_emb(t, B, device)
        act_emb = self.action_embed(noisy_action) + self.pos_embed
        x = torch.cat([obs_feat, act_emb], dim=1) + temb
        x = self.transformer(x)
        noise_pred = self.action_decoder(x[:, 1:])
        loss = F.mse_loss(noise_pred, noise)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def sample(self, obs_state):
        B, device = obs_state.shape[0], obs_state.device
        obs_feat = self.obs_proj(obs_state.reshape(B, -1)).unsqueeze(1)
        action = torch.randn(B, self.action_len, self.action_dim, device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t_step in self.noise_scheduler.timesteps:
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            temb = self._time_emb(t, B, device)
            act_emb = self.action_embed(action) + self.pos_embed
            x = torch.cat([obs_feat, act_emb], dim=1) + temb
            x = self.transformer(x)
            noise_pred = self.action_decoder(x[:, 1:])
            action = self.noise_scheduler.step(noise_pred, t_step, action).prev_sample
        return action


def load_zarr_data(zarr_path):
    """Load PushT zarr, return episodes of (state, action, img)."""
    import zarr
    z = zarr.open(zarr_path, "r")
    state = z["data/state"][:]   # (N, 5)
    action = z["data/action"][:]  # (N, 2)
    ep_ends = z["meta/episode_ends"][:]
    episodes = []
    start = 0
    for end in ep_ends:
        episodes.append((state[start:end], action[start:end]))
        start = end
    return episodes


def build_overfit_dataset(episodes, n_episodes, seq_len, obs_horizon=2, action_horizon=16):
    """Build dataset from first n_episodes with sliding windows."""
    all_obs = []
    all_act = []
    for ep_state, ep_action in episodes[:n_episodes]:
        T = len(ep_state)
        win = seq_len  # obs_horizon + action_horizon = 2 + 16 = 18, we use 19 for overlap
        for i in range(T - win):
            obs_win = ep_state[i:i+obs_horizon]  # (obs_horizon, 5)
            act_win = ep_action[i:i+win]          # (win, 2)
            all_obs.append(obs_win)
            all_act.append(act_win)
    obs_tensor = torch.tensor(np.stack(all_obs), dtype=torch.float32)
    act_tensor = torch.tensor(np.stack(all_act), dtype=torch.float32)
    return TensorDataset(obs_tensor, act_tensor)


def collate_fn(batch):
    obs = torch.stack([b[0] for b in batch])
    act = torch.stack([b[1] for b in batch])
    return obs, act


def compute_normalizer(episodes, n_episodes):
    all_state = np.concatenate([s for s, _ in episodes[:n_episodes]], axis=0)
    all_action = np.concatenate([a for _, a in episodes[:n_episodes]], axis=0)
    return {
        "state_mean": all_state.mean(axis=0),
        "state_std": all_state.std(axis=0) + 1e-6,
        "action_mean": all_action.mean(axis=0),
        "action_std": all_action.std(axis=0) + 1e-6,
    }


def run_pushT_eval_lowdim(model, norm, device, n_episodes=50):
    """Run PushT env eval using FULL 5D state as input."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)

    s_mean = torch.tensor(norm["state_mean"], device=device).float()
    s_std = torch.tensor(norm["state_std"], device=device).float()
    a_mean = torch.tensor(norm["action_mean"], device=device).float()
    a_std = torch.tensor(norm["action_std"], device=device).float()

    results = []
    for ep in range(n_episodes):
        env.seed(ep)
        obs = env.reset()
        rewards = []
        done = False
        step = 0
        state_buffer = []  # accumulate state for obs_horizon=2

        while not done and step < 300:
            # Get full state from env internals (PushTImageEnv extends PushTEnv)
            inner = env.env
            agent_p = np.array(inner.agent.position)  # (2,)
            block_p = np.array(inner.block.position)  # (2,)
            block_ang = inner.block.angle
            full_state = np.concatenate([agent_p, block_p, [block_ang]])  # (5,)
            state_buffer.append(full_state)
            if len(state_buffer) > 2:
                state_buffer = state_buffer[-2:]

            if len(state_buffer) < 2:
                obs, reward, done, info = env.step(np.zeros((8, 2)))
                rewards.append(float(reward))
                step += 1
                continue

            state_np = np.stack(state_buffer)  # (2, 5)
            state_tensor = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_norm = (state_tensor - s_mean) / s_std

            with torch.no_grad():
                action_norm = model.sample(state_norm)[0]
            action_raw = action_norm * a_std + a_mean
            action_exec = action_raw[:8].cpu().numpy()
            obs, reward, done, info = env.step(action_exec)
            rewards.append(float(reward))
            step += 1

        results.append({
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "total_reward": float(sum(rewards)),
            "steps": step,
        })
        if ep < 5 or ep % 10 == 0:
            print(f"  Ep {ep:3d}: max_reward={results[-1]['max_reward']:.4f}")

    return {
        "mean_max_reward": float(np.mean([r["max_reward"] for r in results])),
        "std_max_reward": float(np.std([r["max_reward"] for r in results])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="overfit",
                        choices=["overfit", "full", "eval_only"],
                        help="overfit: 1-5 demos, full: all demos, eval_only: load and eval")
    parser.add_argument("--n-episodes", type=int, default=3,
                        help="Number of demo episodes for overfit")
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs/diag_lowdim")
    parser.add_argument("--eval-pusht", action="store_true", help="Run PushT env eval")
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None, help="Load existing checkpoint for eval")
    args = parser.parse_args()

    device = torch.device(args.device)
    obs_horizon, action_horizon, seq_len = 2, 16, 19
    os.makedirs(args.output_dir, exist_ok=True)

    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    print(f"Loading PushT data from {zarr_path}...")
    episodes = load_zarr_data(zarr_path)
    print(f"Total episodes: {len(episodes)}")

    # Normalizer
    norm = compute_normalizer(episodes, args.n_episodes if args.mode == "overfit" else len(episodes))
    print(f"Action norm: mean={norm['action_mean']}, std={norm['action_std']}")

    # Dataset
    if args.mode == "overfit":
        n_train = args.n_episodes
        print(f"Overfit mode: using {n_train} episodes")
    else:
        n_train = 90
        print(f"Full mode: using {n_train} episodes")

    dataset = build_overfit_dataset(episodes, n_train, seq_len, obs_horizon, action_horizon)
    print(f"Training samples: {len(dataset)}")

    # Model
    model = LowdimStatePolicy().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M")

    # Train
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    data_iter = iter(loader)
    t0 = time.time()

    s_mean = torch.tensor(norm["state_mean"], device=device).float()
    s_std = torch.tensor(norm["state_std"], device=device).float()
    a_mean = torch.tensor(norm["action_mean"], device=device).float()
    a_std = torch.tensor(norm["action_std"], device=device).float()

    for step in range(args.num_steps):
        try:
            obs_np, act_np = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            obs_np, act_np = next(data_iter)

        obs_state = (obs_np.to(device) - s_mean) / s_std  # (B, 2, 5)
        action_target = act_np[:, obs_horizon - 1 : obs_horizon - 1 + action_horizon].to(device)
        action_norm = (action_target - a_mean) / a_std

        model.train()
        loss, info = model(obs_state, action_norm)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0 or step < 5:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  step {step:6d}: loss={loss.item():.6f}  {sps:.1f} s/s")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    # Action prediction error on training set
    model.eval()
    with torch.no_grad():
        obs_test, act_test = dataset[0:min(100, len(dataset))]
        obs_state = (obs_test.to(device) - s_mean) / s_std
        act_target = act_test[:, obs_horizon - 1 : obs_horizon - 1 + action_horizon].to(device)
        act_target_norm = (act_target - a_mean) / a_std
        _, info = model(obs_state, act_target_norm)
        train_loss = info["loss"]
        # Also check prediction MSE
        act_pred = model.sample(obs_state[:5])
        pred_raw = act_pred * a_std + a_mean
        tgt_raw = act_target[:5]
        pred_mse = F.mse_loss(pred_raw, tgt_raw).item()
        print(f"Train action loss: {train_loss:.6f}")
        print(f"Sample action MSE (unnormalized): {pred_mse:.4f}")

    # Save
    save_path = os.path.join(args.output_dir, "latest.pt")
    torch.save({
        "model": model.state_dict(),
        "step": args.num_steps - 1,
        "action_normalizer": {
            "scale": a_std.cpu().numpy().tolist(),
            "offset": a_mean.cpu().numpy().tolist(),
        },
        "state_normalizer": {
            "mean": s_mean.cpu().numpy().tolist(),
            "std": s_std.cpu().numpy().tolist(),
        },
    }, save_path)
    print(f"Saved: {save_path}")

    # PushT env eval
    if args.eval_pusht:
        print(f"\n=== PushT env eval ({args.n_eval_episodes} episodes) ===")
        env_stats = run_pushT_eval_lowdim(model, norm, device, args.n_eval_episodes)
        print(f"\nMean max_reward: {env_stats['mean_max_reward']:.4f}")
        print(f"Std  max_reward: {env_stats['std_max_reward']:.4f}")
        with open(os.path.join(args.output_dir, "eval_log.json"), "w") as f:
            json.dump(env_stats, f, indent=2)


if __name__ == "__main__":
    main()
