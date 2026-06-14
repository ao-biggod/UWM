#!/usr/bin/env python3
"""Step B: Retrain lowdim oracle with fixed normalizer/scheduler config.

Variants:
  B1: mean/std norm + clip_sample=False
  B2: LinearNormalizer (min/max limits) + clip_sample=True  (DP official recipe)

Changes from original:
  - Normalizer now configurable (mean/std vs min/max limits)
  - clip_sample configurable
  - Eval always clamps actions to [0, 512]
  - 20k training steps (was 10k)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))


# ---- Model (same as before but with configurable scheduler) ----
class LowdimStatePolicyV2(torch.nn.Module):
    def __init__(self, obs_dim=5, action_len=16, action_dim=2, embed_dim=256,
                 depth=6, num_heads=8, num_train_steps=100, num_inference_steps=10,
                 clip_sample=True):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.obs_proj = torch.nn.Sequential(
            torch.nn.Linear(obs_dim * 2, embed_dim),
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
            num_train_timesteps=num_train_steps, beta_schedule="squaredcos_cap_v2",
            clip_sample=clip_sample, prediction_type="epsilon")
        self.num_inference_steps = num_inference_steps

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
        obs_feat = self.obs_proj(obs_state.reshape(B, -1)).unsqueeze(1)
        noise = torch.randn_like(action)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
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


def build_dataset(episodes, n_episodes, seq_len, obs_horizon=2, action_horizon=16):
    all_obs, all_act = [], []
    for ep_state, ep_action in episodes[:n_episodes]:
        T = len(ep_state)
        win = seq_len
        for i in range(T - win):
            all_obs.append(ep_state[i:i + obs_horizon])
            all_act.append(ep_action[i:i + win])
    return TensorDataset(
        torch.tensor(np.stack(all_obs), dtype=torch.float32),
        torch.tensor(np.stack(all_act), dtype=torch.float32),
    )


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


# ---- Normalizer ----
class MinMaxNormalizer:
    """DP official LinearNormalizer equivalent: scale to [-1, 1] using min/max."""
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


class MeanStdNormalizer:
    """Original mean/std normalizer."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-6
        return self

    def normalize(self, data):
        return (data - self.mean) / self.std

    def unnormalize(self, data):
        return data * self.std + self.mean


# ---- Eval ----
def run_pushT_eval(model, norm_state, norm_action, norm_type, device, n_episodes=50):
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
    results = []
    all_sampled_raw = []

    for ep in range(n_episodes):
        env.seed(ep)
        obs = env.reset()
        rewards = []
        state_buffer = []
        done = False
        step = 0

        while not done and step < 300:
            inner = env.env
            agent_p = np.array(inner.agent.position)
            block_p = np.array(inner.block.position)
            block_ang = inner.block.angle
            full_state = np.concatenate([agent_p, block_p, [block_ang]])
            state_buffer.append(full_state)
            if len(state_buffer) > 2:
                state_buffer = state_buffer[-2:]

            if len(state_buffer) < 2:
                obs, reward, done, info = env.step(np.zeros((8, 2)))
                rewards.append(float(reward))
                step += 1
                continue

            state_np = np.stack(state_buffer)
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_norm = norm_state.normalize(state_t.cpu().numpy())
            state_norm = torch.from_numpy(state_norm).float().to(device)

            with torch.no_grad():
                action_norm = model.sample(state_norm)[0]

            action_raw = norm_action.unnormalize(action_norm.cpu().numpy())
            # SAFETY CLAMP — always clamp to valid action space
            action_raw = np.clip(action_raw, 0.0, 512.0)
            action_raw_t = torch.from_numpy(action_raw).float().to(device)

            all_sampled_raw.append(action_raw)

            action_exec = action_raw[:8]
            obs, reward, done, info = env.step(action_exec)
            rewards.append(float(reward))
            step += 1

        results.append({
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "steps": step,
        })
        if ep < 5 or ep % 10 == 0:
            print(f"    Ep {ep:3d}: max_reward={results[-1]['max_reward']:.4f}",
                  flush=True)

    all_raw = np.concatenate([a.reshape(-1, 2) for a in all_sampled_raw], axis=0)
    return results, all_raw


# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="B1_meanstd_noclip or B2_minmax_clip")
    parser.add_argument("--norm-type", type=str, required=True, choices=["meanstd", "minmax"])
    parser.add_argument("--clip-sample", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/stepB_retrain")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    obs_horizon, action_horizon, seq_len = 2, 16, 19
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Step B Retrain: {args.run_name}")
    print(f"  Normalizer:      {args.norm_type}")
    print(f"  clip_sample:     {args.clip_sample}")
    print(f"  Train steps:     {args.num_steps}")
    print(f"  Infer steps:     {args.num_inference_steps}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Output:          {run_dir}")
    print(f"{'='*60}")

    # Load data
    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data(zarr_path)
    print(f"Loaded {len(episodes)} episodes")

    # Build normalizers
    all_state_flat = np.concatenate([s for s, _ in episodes[:90]], axis=0)
    all_action_flat = np.concatenate([a for _, a in episodes[:90]], axis=0)

    if args.norm_type == "minmax":
        norm_state = MinMaxNormalizer().fit(all_state_flat)
        norm_action = MinMaxNormalizer().fit(all_action_flat)
    else:
        norm_state = MeanStdNormalizer().fit(all_state_flat)
        norm_action = MeanStdNormalizer().fit(all_action_flat)

    print(f"Action norm: offset/mean={norm_action.offset if args.norm_type == 'minmax' else norm_action.mean}")
    print(f"Action norm: scale/std={norm_action.scale if args.norm_type == 'minmax' else norm_action.std}")

    # Verify normalized range
    act_norm_check = norm_action.normalize(all_action_flat)
    print(f"Normalized action range: [{act_norm_check.min(axis=0)}, {act_norm_check.max(axis=0)}]")

    # Build dataset
    dataset = build_dataset(episodes, 90, seq_len, obs_horizon, action_horizon)
    print(f"Training samples: {len(dataset)}")

    # Model
    model = LowdimStatePolicyV2(clip_sample=args.clip_sample, num_inference_steps=args.num_inference_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M, clip_sample={args.clip_sample}")

    # Train
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    data_iter = iter(loader)
    t0 = time.time()

    for step in range(args.num_steps):
        try:
            obs_np, act_np = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            obs_np, act_np = next(data_iter)

        obs_norm = norm_state.normalize(obs_np.numpy())
        act_norm_data = norm_action.normalize(act_np.numpy())

        obs_state = torch.from_numpy(obs_norm).float().to(device)
        action_target = torch.from_numpy(
            act_norm_data[:, obs_horizon - 1: obs_horizon - 1 + action_horizon]
        ).float().to(device)

        model.train()
        loss, info = model(obs_state, action_target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0 or step < 5:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  step {step:6d}: loss={loss.item():.6f}  {sps:.1f} s/s", flush=True)

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Quick offline check
    model.eval()
    with torch.no_grad():
        obs_test, act_test = dataset[:64]
        obs_norm_test = norm_state.normalize(obs_test.numpy())
        obs_t = torch.from_numpy(obs_norm_test).float().to(device)
        act_t_norm = norm_action.normalize(act_test[:, obs_horizon - 1: obs_horizon - 1 + action_horizon].numpy())
        act_t = torch.from_numpy(act_t_norm).float().to(device)
        loss_test, _ = model(obs_t, act_t)
        pred_norm = model.sample(obs_t[:5])
        pred_raw = norm_action.unnormalize(pred_norm.cpu().numpy())
        gt_raw = act_test[:5, obs_horizon - 1: obs_horizon - 1 + action_horizon].numpy()
        offline_mse = ((pred_raw - gt_raw) ** 2).mean()

    print(f"Train loss (final batch): {loss.item():.6f}")
    print(f"Val loss (first 64):     {loss_test.item():.6f}")
    print(f"Offline action MSE:      {offline_mse:.1f}")
    print(f"Pred raw range:          [{pred_raw.min():.1f}, {pred_raw.max():.1f}]")
    print(f"GT raw range:            [{gt_raw.min():.1f}, {gt_raw.max():.1f}]")

    # Save
    save_path = os.path.join(run_dir, "latest.pt")
    norm_state_dict = {}
    norm_action_dict = {}
    if args.norm_type == "minmax":
        norm_state_dict = {"offset": norm_state.offset.tolist(), "scale": norm_state.scale.tolist()}
        norm_action_dict = {"offset": norm_action.offset.tolist(), "scale": norm_action.scale.tolist()}
    else:
        norm_state_dict = {"mean": norm_state.mean.tolist(), "std": norm_state.std.tolist()}
        norm_action_dict = {"mean": norm_action.mean.tolist(), "std": norm_action.std.tolist()}

    torch.save({
        "model": model.state_dict(),
        "step": args.num_steps - 1,
        "norm_type": args.norm_type,
        "clip_sample": args.clip_sample,
        "action_normalizer": norm_action_dict,
        "state_normalizer": norm_state_dict,
        "config": {
            "obs_horizon": obs_horizon, "action_horizon": action_horizon,
            "seq_len": seq_len, "num_train_steps": 100,
            "num_inference_steps": args.num_inference_steps,
        },
    }, save_path)
    print(f"Saved: {save_path}")

    # PushT env eval
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print(f"PushT env eval: {args.n_eval_episodes} episodes")
        print(f"{'='*60}")
        results, all_raw = run_pushT_eval(
            model, norm_state, norm_action, args.norm_type,
            device, args.n_eval_episodes,
        )
        mean_max = float(np.mean([r["max_reward"] for r in results]))
        std_max = float(np.std([r["max_reward"] for r in results]))
        high = sum(1 for r in results if r["max_reward"] > 0.5)

        print(f"\n  Mean max_reward:  {mean_max:.4f}")
        print(f"  Std  max_reward:  {std_max:.4f}")
        print(f"  Episodes > 0.5:  {high}/{len(results)}")
        print(f"  Sampled raw min:  {all_raw.min(axis=0)}")
        print(f"  Sampled raw max:  {all_raw.max(axis=0)}")
        print(f"  Sampled raw mean: {all_raw.mean(axis=0)}")

        summary = {
            "run_name": args.run_name,
            "norm_type": args.norm_type,
            "clip_sample": args.clip_sample,
            "mean_max_reward": mean_max,
            "std_max_reward": std_max,
            "offline_action_mse": float(offline_mse),
            "sampled_raw_min": all_raw.min(axis=0).tolist(),
            "sampled_raw_max": all_raw.max(axis=0).tolist(),
        }
        with open(os.path.join(run_dir, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to: {run_dir}/eval_summary.json")

        print(f"\n  >>> {args.run_name}: mean_max_reward = {mean_max:.4f} <<<")


if __name__ == "__main__":
    main()
