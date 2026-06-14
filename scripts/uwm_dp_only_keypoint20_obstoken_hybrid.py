#!/usr/bin/env python3
"""
E2-4: UWM DP-only + 20D keypoint + hybrid obs-token conditioning.

Baseline UWM-DP-KP:
    action tokens only, obs -> global_cond -> AdaLN

This ablation:
    [obs_token, action_1, ..., action_16] participate in self-attention
    still keep obs -> global_cond -> AdaLN

Unchanged:
    no video loss
    same 20D keypoint obs
    same UWM 12L/768E/12H backbone scale
    same DDPM scheduler
    same MinMax normalizer
    same action slicing t1:t9
    same 20k train steps for final run
    same 50ep eval for final run
    no EMA
    no cosine LR
"""
import argparse, json, os, sys, time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from models.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from models.common.utils import SinusoidalPosEmb, init_weights


# ─────────────────────────────────────────────────────────────────────
# Data loading (same as uwm_dp_only_keypoint20.py)
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# Core: Hybrid obs-token noise prediction net
# ─────────────────────────────────────────────────────────────────────
class TransformerNoisePredictionNetHybrid(nn.Module):
    """Like TransformerNoisePredictionNet, but prepends an obs_token to the
    action sequence. The obs_token participates in bidirectional self-attention
    with all action tokens. AdaLN modulation via global_cond (obs_embed) is
    still kept."""

    def __init__(
        self,
        input_len: int = 16,
        input_dim: int = 2,
        global_cond_dim: int = 768,
        timestep_embed_dim: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.input_len = input_len
        self.embed_dim = embed_dim

        # Input encoder/decoder (same as original)
        hidden_dim = int(max(input_dim, embed_dim) * mlp_ratio)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Timestep encoder
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        # Positional embedding (action tokens only)
        self.pos_embed = nn.Parameter(
            torch.empty(1, input_len, embed_dim).normal_(std=0.02)
        )
        # Positional embedding (obs token)
        self.obs_pos_embed = nn.Parameter(
            torch.empty(1, 1, embed_dim).normal_(std=0.02)
        )

        # AdaLN blocks
        cond_dim = global_cond_dim + timestep_embed_dim
        self.blocks = nn.ModuleList([
            AdaLNAttentionBlock(
                dim=embed_dim, cond_dim=cond_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            )
            for _ in range(depth)
        ])
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(init_weights)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(self, sample, timestep, obs_embed):
        """sample: [B, 16, 2] noisy action, obs_embed: [B, 768]"""
        B = sample.shape[0]
        device = sample.device

        # Encode action
        act_embed = self.input_encoder(sample)          # [B, 16, 768]
        act_tokens = act_embed + self.pos_embed         # [B, 16, 768]  (pos embed on action only)

        # Obs token (with its own positional embedding)
        obs_token = obs_embed.unsqueeze(1)               # [B, 1, 768]
        obs_token = obs_token + self.obs_pos_embed       # [B, 1, 768]

        # Concat: [obs, a1..a16]
        x = torch.cat([obs_token, act_tokens], dim=1)    # [B, 17, 768]

        # Timestep + global condition (AdaLN path kept)
        if len(timestep.shape) == 0:
            timestep = timestep.expand(B).to(dtype=torch.long, device=device)
        temb = self.timestep_encoder(timestep)            # [B, 256]
        cond = torch.cat([obs_embed, temb], dim=-1)       # [B, 768+256=1024]

        # Forward
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)                            # [B, 17, 768]

        # Decode only action tokens (skip obs token at position 0)
        x_action = x[:, 1:]                               # [B, 16, 768]
        noise_pred = self.output_decoder(x_action)        # [B, 16, 2]
        return noise_pred


# ─────────────────────────────────────────────────────────────────────
# Lowdim obs encoder (same as uwm_dp_only_keypoint20.py)
# ─────────────────────────────────────────────────────────────────────
class LowdimObsEncoder(nn.Module):
    def __init__(self, obs_dim, num_frames, embed_dim):
        super().__init__()
        in_dim = obs_dim * num_frames
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim))

    def forward(self, obs):
        B, T, D = obs.shape
        return self.net(obs.reshape(B, T * D))


# ─────────────────────────────────────────────────────────────────────
# Policy wrapper
# ─────────────────────────────────────────────────────────────────────
class LowdimDiTPolicyHybrid(nn.Module):
    def __init__(self, obs_dim, action_len, action_dim, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.obs_encoder = LowdimObsEncoder(obs_dim, 2, embed_dim)
        self.noise_pred_net = TransformerNoisePredictionNetHybrid(
            input_len=action_len, input_dim=action_dim,
            global_cond_dim=embed_dim,
            timestep_embed_dim=256, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=4, qkv_bias=True)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon")
        self.num_inference_steps = 10
        self._first_forward = True

    def forward(self, obs, action):
        B = action.shape[0]
        obs_embed = self.obs_encoder(obs)
        noise = torch.randn_like(action)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                          (B,), device=action.device).long()
        noisy_action = self.noise_scheduler.add_noise(action, noise, t)
        noise_pred = self.noise_pred_net(noisy_action, t, obs_embed=obs_embed)

        if self._first_forward:
            self._first_forward = False
            _act_embed = self.noise_pred_net.input_encoder(noisy_action[:1])
            _act_tokens = _act_embed + self.noise_pred_net.pos_embed
            _obs_token = obs_embed[:1].unsqueeze(1) + self.noise_pred_net.obs_pos_embed
            _x = torch.cat([_obs_token, _act_tokens], dim=1)
            _temb = self.noise_pred_net.timestep_encoder(t[:1])
            _cond = torch.cat([obs_embed[:1], _temb], dim=-1)
            print("\n[SANITY SHAPES — first forward]")
            print(f"  obs shape:              {obs.shape}  (B={B}, T, D)")
            print(f"  obs_embed shape:        {obs_embed.shape}  (B={B}, E={self.embed_dim})")
            print(f"  obs_token shape:        {_obs_token.shape}")
            print(f"  action input shape:     {noisy_action.shape}")
            assert action.shape == (B, self.action_len, self.action_dim), f"action shape mismatch: {action.shape}"
            assert obs_embed.shape == (B, self.embed_dim), f"obs_embed shape: {obs_embed.shape}"
            assert _obs_token.shape == (1, 1, self.embed_dim), f"obs_token shape: {_obs_token.shape}"
            assert _act_tokens.shape == (1, self.action_len, self.embed_dim), f"act_tokens shape: {_act_tokens.shape}"
            print(f"  action_token shape:     {_act_tokens.shape}")
            assert _x.shape == (1, 1 + self.action_len, self.embed_dim), f"concat shape: {_x.shape}"
            print(f"  concat token shape:     {_x.shape}  [obs + a1..a{self.action_len}]")
            assert _cond.shape == (1, self.embed_dim + 256), f"cond shape: {_cond.shape}"
            print(f"  cond shape:             {_cond.shape}")
            assert noise_pred.shape == (B, self.action_len, self.action_dim), f"noise_pred shape: {noise_pred.shape}"
            print(f"  decoded action shape:   {noise_pred.shape}")
            print(f"  exec action slice:      t1:t9  (n_action_steps=8)\n")

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, obs):
        """Sample actions. obs: [B, 2, 20] normalized."""
        B, device = obs.shape[0], obs.device
        obs_embed = self.obs_encoder(obs)
        action = torch.randn(B, self.action_len, self.action_dim, device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t_step in self.noise_scheduler.timesteps:
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            noise_pred = self.noise_pred_net(action, t, obs_embed=obs_embed)
            action = self.noise_scheduler.step(noise_pred, t_step, action).prev_sample
        return action


# ─────────────────────────────────────────────────────────────────────
# Fixed-buffer eval (same as uwm_dp_only_keypoint20.py)
# ─────────────────────────────────────────────────────────────────────
def eval_fixed_buffer(model, norm_state, norm_action, device, n_eps=50):
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
        obs_buffer.append(raw_obs[0, :Do])
        obs_buffer.append(raw_obs[1, :Do])
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
            exec_actions = action_raw[:8]  # first 8 = t1:t9 in training convention

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
    print(f"\n  eval ({n_eps}eps): mean={scores.mean():.4f} std={scores.std():.4f} "
          f"median={np.median(scores):.4f} ep>0.5={(scores>0.5).sum()}/{n_eps}")
    return scores


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-train-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str,
                        default="outputs/e2_uwm_kp_obstoken_hybrid")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    obs_horizon, action_horizon, seq_len = 2, 16, 19
    obs_dim = 20
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"E2-4: UWM DP-only + 20D Keypoint + HYBRID OBS-TOKEN CONDITIONING")
    print(f"  Model variant: obstoken_hybrid")
    print(f"  Backbone: TransformerNoisePredictionNetHybrid (12L/768E/12H)")
    print(f"  Obs: 20D keypoint")
    print(f"  Self-attention tokens: 17 (1 obs + 16 action)")
    print(f"  AdaLN: KEPT (obs → global_cond)")
    print(f"  Training steps: {args.max_train_steps}")
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
    model = LowdimDiTPolicyHybrid(
        obs_dim=obs_dim, action_len=16, action_dim=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M")
    print(f"Model variant: obstoken_hybrid")

    # Training
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    data_iter = iter(loader)
    t0 = time.time()
    losses_log = []

    for step in range(args.max_train_steps):
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

        if step < 10 or step % 100 == 0:
            elapsed = max(time.time() - t0, 1e-6)
            print(f"  step {step:6d}: loss={loss.item():.6f}  {step / elapsed:.1f} s/s", flush=True)
        losses_log.append(float(loss.item()))

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s  ({args.max_train_steps} steps)")

    # First 10 losses
    print(f"\nFirst 10 train losses:")
    for i, l in enumerate(losses_log[:10]):
        print(f"  step {i}: {l:.6f}")

    # Offline MSE (quick)
    model.eval()
    obs_test, act_test = dataset[:64]
    obs_norm = (obs_test.to(device) - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()
    with torch.no_grad():
        act_pred_norm = model.sample(obs_norm)
    act_pred_raw = act_pred_norm * torch.tensor(norm_action.scale, device=device).float() + torch.tensor(norm_action.offset, device=device).float()
    gt16 = act_test[:, obs_horizon - 1: obs_horizon - 1 + action_horizon].to(device)
    mse_all = F.mse_loss(act_pred_raw[:, :16], gt16[:, :16]).item()
    exec_start, exec_end = obs_horizon - 1, obs_horizon - 1 + 8
    mse_exec8 = F.mse_loss(act_pred_raw[:, exec_start:exec_end], gt16[:, exec_start:exec_end]).item()
    print(f"  offline MSE all16: {mse_all:.1f}  exec8(t1:t9): {mse_exec8:.1f}")

    # Save checkpoint
    torch.save({
        "model": model.state_dict(),
        "step": args.max_train_steps - 1,
        "action_normalizer": {"offset": norm_action.offset.tolist(), "scale": norm_action.scale.tolist()},
        "state_normalizer": {"offset": norm_state.offset.tolist(), "scale": norm_state.scale.tolist()},
        "config": {"model_variant": "obstoken_hybrid", "obs_horizon": obs_horizon,
                   "action_horizon": action_horizon, "obs_dim": obs_dim, "action_len": 16},
    }, os.path.join(args.output_dir, "latest.pt"))
    print(f"Saved: {args.output_dir}/latest.pt")

    # Eval
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print(f"Eval: {args.n_eval_episodes} episodes")
        scores = eval_fixed_buffer(model, norm_state, norm_action, device, args.n_eval_episodes)
        summary = {"model_variant": "obstoken_hybrid",
                   "mean": float(scores.mean()), "std": float(scores.std()),
                   "median": float(np.median(scores)), "n_eps": args.n_eval_episodes,
                   "ep_gt_05": int((scores > 0.5).sum()),
                   "offline_mse_all16": mse_all, "offline_mse_exec8": mse_exec8}
        with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Eval summary saved: {args.output_dir}/eval_summary.json")

    print(f"\n{'='*60}")
    print(f"Sanity complete. Model variant: obstoken_hybrid")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
