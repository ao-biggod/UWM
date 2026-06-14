#!/usr/bin/env python3
"""
E3: KP-TransformerForDiffusion — contract-alignment diagnostic.

Reuses DP official TransformerForDiffusion backbone with:
  - 20D keypoint obs, obs_horizon=2, action_horizon=16
  - causal self-attn on actions + cross-attention to [time, obs0, obs1] memory
  - 8L / 256E / 4H
  - EMA 0.9999, cosine LR + warmup
  - 100 DDPM inference steps

Goal: measure how much of the UWM-vs-B3 gap comes from the AdaLN-DiT
policy contract vs training recipe.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# ═══════════════════════════════════════════════════════════════════
# Data (same as E2)
# ═══════════════════════════════════════════════════════════════════
def load_zarr_data_keypoint(zarr_path):
    import zarr
    z = zarr.open(zarr_path, "r")
    kp = z["data/keypoint"][:]; st = z["data/state"][:]; ac = z["data/action"][:]
    agent_pos = st[:, :2]
    obs_20d = np.concatenate([kp.reshape(kp.shape[0], -1), agent_pos], axis=-1)
    eps = []; s = 0
    for e in z["meta/episode_ends"][:]:
        eps.append((obs_20d[s:e], ac[s:e])); s = e
    return eps

def build_dataset_kp(eps, n_episodes, seq_len, obs_horizon=2, action_horizon=16):
    all_obs, all_act = [], []
    for o, a in eps[:n_episodes]:
        T = len(o)
        for i in range(T - seq_len):
            all_obs.append(o[i:i+obs_horizon])
            all_act.append(a[i:i+seq_len])
    return torch.utils.data.TensorDataset(
        torch.tensor(np.stack(all_obs), dtype=torch.float32),
        torch.tensor(np.stack(all_act), dtype=torch.float32))

def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

class MinMaxNormalizer:
    def __init__(self):
        self.offset = self.scale = None
    def fit(self, data):
        dmin = data.min(axis=0); dmax = data.max(axis=0)
        self.offset = (dmax + dmin)/2.0
        self.scale = (dmax - dmin)/2.0
        self.scale[self.scale < 1e-6] = 1.0
        return self
    def normalize(self, data):
        return (data - self.offset) / self.scale
    def unnormalize(self, data):
        return data * self.scale + self.offset

# ═══════════════════════════════════════════════════════════════════
# Policy wrapper
# ═══════════════════════════════════════════════════════════════════
class TransformerForDiffusionPolicy(nn.Module):
    """Action-only diffusion policy using DP official TransformerForDiffusion."""

    def __init__(self, obs_dim=20, action_dim=2, horizon=16, n_obs_steps=2,
                 n_layer=8, n_head=4, n_emb=256, p_drop_attn=0.01,
                 causal_attn=True, time_as_cond=True, obs_as_cond=True,
                 n_cond_layers=0, num_train_timesteps=100, num_inference_steps=100):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps

        self.model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=0.0,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def forward(self, obs, action):
        """obs: [B, 2, 20] normalized, action: [B, 16, 2] normalized."""
        B = action.shape[0]
        noise = torch.randn_like(action)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                          (B,), device=action.device).long()
        noisy_action = self.noise_scheduler.add_noise(action, noise, t)
        noise_pred = self.model(noisy_action, t, cond=obs)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, obs):
        """obs: [B, 2, 20] normalized → action: [B, 16, 2] normalized."""
        B, device = obs.shape[0], obs.device
        action = torch.randn(B, self.horizon, self.action_dim, device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t_step in self.noise_scheduler.timesteps:
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            noise_pred = self.model(action, t, cond=obs)
            action = self.noise_scheduler.step(noise_pred, t_step, action).prev_sample
        return action


# ═══════════════════════════════════════════════════════════════════
# Fixed-buffer eval (same as E2)
# ═══════════════════════════════════════════════════════════════════
def eval_fixed_buffer(model, ema_model, norm_state, norm_action, device, n_eps=50,
                      exec_slice="dp"):
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    if ema_model is not None:
        ema_model.copy_to(model.parameters())

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
        obs_buffer.append(raw_obs[0, :Do]); obs_buffer.append(raw_obs[1, :Do])
        rewards = []; done = False; step = 0

        while not done and step < 300:
            state_np = np.stack(list(obs_buffer))
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_norm = (state_t - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()
            with torch.no_grad():
                action_norm = model.sample(state_norm)[0]
            action_raw = action_norm * torch.tensor(norm_action.scale, device=device).float() + torch.tensor(norm_action.offset, device=device).float()
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
            if exec_slice == "uwm":
                exec_actions = action_raw[:8]
            else:
                exec_actions = action_raw[1:9]
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


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-train-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/e3_kp_tfd")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--ema", default="0.9999", type=str,
                        help="EMA decay or 'none' to disable")
    parser.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--target-slice", default="dp", choices=["dp", "uwm"],
                        help="dp: act[:, :16], uwm: act[:, 1:17]")
    parser.add_argument("--exec-slice", default="dp", choices=["dp", "uwm"],
                        help="dp: pred[1:9], uwm: pred[0:8]")
    args = parser.parse_args()

    device = torch.device(args.device)
    obs_horizon, action_horizon, seq_len = 2, 16, 19
    obs_dim = 20
    os.makedirs(args.output_dir, exist_ok=True)

    use_ema = args.ema.lower() != "none"
    ema_decay = float(args.ema) if use_ema else 0.0

    # Action slicing conventions
    if args.target_slice == "dp":
        _get_target = lambda act_np: act_np[:, :action_horizon]           # act[:, :16]
    else:
        _get_target = lambda act_np: act_np[:, obs_horizon - 1: obs_horizon - 1 + action_horizon]  # act[:, 1:17]

    if args.exec_slice == "dp":
        _get_exec = lambda action_raw: action_raw[1:9]                     # pred[1:9]
    else:
        _get_exec = lambda action_raw: action_raw[:8]                      # pred[:8]

    exp_name = "E3" if (args.target_slice == "dp" and args.exec_slice == "dp") else \
               "E4" if (args.target_slice == "uwm" and args.exec_slice == "uwm") else \
               "E3/E4-custom"

    print("=" * 60)
    print(f"{exp_name}: KP-TransformerForDiffusion")
    print(f"  Architecture: TransformerForDiffusion (8L/256E/4H)")
    print(f"  Causal self-attn: True")
    print(f"  Obs conditioning: cross-attention [time, obs0, obs1] → memory")
    print(f"  Target slice: {args.target_slice}  ({'act[:,:16]' if args.target_slice=='dp' else 'act[:,1:17]'})")
    print(f"  Exec slice:   {args.exec_slice}  ({'pred[1:9]' if args.exec_slice=='dp' else 'pred[0:8]'})")
    print(f"  EMA: {'0.9999' if use_ema else 'none'}")
    print(f"  LR: {args.lr_schedule}{' + warmup=' + str(args.lr_warmup_steps) if args.lr_schedule == 'cosine' else ''}")
    print(f"  Inf steps: 100 (DDPM)")
    print(f"  Training steps: {args.max_train_steps}")
    print("=" * 60)

    # Data
    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    eps = load_zarr_data_keypoint(zarr_path)
    all_obs = np.concatenate([s for s, _ in eps[:90]], axis=0)
    all_action = np.concatenate([a for _, a in eps[:90]], axis=0)
    norm_state = MinMaxNormalizer().fit(all_obs)
    norm_action = MinMaxNormalizer().fit(all_action)
    dataset = build_dataset_kp(eps, 90, seq_len, obs_horizon, action_horizon)
    print(f"Training samples: {len(dataset)}")

    # Model
    model = TransformerForDiffusionPolicy(
        obs_dim=obs_dim, action_dim=2, horizon=16, n_obs_steps=2,
        n_layer=8, n_head=4, n_emb=256, p_drop_attn=0.01,
        causal_attn=True, time_as_cond=True, obs_as_cond=True,
        n_cond_layers=0, num_train_timesteps=100, num_inference_steps=100,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    # EMA
    ema_model = None
    if use_ema:
        ema_model = EMAModel(
            model.parameters(), decay=ema_decay, min_decay=0.0,
            update_after_step=0, use_ema_warmup=True,
            inv_gamma=1.0, power=0.75)
        print(f"EMA: decay={ema_decay}, power=0.75")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    lr_scheduler = None
    warmup_steps = args.lr_warmup_steps if args.lr_schedule == "cosine" else 0
    if args.lr_schedule == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=args.max_train_steps)
        print(f"LR schedule: cosine + {warmup_steps}-step warmup")

    # ── SANITY CHECK: first batch shapes ──
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True)
    data_iter = iter(loader)
    obs_batch, act_batch = next(data_iter)
    obs_norm_batch = (obs_batch.to(device) - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()
    # DP official convention: train on act[:, :16], execute pred[1:9]
    action_target = _get_target(act_batch).to(device)
    action_norm_batch = (action_target - torch.tensor(norm_action.offset, device=device).float()) / torch.tensor(norm_action.scale, device=device).float()

    B = obs_batch.shape[0]
    print(f"\n[SANITY 1: Shapes]")
    print(f"  obs batch:          {obs_batch.shape}  (B, To=2, Do=20)")
    print(f"  action target:      {action_target.shape}  (B, Ta=16, Da=2)")
    print(f"  action norm range:  [{action_norm_batch.min():.3f}, {action_norm_batch.max():.3f}]")

    # ── SANITY CHECK: first forward pass ──
    model.train()
    loss = model(obs_norm_batch, action_norm_batch)
    print(f"  first train loss:   {loss.item():.6f}")

    # ── SANITY CHECK: model output shape ──
    with torch.no_grad():
        # 1 sample
        t_test = torch.zeros(B, device=device, dtype=torch.long)
        noise_pred_single = model.model(
            torch.randn(B, 16, 2, device=device), t_test,
            cond=obs_norm_batch)
        print(f"  model output shape: {noise_pred_single.shape}  expected: [{B}, 16, 2]")
        assert noise_pred_single.shape == (B, 16, 2), f"BAD SHAPE: {noise_pred_single.shape}"

        # full sample path
        act_sample_norm = model.sample(obs_norm_batch[:1])
        print(f"  sample output shape:{act_sample_norm.shape}  expected: [1, 16, 2]")
        assert act_sample_norm.shape == (1, 16, 2), f"BAD SAMPLE SHAPE"

    # ── SANITY CHECK: architecture details ──
    print(f"\n[SANITY 2: Architecture]")
    print(f"  model.causal_attn: {model.model.causal_attn if hasattr(model.model, 'causal_attn') else 'N/A'}")
    print(f"  model.obs_as_cond: {model.model.obs_as_cond}")
    print(f"  model.time_as_cond: {model.model.time_as_cond}")
    print(f"  model.encoder_only: {model.model.encoder_only}")
    print(f"  model.T (action tokens): {model.model.T}")
    print(f"  model.T_cond (condition tokens): {model.model.T_cond}")
    if hasattr(model.model, 'mask'):
        print(f"  causal mask shape: {model.model.mask.shape if model.model.mask is not None else 'None'}")
    print(f"  input_emb: {model.model.input_emb}")
    print(f"  head: {model.model.head}")
    print(f"  noise_scheduler.train_timesteps: {model.noise_scheduler.config.num_train_timesteps}")
    print(f"  inf steps: {model.num_inference_steps}")

    # ── SANITY CHECK: compare with DP official checkpoint (first-call parity) ──
    print(f"\n[SANITY 3: DP official parity]")
    ckpt_path = "outputs/e0_lowdim_official_full/checkpoints/epoch=0190-test_mean_score=1.000.ckpt"
    if os.path.exists(ckpt_path):
        dp_ckpt = torch.load(ckpt_path, map_location=device)
        # Verify key shapes match
        our_sd = model.state_dict()
        dp_sd = dp_ckpt["state_dicts"]["model"]
        # Compare key parameter shapes
        keys_to_check = [
            "model.input_emb.weight", "model.pos_emb", "model.head.weight",
            "model.cond_obs_emb.weight",
        ]
        all_match = True
        for k in keys_to_check:
            if k in our_sd and k in dp_sd:
                ourshape = our_sd[k].shape; dshape = dp_sd[k].shape
                ok = "MATCH" if ourshape == dshape else "DIFF"
                if ok == "DIFF": all_match = False
                print(f"  {k}: ours={list(ourshape)}  dp={list(dshape)}  [{ok}]")
            else:
                print(f"  {k}: missing in one dict")
        if all_match:
            print("  All key shapes match DP official checkpoint ✓")
        else:
            print("  Shape mismatch with DP official — check config")
    else:
        print(f"  DP official checkpoint not found at {ckpt_path} — skip parity check")

    # ── SANITY CHECK: loss a few steps ──
    print(f"\n[SANITY 4: First 10 losses]")
    losses_10 = []
    for step in range(10):
        try: obs_np, act_np = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); obs_np, act_np = next(data_iter)
        obs_n = (obs_np.to(device) - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()
        act_target = _get_target(act_np).to(device)
        act_n = (act_target - torch.tensor(norm_action.offset, device=device).float()) / torch.tensor(norm_action.scale, device=device).float()
        model.train()
        l = model(obs_n, act_n)
        optimizer.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if lr_scheduler: lr_scheduler.step()
        if ema_model: ema_model.step(model.parameters())
        losses_10.append(float(l.item()))
        print(f"  step {step}: loss={l.item():.6f}", flush=True)

    print(f"\nAll sanity checks complete.")
    if any(math.isnan(l) for l in losses_10):
        print("WARNING: NaN detected in losses!")
        return

    print(f"\n{'='*60}")
    print(f"Starting training: {args.max_train_steps} steps")
    print(f"{'='*60}")

    # ── Training loop ──
    data_iter = iter(loader)
    t0 = time.time()
    train_log = []

    for step in range(args.max_train_steps):
        try: obs_np, act_np = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); obs_np, act_np = next(data_iter)

        obs_n = (obs_np.to(device) - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()
        act_target = _get_target(act_np).to(device)
        act_norm_t = (act_target - torch.tensor(norm_action.offset, device=device).float()) / torch.tensor(norm_action.scale, device=device).float()

        model.train()
        loss = model(obs_n, act_norm_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if lr_scheduler: lr_scheduler.step()
        if ema_model: ema_model.step(model.parameters())

        train_log.append(float(loss.item()))
        if step % 500 == 0 or step < 10:
            elapsed = max(time.time() - t0, 1e-6)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  step {step:6d}: loss={loss.item():.6f}  lr={lr_now:.2e}  {step/elapsed:.1f} s/s", flush=True)

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s ({args.max_train_steps} steps)")
    print(f"Final loss: {train_log[-1]:.6f}")

    # ── Save checkpoint ──
    save_dict = {
        "model": model.state_dict(),
        "step": args.max_train_steps,
        "action_normalizer": {"offset": norm_action.offset.tolist(), "scale": norm_action.scale.tolist()},
        "state_normalizer": {"offset": norm_state.offset.tolist(), "scale": norm_state.scale.tolist()},
        "config": {
            "model_name": "TransformerForDiffusion",
            "n_layer": 8, "n_head": 4, "n_emb": 256,
            "obs_horizon": obs_horizon, "action_horizon": action_horizon,
            "obs_dim": obs_dim, "action_dim": 2, "horizon": 16,
            "use_ema": use_ema, "ema_decay": ema_decay,
            "lr_schedule": args.lr_schedule, "lr_warmup_steps": warmup_steps,
            "inf_steps": 100,
        },
    }
    if ema_model:
        save_dict["ema"] = ema_model.state_dict()
    torch.save(save_dict, os.path.join(args.output_dir, "latest.pt"))
    print(f"Saved: {args.output_dir}/latest.pt")

    # ── Eval ──
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print(f"Eval: {args.n_eval_episodes} episodes (EMA={'on' if use_ema else 'off'})")
        scores = eval_fixed_buffer(model, ema_model, norm_state, norm_action, device,
                                   args.n_eval_episodes, exec_slice=args.exec_slice)
        summary = {
            "model_name": "TransformerForDiffusion",
            "mean": float(scores.mean()), "std": float(scores.std()),
            "median": float(np.median(scores)), "n_eps": args.n_eval_episodes,
            "ep_gt_05": int((scores > 0.5).sum()),
            "train_loss_final": train_log[-1] if train_log else None,
        }
        with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Eval summary: {args.output_dir}/eval_summary.json")

    print(f"\nE3 complete. Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
