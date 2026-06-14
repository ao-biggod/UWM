#!/usr/bin/env python3
"""Step D2: B4 training with DP-official EMA and/or cosine LR.

Variants:
  B4a: EMA=True,  LR=constant  → isolate EMA
  B4b: EMA=False, LR=cosine    → isolate cosine LR
  B4c: EMA=True,  LR=cosine    → match DP-style recipe

Uses DP official EMAModel and get_scheduler directly.
Saves checkpoints every 5k steps with both raw and EMA weights.
Prints EMA decay schedule.
"""
import argparse, copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for scripts imports

from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler


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
    return torch.utils.data.TensorDataset(
        torch.tensor(np.stack(all_obs), dtype=torch.float32),
        torch.tensor(np.stack(all_act), dtype=torch.float32),
    )


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

    def to_dict(self):
        return {"offset": self.offset.tolist(), "scale": self.scale.tolist()}


def build_model(clip_sample=True, num_inference_steps=10):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stepB_retrain_lowdim",
        str(Path(__file__).resolve().parent / "stepB_retrain_lowdim.py"))
    stepB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stepB)
    return stepB.LowdimStatePolicyV2(clip_sample=clip_sample, num_inference_steps=num_inference_steps)


def get_full_state(env):
    agent_p = np.array(env.agent.position)
    block_p = np.array(env.block.position)
    block_ang = env.block.angle
    return np.concatenate([agent_p, block_p, [block_ang]])


def eval_fixed_buffer(model, norm_state, unnorm_action, device, n_eps=50, n_action_exec=8):
    """Fixed-buffer eval using given model (can be raw or EMA)."""
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    model.eval()
    results = []
    all_deltas = []

    for ep in range(n_eps):
        raw_env = PushTImageEnv(legacy=True)
        raw_env.seed(ep)
        raw_env.reset()

        obs_buffer = deque(maxlen=2)
        rewards = []
        done = False
        physical_step = 0
        ep_deltas = []

        # Prime
        s0 = get_full_state(raw_env)
        obs_buffer.append(s0)
        raw_env.step(np.zeros(2))
        physical_step += 1
        s1 = get_full_state(raw_env)
        obs_buffer.append(s1)

        while not done and physical_step < 300:
            state_np = np.stack(list(obs_buffer))
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_normed = norm_state(state_t)

            with torch.no_grad():
                action_norm = model.sample(state_normed)[0]
            action_raw = unnorm_action(action_norm)
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

            exec_actions = action_raw[:n_action_exec]
            delta = exec_actions[1:] - exec_actions[:-1]
            delta_l2 = np.linalg.norm(delta, axis=-1)
            ep_deltas.append(float(delta_l2.mean()))

            for k in range(n_action_exec):
                act = exec_actions[k]
                _obs, reward, _done, _info = raw_env.step(act)
                physical_step += 1
                full_state = get_full_state(raw_env)
                obs_buffer.append(full_state)
                rewards.append(float(reward))
                if _done or physical_step >= 300:
                    done = True
                    break

        max_reward = float(max(rewards)) if rewards else 0.0
        results.append({"max_reward": max_reward, "steps": physical_step})
        all_deltas.append(float(np.mean(ep_deltas)))

    mean_max = float(np.mean([r["max_reward"] for r in results]))
    std_max = float(np.std([r["max_reward"] for r in results]))
    median = float(np.median([r["max_reward"] for r in results]))
    p25 = float(np.percentile([r["max_reward"] for r in results], 25))
    p75 = float(np.percentile([r["max_reward"] for r in results], 75))
    ep_gt_05 = sum(1 for r in results if r["max_reward"] > 0.5)
    ep_gt_08 = sum(1 for r in results if r["max_reward"] > 0.8)
    min_score = float(min(r["max_reward"] for r in results))

    return {
        "mean_max_reward": mean_max, "std_max_reward": std_max,
        "median": median, "p25": p25, "p75": p75, "min": min_score,
        "max": float(max(r["max_reward"] for r in results)),
        "ep_gt_05": ep_gt_05, "ep_gt_08": ep_gt_08,
        "n_episodes": n_eps,
        "action_delta_mean": float(np.mean(all_deltas)),
    }


def compute_offline_mse(model, norm_state, norm_action, unnorm_action, dataset, device,
                        obs_horizon=2, action_horizon=16):
    """Compute offline MSE on first 64 validation samples."""
    model.eval()
    obs_test, act_test = dataset[:min(64, len(dataset))]
    obs_norm = norm_state(torch.from_numpy(obs_test.numpy()).float().to(device))
    with torch.no_grad():
        act_pred_norm = model.sample(obs_norm[:5])
    act_pred_raw = unnorm_action(act_pred_norm)
    act_gt_raw = act_test[:5, obs_horizon - 1: obs_horizon - 1 + action_horizon]
    mse = F.mse_loss(act_pred_raw, act_gt_raw.to(device)).item()
    return mse


def train_and_eval(args):
    device = torch.device(args.device)
    obs_horizon, action_horizon, seq_len = 2, 16, 19

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"D2 B4 Training: {args.run_name}")
    print(f"  EMA:         {args.use_ema}")
    print(f"  LR schedule: {args.lr_scheduler}")
    print(f"  Train steps: {args.num_steps}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Output:      {run_dir}")
    print(f"{'='*60}")

    # ---- Data ----
    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data(zarr_path)
    all_action = np.concatenate([a for _, a in episodes[:90]], axis=0)
    all_state = np.concatenate([s for s, _ in episodes[:90]], axis=0)

    norm_state = MinMaxNormalizer().fit(all_state)
    norm_action = MinMaxNormalizer().fit(all_action)

    s_offset = torch.tensor(norm_state.offset, device=device).float()
    s_scale = torch.tensor(norm_state.scale, device=device).float()
    a_offset = torch.tensor(norm_action.offset, device=device).float()
    a_scale = torch.tensor(norm_action.scale, device=device).float()

    def ns(x):
        return (x - s_offset) / s_scale
    def ua(x):
        return x * a_scale + a_offset
    def na(x):
        return (x - a_offset) / a_scale

    dataset = build_dataset(episodes, 90, seq_len, obs_horizon, action_horizon)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True)
    print(f"Training samples: {len(dataset)}")

    # ---- Model ----
    raw_model = build_model(clip_sample=True, num_inference_steps=10).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M")

    # EMA model
    ema_model = None
    ema = None
    if args.use_ema:
        ema_model = copy.deepcopy(raw_model)
        ema_model.eval()
        ema_model.requires_grad_(False)
        ema = EMAModel(
            model=ema_model,
            update_after_step=0,
            inv_gamma=1.0,
            power=0.75,
            min_value=0.0,
            max_value=0.9999,
        )

    # ---- Optimizer + LR scheduler ----
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=1e-6)
    lr_scheduler = None
    if args.lr_scheduler == "cosine":
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=args.num_steps,
        )

    # ---- Training ----
    data_iter = iter(loader)
    t0 = time.time()
    checkpoint_steps = list(range(args.checkpoint_every, args.num_steps + 1, args.checkpoint_every))
    ema_decay_log = []

    for step in range(args.num_steps):
        try:
            obs_np, act_np = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            obs_np, act_np = next(data_iter)

        obs_state = ns(torch.from_numpy(obs_np.numpy()).float().to(device))
        action_target = na(torch.from_numpy(
            act_np.numpy()[:, obs_horizon - 1: obs_horizon - 1 + action_horizon]
        ).float().to(device))

        raw_model.train()
        loss, info = raw_model(obs_state, action_target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # EMA step after optimizer step (DP official ordering)
        if ema is not None:
            ema.step(raw_model)

        # Log EMA decay
        if ema is not None and step in [0, 1, 10, 100, 1000, 5000, 10000, 15000, 19999]:
            ema_decay_log.append({"step": step, "decay": ema.decay})

        if step % 1000 == 0 or step < 5:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else args.lr
            ema_d = f"ema_decay={ema.decay:.4f}" if ema else ""
            print(f"  step {step:6d}: loss={loss.item():.6f}  lr={lr:.2e}  {ema_d}  {sps:.1f} s/s",
                  flush=True)

        # Save checkpoint
        if (step + 1) in checkpoint_steps:
            ckpt = {
                "step": step,
                "raw_model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "state_normalizer": norm_state.to_dict(),
                "action_normalizer": norm_action.to_dict(),
                "config": {"obs_horizon": obs_horizon, "action_horizon": action_horizon,
                          "seq_len": seq_len, "use_ema": args.use_ema,
                          "lr_scheduler": args.lr_scheduler},
            }
            if lr_scheduler:
                ckpt["lr_scheduler"] = lr_scheduler.state_dict()
            if ema_model is not None:
                ckpt["ema_model"] = ema_model.state_dict()
                ckpt["ema_config"] = {"inv_gamma": 1.0, "power": 0.75,
                                      "min_value": 0.0, "max_value": 0.9999}

            ckpt_path = os.path.join(run_dir, f"checkpoint_step_{step+1:06d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  [Checkpoint saved: {ckpt_path}]", flush=True)

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ---- Final save ----
    final_ckpt = {
        "step": args.num_steps - 1,
        "raw_model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state_normalizer": norm_state.to_dict(),
        "action_normalizer": norm_action.to_dict(),
        "config": {"obs_horizon": obs_horizon, "action_horizon": action_horizon,
                  "seq_len": seq_len, "use_ema": args.use_ema,
                  "lr_scheduler": args.lr_scheduler},
        "ema_decay_log": ema_decay_log,
    }
    if lr_scheduler:
        final_ckpt["lr_scheduler"] = lr_scheduler.state_dict()
    if ema_model is not None:
        final_ckpt["ema_model"] = ema_model.state_dict()
        final_ckpt["ema_config"] = {"inv_gamma": 1.0, "power": 0.75,
                                    "min_value": 0.0, "max_value": 0.9999}

    latest_path = os.path.join(run_dir, "latest.pt")
    torch.save(final_ckpt, latest_path)
    print(f"Final checkpoint saved: {latest_path}")

    # ---- EMA decay schedule ----
    if ema_decay_log:
        print(f"\n{'='*40}")
        print(f"EMA Decay Schedule")
        print(f"{'='*40}")
        print(f"  {'Step':>8s}  {'Decay':>10s}")
        for entry in ema_decay_log:
            print(f"  {entry['step']:8d}  {entry['decay']:10.6f}")

    # ---- Offline MSE ----
    offline_mse_raw = compute_offline_mse(raw_model, ns, na, ua, dataset, device)
    print(f"\nOffline MSE (raw): {offline_mse_raw:.2f}")
    if ema_model is not None:
        offline_mse_ema = compute_offline_mse(ema_model, ns, na, ua, dataset, device)
        print(f"Offline MSE (EMA): {offline_mse_ema:.2f}")

    # ---- Eval ----
    if not args.skip_eval:
        eval_summary = {}

        # Raw weights eval
        print(f"\n{'='*60}")
        print(f"EVAL: raw weights, {args.n_eval_episodes} episodes")
        print(f"{'='*60}")
        raw_eval = eval_fixed_buffer(raw_model, ns, ua, device, args.n_eval_episodes)
        print(f"  Raw: mean={raw_eval['mean_max_reward']:.4f} std={raw_eval['std_max_reward']:.4f} "
              f"median={raw_eval['median']:.4f} ep>0.5={raw_eval['ep_gt_05']}/{args.n_eval_episodes} "
              f"ep>0.8={raw_eval['ep_gt_08']}")
        eval_summary["raw_weights"] = raw_eval

        # EMA weights eval
        if ema_model is not None:
            print(f"\n{'='*60}")
            print(f"EVAL: EMA weights, {args.n_eval_episodes} episodes")
            print(f"{'='*60}")
            ema_eval = eval_fixed_buffer(ema_model, ns, ua, device, args.n_eval_episodes)
            print(f"  EMA: mean={ema_eval['mean_max_reward']:.4f} std={ema_eval['std_max_reward']:.4f} "
                  f"median={ema_eval['median']:.4f} ep>0.5={ema_eval['ep_gt_05']}/{args.n_eval_episodes} "
                  f"ep>0.8={ema_eval['ep_gt_08']}")
            eval_summary["ema_weights"] = ema_eval

        # Save eval summary
        eval_path = os.path.join(run_dir, "eval_summary.json")
        with open(eval_path, "w") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"Eval summary saved: {eval_path}")

    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Step D2: B4 training")
    parser.add_argument("--run-name", type=str, required=True,
                        help="B4a_ema_constant / B4b_noema_cosine / B4c_ema_cosine")
    parser.add_argument("--use-ema", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--lr-scheduler", type=str, required=True, choices=["constant", "cosine"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="artifacts_keep")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    train_and_eval(args)


if __name__ == "__main__":
    main()
