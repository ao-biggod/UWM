#!/usr/bin/env python3
"""B3: Local DiT + 20D keypoint observation. Minimal change from B2.

Uses same model backbone as B2 (LowdimStatePolicyV2), same training recipe,
no EMA, no cosine LR, no video loss. Only changes obs from 5D → 20D.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util
spec = importlib.util.spec_from_file_location("stepB_retrain_lowdim", str(Path(__file__).resolve().parent / "stepB_retrain_lowdim.py"))
stepB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stepB)


def load_zarr_data_20d(zarr_path):
    """Load PushT zarr, return episodes with 20D obs = [keypoint(18), agent_pos(2)]."""
    import zarr
    z = zarr.open(zarr_path, "r")
    keypoint = z["data/keypoint"][:]   # (N, 9, 2)
    state = z["data/state"][:]         # (N, 5)
    action = z["data/action"][:]       # (N, 2)
    agent_pos = state[:, :2]           # (N, 2)
    obs_20d = np.concatenate([keypoint.reshape(keypoint.shape[0], -1), agent_pos], axis=-1)  # (N, 20)
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
    return torch.utils.data.TensorDataset(
        torch.tensor(np.stack(all_obs), dtype=torch.float32),
        torch.tensor(np.stack(all_act), dtype=torch.float32),
    )


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


class MinMaxNormalizer:
    def __init__(self):
        self.offset = None; self.scale = None
    def fit(self, data):
        dmin = data.min(axis=0); dmax = data.max(axis=0)
        self.offset = (dmax + dmin) / 2.0
        self.scale = (dmax - dmin) / 2.0
        self.scale[self.scale < 1e-6] = 1.0
        return self
    def normalize(self, data): return (data - self.offset) / self.scale
    def unnormalize(self, data): return data * self.scale + self.offset
    def to_dict(self): return {"offset": self.offset.tolist(), "scale": self.scale.tolist()}


def get_full_state_from_env(env):
    agent_p = np.array(env.agent.position)
    block_p = np.array(env.block.position)
    block_ang = env.block.angle
    return np.concatenate([agent_p, block_p, [block_ang]])


def get_keypoint_20d_from_env(env):
    """Extract 20D keypoint observation from PushTKeypointsEnv internals.
    Matches PushTKeypointsEnv._get_obs() format. """
    from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
    if not hasattr(env, 'kp_manager'):
        kp_kwargs = env.__class__.genenerate_keypoint_manager_params()
        env.kp_manager = PymunkKeypointManager(
            local_keypoint_map=kp_kwargs['local_keypoint_map'],
            color_map=kp_kwargs['color_map'])
    obj_map = {'block': env.block}
    kp_map = env.kp_manager.get_keypoints_global(pose_map=obj_map, is_obj=True)
    kps = np.concatenate(list(kp_map.values()), axis=0)
    agent_pos = np.array(env.agent.position)
    obs = np.concatenate([kps.flatten(), agent_pos])  # (20,)
    return obs


def eval_fixed_buffer_20d(model, norm_state, unnorm_action, device, n_eps=50, n_action_exec=8):
    """Fixed-buffer eval using PushTKeypointsEnv for 20D keypoint obs."""
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
        obs = env.reset()

        obs_buffer = deque(maxlen=2)
        rewards = []
        done = False
        step = 0

        # obs from MultiStepWrapper is (2, 40) = [data(20), mask(20)] × 2 frames
        Do = obs.shape[-1] // 2  # 20
        obs_buffer.append(obs[..., 0, :Do])  # first frame data
        obs_buffer.append(obs[..., 1, :Do])  # second frame data

        while not done and step < 300:
            state_np = np.stack(list(obs_buffer))  # (2, 20)
            state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            state_norm = (state_t - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()

            with torch.no_grad():
                action_norm = model.sample(state_norm)[0]
            action_raw = action_norm * torch.tensor(unnorm_action.scale, device=device).float() + torch.tensor(unnorm_action.offset, device=device).float()
            action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
            exec_actions = action_raw[:n_action_exec]

            obs, reward, done, info = env.step(exec_actions)
            rewards.append(float(reward))
            done = bool(np.all(done))

            Do = obs.shape[-1] // 2
            obs_buffer.append(obs[..., 1, :Do])
            step += 1

        max_r = float(max(rewards)) if rewards else 0.0
        results.append(max_r)
        if ep < 5 or ep % 10 == 0:
            print(f"  Ep {ep:3d}: max_reward={max_r:.4f}", flush=True)

    scores = np.array(results)
    print(f"\n  B3 eval: mean={scores.mean():.4f} std={scores.std():.4f} median={np.median(scores):.4f} "
          f"ep>0.5={(scores>0.5).sum()}/{n_eps} ep>0.8={(scores>0.8).sum()}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="artifacts_keep/B3_keypoint20_local_dit_20k")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    obs_horizon, action_horizon, seq_len = 2, 16, 19
    obs_dim = 20
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"B3: Local DiT + {obs_dim}D Keypoint Observation")
    print(f"  Train steps: {args.num_steps}")
    print(f"  Same architecture as B2, only obs changes")
    print(f"{'='*60}")

    # ---- E1-2: Dataset parity + audit ----
    zarr_path = "diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
    episodes = load_zarr_data_20d(zarr_path)
    print(f"Loaded {len(episodes)} episodes")

    all_obs = np.concatenate([s for s, _ in episodes[:90]], axis=0)
    all_action = np.concatenate([a for _, a in episodes[:90]], axis=0)
    print(f"obs shape: {all_obs.shape}, action shape: {all_action.shape}")
    print(f"obs range: [{all_obs.min(axis=0)[:3]}...], [{all_obs.max(axis=0)[:3]}...]")
    print(f"obs mean/std: {all_obs.mean():.1f} / {all_obs.std():.1f}")
    print(f"action range: [{all_action.min(axis=0)}, {all_action.max(axis=0)}]")
    print(f"action mean/std: {all_action.mean():.1f} / {all_action.std():.1f}")

    # Normalizer
    norm_state = MinMaxNormalizer().fit(all_obs)
    norm_action = MinMaxNormalizer().fit(all_action)
    obs_normed = norm_state.normalize(all_obs)
    act_normed = norm_action.normalize(all_action)
    print(f"normalized obs range: [{obs_normed.min(axis=0)[:3]}...], [{obs_normed.max(axis=0)[:3]}...]")
    print(f"normalized action range: [{act_normed.min(axis=0)}, {act_normed.max(axis=0)}]")

    # Print 3 sample comparisons
    print(f"\n  Sample obs[0,0,:5]: {all_obs[0,:5]}")
    print(f"  Sample obs[0,1,:5]: {all_obs[1,:5]}")
    print(f"  Sample action[0,:3]: {all_action[0,:3]}")

    # ---- E1-2: Parity check with DP official dataset ----
    if not args.skip_parity:
        from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
        dp_ds = PushTLowdimDataset(
            zarr_path=zarr_path, horizon=16, pad_before=obs_horizon-1,
            pad_after=action_horizon-1, seed=42, val_ratio=0.0, max_train_episodes=90)
        dp_sample = dp_ds[0]
        dp_obs = dp_sample['obs'].numpy()  # (16, 20)
        dp_action = dp_sample['action'].numpy()  # (16, 2)

        our_sample = build_dataset(episodes, 90, seq_len, obs_horizon, action_horizon)
        our_obs, our_act = our_sample[0]
        our_obs_np = our_obs.numpy()  # (2, 20)
        our_act_np = our_act.numpy()  # (19, 2)

        # DP dataset pads horizon to 16, our dataset returns 2-frame obs
        # Compare first 2 obs frames
        obs_diff = np.abs(dp_obs[:2] - our_obs_np).max()
        act_diff = np.abs(dp_action[:2] - our_act_np[:2]).max()
        print(f"\n  Parity check (first 2 frames):")
        print(f"    obs max_abs_diff: {obs_diff:.10f}")
        print(f"    action max_abs_diff: {act_diff:.10f}")
        print(f"    Parity: {'PASSED' if obs_diff < 1e-5 and act_diff < 1e-5 else 'FAILED'}")

    # ---- Dataset ----
    dataset = build_dataset(episodes, 90, seq_len, obs_horizon, action_horizon)
    print(f"\nTraining samples: {len(dataset)}")

    s_offset = torch.tensor(norm_state.offset, device=device).float()
    s_scale = torch.tensor(norm_state.scale, device=device).float()
    a_offset = torch.tensor(norm_action.offset, device=device).float()
    a_scale = torch.tensor(norm_action.scale, device=device).float()

    # ---- Model (same as B2 but obs_dim=20) ----
    B3Model = stepB.LowdimStatePolicyV2
    model = B3Model(obs_dim=obs_dim, clip_sample=True, num_inference_steps=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M (B2 was 4.9M, now {obs_dim}D input)")

    # ---- Training ----
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

        obs_state = (obs_np.to(device) - s_offset) / s_scale
        action_target = act_np[:, obs_horizon - 1: obs_horizon - 1 + action_horizon].to(device)
        action_norm = (action_target - a_offset) / a_scale

        model.train()
        loss, info = model(obs_state, action_norm)
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

    # ---- Offline prediction check (E1-3) ----
    model.eval()
    obs_test, act_test = dataset[:64]
    obs_norm = (obs_test.to(device) - s_offset) / s_scale
    with torch.no_grad():
        act_pred_norm = model.sample(obs_norm)
    act_pred_raw = act_pred_norm * a_scale + a_offset
    act_gt_raw = act_test[:, obs_horizon - 1: obs_horizon - 1 + action_horizon].to(device)

    raw_mse_all = F.mse_loss(act_pred_raw, act_gt_raw).item()
    raw_mse_8 = F.mse_loss(act_pred_raw[:, :8], act_gt_raw[:, :8]).item()
    t0_mse = F.mse_loss(act_pred_raw[:, 0], act_gt_raw[:, 0]).item()
    t7_mse = F.mse_loss(act_pred_raw[:, 7], act_gt_raw[:, 7]).item()
    t15_mse = F.mse_loss(act_pred_raw[:, 15], act_gt_raw[:, 15]).item()

    action_std = all_action.std(axis=0)
    rmse = np.sqrt(raw_mse_all)
    rmse_ratio = [rmse / action_std[0], rmse / action_std[1]]

    print(f"\n  Offline MSE (N=64):")
    print(f"    raw_mse_all:  {raw_mse_all:.1f}")
    print(f"    raw_mse_8:    {raw_mse_8:.1f}")
    print(f"    t=0 MSE:      {t0_mse:.1f}")
    print(f"    t=7 MSE:      {t7_mse:.1f}")
    print(f"    t=15 MSE:     {t15_mse:.1f}")
    print(f"    RMSE/std:     {rmse_ratio}")

    # ---- Save ----
    save_path = os.path.join(args.output_dir, "latest.pt")
    torch.save({
        "model": model.state_dict(),
        "step": args.num_steps - 1,
        "norm_type": "minmax",
        "obs_dim": obs_dim,
        "clip_sample": True,
        "action_normalizer": norm_action.to_dict(),
        "state_normalizer": norm_state.to_dict(),
        "offline_mse": raw_mse_all,
        "config": {"obs_horizon": obs_horizon, "action_horizon": action_horizon,
                   "seq_len": seq_len, "obs_dim": obs_dim},
    }, save_path)
    print(f"\nSaved: {save_path}")

    # ---- Eval (E1-4) ----
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print(f"B3 eval: {args.n_eval_episodes} episodes, fixed-buffer")
        print(f"{'='*60}")
        eval_scores = eval_fixed_buffer_20d(
            model, norm_state, norm_action, device, args.n_eval_episodes)
        eval_summary = {
            "mean": float(eval_scores.mean()),
            "std": float(eval_scores.std()),
            "median": float(np.median(eval_scores)),
            "p25": float(np.percentile(eval_scores, 25)),
            "p75": float(np.percentile(eval_scores, 75)),
            "min": float(eval_scores.min()),
            "max": float(eval_scores.max()),
            "ep_gt_05": int((eval_scores > 0.5).sum()),
            "ep_gt_08": int((eval_scores > 0.8).sum()),
            "offline_mse_all": raw_mse_all,
            "offline_mse_8": raw_mse_8,
        }
        with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"Eval summary saved")


if __name__ == "__main__":
    main()
