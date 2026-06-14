#!/usr/bin/env python3
"""Evaluate DP ablation checkpoints (R1-R3) with deterministic seeding.

Evaluates both raw and EMA weights (if EMA available).
Config C: norm_agent_pos=True, clip_sample=False
"""
import argparse, json, os, sys, time, random
from pathlib import Path
from functools import partial
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device, clip_sample=False):
    from models.dp import ImageDiffusionPolicy, ImageObservationEncoder
    from models.dp import TransformerNoisePredictionNet
    shape_meta = {"obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                           "agent_pos": {"shape": [2], "type": "low_dim"}},
                   "action": {"shape": [2]}}
    obs_encoder = ImageObservationEncoder(shape_meta=shape_meta, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False, pretrained_weights=None,
        use_low_dim=True, use_language=False)
    model = ImageDiffusionPolicy(action_len=16, action_dim=2, obs_encoder=obs_encoder,
        noise_pred_net=partial(TransformerNoisePredictionNet, input_len=16, input_dim=2,
            timestep_embed_dim=256, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True),
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample)
    return model.to(device)


def get_lowdim_normalizer(ckpt, zarr_path):
    if "lowdim_normalizer" in ckpt:
        ln = ckpt["lowdim_normalizer"]
        if "agent_pos" in ln:
            return np.array(ln["agent_pos"]["scale"]), np.array(ln["agent_pos"]["offset"])
    import zarr
    z = zarr.open(zarr_path, "r")
    ep_ends = z["meta/episode_ends"][:]
    state = z["data/state"][:ep_ends[89], :2]
    scale = (state.max(axis=0) - state.min(axis=0)) / 2.0
    offset = (state.max(axis=0) + state.min(axis=0)) / 2.0
    return scale, offset


def eval_deterministic(model, device, action_scale, action_offset,
                       ap_scale, ap_offset, seeds):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    model.eval()
    scores = []; t0 = time.time()
    for i, seed in enumerate(seeds):
        seed_everything(seed)
        env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
        env.seed(seed); obs = env.reset()
        rewards = []; done = False; step = 0
        while not done and step < 300:
            img = obs["image"]; agent_pos = obs["agent_pos"]
            if isinstance(img, np.ndarray): img = torch.from_numpy(img)
            if isinstance(agent_pos, np.ndarray): agent_pos = torch.from_numpy(agent_pos)
            img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            ap = agent_pos.float().to(device)
            ap = (ap - torch.tensor(ap_offset, device=device).float()) / torch.tensor(ap_scale, device=device).float()
            obs_dict = {"image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
                        "agent_pos": ap.unsqueeze(0)}
            with torch.no_grad():
                action_norm = model.sample(obs_dict)[0]
            action_raw = (action_norm * torch.tensor(action_scale, device=device).float()
                          + torch.tensor(action_offset, device=device).float())
            action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
            obs, reward, done, info = env.step(action_raw_np[:8])
            rewards.append(float(reward)); done = bool(np.all(done)); step += 1
        max_r = float(max(rewards)) if rewards else 0.0
        scores.append(max_r)
        if i < 5 or i % 10 == 0 or i == len(seeds) - 1:
            print(f"  Ep {i:3d} (seed={seed}): {max_r:.4f}", flush=True)
    elapsed = time.time() - t0
    arr = np.array(scores)
    print(f"  {len(seeds)}eps in {elapsed:.0f}s: mean={arr.mean():.4f} median={np.median(arr):.4f} "
          f"ep>0.5={(arr>0.5).sum()}/{len(seeds)}", flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--zarr-path", type=str, default="diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-eps", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-tag", type=str, default="unknown")
    args = parser.parse_args()

    device = torch.device(args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.n_eps))
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    an = ckpt["action_normalizer"]
    action_scale = np.array(an["scale"]); action_offset = np.array(an["offset"])
    ap_scale, ap_offset = get_lowdim_normalizer(ckpt, args.zarr_path)
    has_ema = "ema_model" in ckpt

    print(f"Eval: {args.run_tag}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Seeds: {seeds[0]}-{seeds[-1]}")
    print(f"  EMA available: {has_ema}")

    all_results = {}

    # Raw weights
    print(f"\n--- Raw weights ---")
    model = build_model(device, clip_sample=False)
    model.load_state_dict(ckpt["model"])
    scores = eval_deterministic(model, device, action_scale, action_offset,
                                ap_scale, ap_offset, seeds)
    arr = np.array(scores)
    all_results["raw"] = {"mean": float(arr.mean()), "median": float(np.median(arr)),
                          "std": float(arr.std()), "ep_gt_05": int((arr>0.5).sum()),
                          "scores": [float(x) for x in scores]}

    # EMA weights
    if has_ema:
        print(f"\n--- EMA weights ---")
        model_ema = build_model(device, clip_sample=False)
        model_ema.load_state_dict(ckpt["model"])  # load raw first
        # Apply EMA
        ema_state = ckpt["ema_model"]
        with torch.no_grad():
            for name, param in model_ema.named_parameters():
                if name in ema_state:
                    param.data.copy_(ema_state[name].to(device))
        scores_ema = eval_deterministic(model_ema, device, action_scale, action_offset,
                                        ap_scale, ap_offset, seeds)
        arr_ema = np.array(scores_ema)
        all_results["ema"] = {"mean": float(arr_ema.mean()), "median": float(np.median(arr_ema)),
                              "std": float(arr_ema.std()), "ep_gt_05": int((arr_ema>0.5).sum()),
                              "scores": [float(x) for x in scores_ema]}

    # Save
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*58}")
    print(f"{'Weight':<15} {'Mean':>8} {'Median':>8} {'Std':>8} {'ep>0.5':>10}")
    print("-" * 48)
    for k, v in all_results.items():
        print(f"  {k:<13} {v['mean']:8.4f} {v['median']:8.4f} {v['std']:8.4f} "
              f"{v['ep_gt_05']:>8}/{args.n_eps}")
    print(f"\nSaved: {os.path.join(args.output_dir, 'eval_results.json')}")


if __name__ == "__main__":
    main()
