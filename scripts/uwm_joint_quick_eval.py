#!/usr/bin/env python3
"""Quick UWM joint eval: A and C configs, 10ep sanity, then 50ep for best."""
import sys, os, time, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


def build_uwm(device, clip_sample):
    from models.uwm import UnifiedWorldModel
    from models.uwm.obs_encoder import UWMObservationEncoder
    sm = {"obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                  "agent_pos": {"shape": [2], "type": "low_dim"}},
          "action": {"shape": [2]}}
    oe = UWMObservationEncoder(shape_meta=sm, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False,
        vision_backbone="resnet", use_low_dim=True, use_language=False)
    m = UnifiedWorldModel(action_len=16, action_dim=2, obs_encoder=oe,
        embed_dim=768, timestep_embed_dim=512,
        latent_patch_shape=[2, 4, 4], depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, num_registers=8,
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample)
    return m.to(device)


def eval_config(label, norm_ap, clip_sample, ckpt, action_scale, action_offset,
                ap_scale, ap_offset, device, n_eps=10, seed_start=100000):
    print(f"\n[{label}] norm_ap={norm_ap} clip_sample={clip_sample} seeds=[{seed_start},{seed_start+n_eps-1}]")
    model = build_uwm(device, clip_sample)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    results = []; t0 = time.time()

    for ep in range(n_eps):
        seed = seed_start + ep
        env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
        env.seed(seed)
        obs = env.reset()
        rewards = []; done = False; step = 0

        while not done and step < 300:
            img = torch.from_numpy(obs["image"])
            ap = torch.from_numpy(obs["agent_pos"]).float().to(device)
            if norm_ap:
                ap_off = torch.tensor(ap_offset, device=device).float()
                ap_sc = torch.tensor(ap_scale, device=device).float()
                ap = (ap - ap_off) / ap_sc

            img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            obs_uwm = {
                "image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
                "agent_pos": ap.unsqueeze(0),
            }

            with torch.no_grad():
                action_norm = model.sample(obs_uwm)[0]
            action_raw = action_norm * torch.tensor(action_scale, device=device).float() + torch.tensor(action_offset, device=device).float()
            action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
            exec_actions = action_raw_np[:8]

            if ep == 0 and step == 0:
                print(f"  action_norm: [{action_norm.min():.3f},{action_norm.max():.3f}]")
                print(f"  action_raw:  x∈[{action_raw_np[:,0].min():.0f},{action_raw_np[:,0].max():.0f}] "
                      f"y∈[{action_raw_np[:,1].min():.0f},{action_raw_np[:,1].max():.0f}]")

            obs, reward, done, info = env.step(exec_actions)
            rewards.append(float(reward)); done = bool(np.all(done)); step += 1

        max_r = float(max(rewards)) if rewards else 0.0
        results.append(max_r)
        print(f"  Ep {ep:3d}: mreward={max_r:.4f}", flush=True)

    scores = np.array(results)
    elapsed = time.time() - t0
    print(f"  [{label}] {n_eps}eps: mean={scores.mean():.4f} median={np.median(scores):.4f} "
          f"ep>0.5={(scores>0.5).sum()}/{n_eps}  ({elapsed:.0f}s)")
    return {"mean": float(scores.mean()), "median": float(np.median(scores)),
            "std": float(scores.std()), "ep_gt_05": int((scores>0.5).sum()),
            "scores": scores.tolist()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-eps", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--config", type=str, default=None,
                        choices=["uwm_A", "uwm_C", "both"])
    parser.add_argument("--output-dir", type=str, default="outputs/eval_uwm_joint_fixed")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load("artifacts_keep/uwm_20k/checkpoint_20k_latest.pt", map_location=device)
    an = ckpt["action_normalizer"]
    action_scale = np.array(an["scale"]); action_offset = np.array(an["offset"])
    ln = ckpt["lowdim_normalizer"]["agent_pos"]
    ap_scale = np.array(ln["scale"]); ap_offset = np.array(ln["offset"])

    print(f"UWM joint eval: config={args.config} seeds={args.seed_start}-{args.seed_start+args.n_eps-1}")
    print(f"  ap_scale={ap_scale}  ap_offset={ap_offset}")
    print(f"  action_scale={action_scale}  action_offset={action_offset}")

    if args.config == "both":
        configs = [("uwm_A", True, True), ("uwm_C", True, False)]
    elif args.config == "uwm_A":
        configs = [("uwm_A", True, True)]
    else:
        configs = [("uwm_C", True, False)]

    all_results = {}
    for label, nap, cs in configs:
        r = eval_config(label, nap, cs, ckpt, action_scale, action_offset,
                        ap_scale, ap_offset, device, n_eps=args.n_eps,
                        seed_start=args.seed_start)
        all_results[label] = r

    # Summary
    print(f"\n{'='*50}")
    print(f"{'Config':<16} {'seeds':>12} {'mean':>8} {'median':>8} {'ep>0.5':>8}")
    print("-" * 50)
    for label, r in all_results.items():
        n = len(r["scores"])
        seed_range = f"{args.seed_start}-{args.seed_start+n-1}"
        print(f"  {label:<14} {seed_range:>12} {r['mean']:8.4f} {r['median']:8.4f} "
              f"{r['ep_gt_05']:>6}/{n}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
