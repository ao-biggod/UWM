#!/usr/bin/env python3
"""Quick single-model deterministic eval with Config C, designed for 10-ep screening.

Usage:
  cd /root/autodl-tmp/UWM_pushT
  source /root/miniconda3/bin/activate robodiff
  export SDL_VIDEODRIVER=dummy
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH=unified-world-model-main:diffusion_policy-main:$PYTHONPATH

  python scripts/quick_deterministic_eval.py \
    --checkpoint outputs/uwm_pusht_r1_loss_off/quick_5k/latest.pt \
    --device cuda:0 --seeds 100000 100009 \
    --output outputs/uwm_pusht_r1_loss_off/eval_10ep_det
"""

import argparse, json, os, sys, time, random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_cudnn_deterministic():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def build_model(device, ckpt):
    from models.uwm import UnifiedWorldModel
    from models.uwm.obs_encoder import UWMObservationEncoder

    cfg = ckpt.get("config", {})
    cond_type = ckpt.get("conditioning_type") or cfg.get("conditioning_type", "adaln")
    dp_only = ckpt.get("dp_only", cfg.get("dp_only", False))
    dyn_weight = cfg.get("dynamics_loss_weight", 1.0)
    sa_mask = cfg.get("self_attn_mask", None)

    sm = {"obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                  "agent_pos": {"shape": [2], "type": "low_dim"}},
          "action": {"shape": [2]}}
    oe = UWMObservationEncoder(
        shape_meta=sm, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False,
        vision_backbone="resnet", use_low_dim=True, use_language=False,
    )
    m = UnifiedWorldModel(
        action_len=16, action_dim=2, obs_encoder=oe,
        embed_dim=768, timestep_embed_dim=512,
        latent_patch_shape=[2, 4, 4], depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, num_registers=8,
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=False,
        conditioning_type=cond_type,
        dp_only=dp_only,
        dynamics_loss_weight=dyn_weight,
        self_attn_mask=sa_mask,
    )
    m.load_state_dict(ckpt["model"], strict=False)
    return m.to(device)


def make_env(seed):
    env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
    env.seed(seed)
    return env


def eval_one(model, device, action_scale, action_offset, ap_scale, ap_offset, seed):
    model.eval()
    env = make_env(seed)
    obs = env.reset()
    rewards = []
    step = 0

    while step < 300:
        img = obs["image"]
        agent_pos = obs["agent_pos"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(agent_pos, np.ndarray):
            agent_pos = torch.from_numpy(agent_pos)

        img_hwc = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        ap = agent_pos.float().to(device)
        ap = (ap - torch.tensor(ap_offset, device=device).float()) / torch.tensor(ap_scale, device=device).float()

        obs_model = {"image": torch.from_numpy(img_hwc).to(device).unsqueeze(0),
                     "agent_pos": ap.unsqueeze(0)}

        with torch.no_grad():
            action_norm = model.sample(obs_model)[0]

        action_raw = (action_norm
                      * torch.tensor(action_scale, device=device).float()
                      + torch.tensor(action_offset, device=device).float())
        action_raw_np = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)

        obs, reward, done, _ = env.step(action_raw_np[:8])
        rewards.append(float(reward))
        if np.all(done):
            break
        step += 1

    return float(max(rewards)) if rewards else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs=2, default=[100000, 100009])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    set_cudnn_deterministic()
    seeds = list(range(args.seeds[0], args.seeds[1] + 1))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    an = ckpt["action_normalizer"]
    action_scale = np.array(an["scale"])
    action_offset = np.array(an["offset"])
    ln = ckpt["lowdim_normalizer"]["agent_pos"]
    ap_scale = np.array(ln["scale"])
    ap_offset = np.array(ln["offset"])

    model = build_model(device, ckpt)
    del ckpt

    print(f"eval: {len(seeds)} eps, seeds {seeds[0]}-{seeds[-1]}, Config C")
    scores = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        seed_everything(seed)
        s = eval_one(model, device, action_scale, action_offset, ap_scale, ap_offset, seed)
        scores.append(s)
        print(f"  ep {i:2d} seed={seed}: {s:.4f}")

    elapsed = time.time() - t0
    arr = np.array(scores)
    result = {
        "checkpoint": args.checkpoint,
        "seeds": list(seeds),
        "scores": [float(x) for x in scores],
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "ep_gt_0_5": int((arr > 0.5).sum()),
        "elapsed_s": elapsed,
    }
    print(f"\n  mean={arr.mean():.4f} median={np.median(arr):.4f} std={arr.std():.4f} ep>0.5={(arr>0.5).sum()}/{len(seeds)}")

    out_path = args.output or f"{os.path.dirname(args.checkpoint)}/eval_det_10ep.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
