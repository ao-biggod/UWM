#!/usr/bin/env python3
"""Loss scale & gradient conflict diagnostic for cross_attn joint training.

Measures on 128 training batches (no training, just forward+backward):
  - action_loss / dynamics_loss mean, std, ratio
  - grad_norm for shared transformer backbone
  - cosine similarity between grad(action_loss) and grad(dynamics_loss)
  - effective_video_grad_ratio for candidate lambda_video values

Usage:
  python scripts/diag_loss_grad_scale.py \
    --checkpoint outputs/uwm_pusht_crossattn_joint_C/main_20k/latest.pt \
    --num-batches 128 --batch-size 16 --device cuda:0
"""

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))


def collate_fn(batch):
    if isinstance(batch[0], dict):
        return {k: collate_fn([b[k] for b in batch]) for k in batch[0]}
    return torch.stack(batch)


def process_batch(batch, device):
    obs_horizon, action_horizon = 2, 16
    action_start = obs_horizon - 1
    action_end = action_start + action_horizon
    curr_obs = {k: v[:, :action_start + 1].to(device) for k, v in batch["obs"].items()}
    next_obs = {k: v[:, action_end:].to(device) for k, v in batch["obs"].items()}
    actions = batch["action"][:, action_start:action_end].to(device)
    return curr_obs, next_obs, actions


def build_model(device, ckpt_path):
    from models.uwm import UnifiedWorldModel
    from models.uwm.obs_encoder import UWMObservationEncoder

    ckpt = torch.load(ckpt_path, map_location="cpu")
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
    model = UnifiedWorldModel(
        action_len=16, action_dim=2, obs_encoder=oe,
        embed_dim=768, timestep_embed_dim=512,
        latent_patch_shape=[2, 4, 4], depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, num_registers=8,
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=True,
        conditioning_type=cond_type, dp_only=dp_only,
        dynamics_loss_weight=dyn_weight, self_attn_mask=sa_mask,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    del ckpt
    return model.to(device)


def compute_losses(model, curr_obs, next_obs, actions):
    """Run forward and return action_loss, dynamics_loss as tensors."""
    batch_size, device = actions.shape[0], actions.device

    if model.conditioning_type == "cross_attn":
        obs_memory = model.obs_encoder.encode_obs_memory(curr_obs)
        next_obs_latent = model.obs_encoder.encode_next_obs(next_obs)
        obs_global = None
    else:
        obs_global, next_obs_latent = model.obs_encoder.encode_curr_and_next_obs(curr_obs, next_obs)
        obs_memory = None

    action_noise = torch.randn_like(actions)
    action_t = torch.randint(0, model.num_train_steps, (batch_size,), device=device).long()
    noisy_action = model.noise_scheduler.add_noise(actions, action_noise, action_t)

    next_obs_noise = torch.randn_like(next_obs_latent)
    next_obs_t = torch.randint(0, model.num_train_steps, (batch_size,), device=device).long()
    noisy_next_obs = model.noise_scheduler.add_noise(next_obs_latent, next_obs_noise, next_obs_t)

    action_noise_pred, next_obs_noise_pred = model.noise_pred_net(
        obs_global, noisy_action, action_t, noisy_next_obs, next_obs_t, obs_memory
    )
    action_loss = F.mse_loss(action_noise_pred, action_noise)
    dynamics_loss = F.mse_loss(next_obs_noise_pred, next_obs_noise)
    return action_loss, dynamics_loss


def get_shared_params(model):
    """Return list of (name, param) for shared backbone params in noise_pred_net."""
    params = []
    for name, p in model.named_parameters():
        if "noise_pred_net" in name and p.requires_grad:
            params.append((name, p))
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--zarr-path", type=str,
                        default="/root/autodl-tmp/UWM_pushT/diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("Loss Scale & Gradient Conflict Diagnostic")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  num_batches: {args.num_batches}")
    print(f"  batch_size: {args.batch_size}")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    model = build_model(device, args.checkpoint)
    shared_params = get_shared_params(model)
    param_names = [n for n, _ in shared_params]
    params = [p for _, p in shared_params]
    print(f"  Shared backbone params: {len(params)}")

    if model.dp_only:
        print("ERROR: checkpoint is dp_only, need joint model")
        sys.exit(1)

    # Load dataset
    print("\n[2/4] Loading dataset...")
    from torch.utils.data import DataLoader
    from datasets.pusht import make_pusht_dataset

    dataset, _ = make_pusht_dataset(
        name="pusht_diag", zarr_path=args.zarr_path,
        shape_meta={"obs": {"image": {"shape": [96,96,3], "type": "rgb"},
                            "agent_pos": {"shape": [2], "type": "low_dim"}},
                    "action": {"shape": [2]}},
        seq_len=19, val_ratio=0.02, max_train_episodes=90, seed=42,
        normalize_action=True, normalize_lowdim=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)

    # Collect stats
    print(f"\n[3/4] Running {args.num_batches} batches...")
    action_losses = []
    dynamics_losses = []
    all_cos = []
    all_act_grad_norms = []
    all_dyn_grad_norms = []
    per_layer_cos = defaultdict(list)
    per_layer_act_norm = defaultdict(list)
    per_layer_dyn_norm = defaultdict(list)

    data_iter = iter(loader)
    model.train()
    t0 = time.time()

    for batch_idx in range(args.num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        curr_obs, next_obs, actions = process_batch(batch, device)

        # Forward: compute losses separately
        model.zero_grad()
        action_loss, dynamics_loss = compute_losses(model, curr_obs, next_obs, actions)
        action_losses.append(action_loss.item())
        dynamics_losses.append(dynamics_loss.item())

        # Gradients for action_loss
        model.zero_grad()
        action_loss.backward(retain_graph=True)
        act_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]

        # Gradients for dynamics_loss
        model.zero_grad()
        dynamics_loss.backward()
        dyn_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        model.zero_grad()

        # Per-param stats
        batch_cos = []
        for i, name in enumerate(param_names):
            ga = act_grads[i].flatten()
            gd = dyn_grads[i].flatten()
            ga_norm = ga.norm().item()
            gd_norm = gd.norm().item()
            cos = (ga @ gd) / (ga_norm * gd_norm + 1e-12)

            batch_cos.append(cos.item())
            all_act_grad_norms.append(ga_norm)
            all_dyn_grad_norms.append(gd_norm)

            # Layer aggregation
            parts = name.split(".")
            if "blocks" in parts:
                idx = parts.index("blocks")
                layer_key = f"blocks.{parts[idx+1]}"
            else:
                layer_key = "other"
            per_layer_cos[layer_key].append(cos.item())
            per_layer_act_norm[layer_key].append(ga_norm)
            per_layer_dyn_norm[layer_key].append(gd_norm)

        all_cos.extend(batch_cos)

        if batch_idx < 3 or batch_idx % 20 == 0:
            mean_cos_batch = np.mean(batch_cos)
            print(f"  batch {batch_idx:3d}: act={action_loss.item():.4f} dyn={dynamics_loss.item():.4f} "
                  f"cos={mean_cos_batch:.4f}")

    elapsed = time.time() - t0
    print(f"  Completed {args.num_batches} batches in {elapsed:.1f}s ({elapsed/args.num_batches:.2f}s/batch)")

    # Summarize
    act_arr = np.array(action_losses)
    dyn_arr = np.array(dynamics_losses)
    ratio_arr = dyn_arr / (act_arr + 1e-12)
    cos_arr = np.array(all_cos)
    act_gn_arr = np.array(all_act_grad_norms)
    dyn_gn_arr = np.array(all_dyn_grad_norms)
    gn_ratio_arr = dyn_gn_arr / (act_gn_arr + 1e-12)

    mean_gn_ratio = np.mean(gn_ratio_arr)

    print("\n[4/4] Results")
    print("=" * 60)
    print("\nLoss Scale:")
    print(f"  action_loss:   mean={act_arr.mean():.4f}  std={act_arr.std():.4f}  "
          f"median={np.median(act_arr):.4f}")
    print(f"  dynamics_loss: mean={dyn_arr.mean():.4f}  std={dyn_arr.std():.4f}  "
          f"median={np.median(dyn_arr):.4f}")
    print(f"  ratio dyn/act: mean={ratio_arr.mean():.2f}  std={ratio_arr.std():.2f}  "
          f"median={np.median(ratio_arr):.2f}")

    print("\nGradient Stats (shared backbone):")
    print(f"  ||grad_act||:  mean={act_gn_arr.mean():.6f}  median={np.median(act_gn_arr):.6f}")
    print(f"  ||grad_dyn||:  mean={dyn_gn_arr.mean():.6f}  median={np.median(dyn_gn_arr):.6f}")
    print(f"  ratio dyn/act: mean={gn_ratio_arr.mean():.4f}  median={np.median(gn_ratio_arr):.4f}  "
          f"min={gn_ratio_arr.min():.4f}  max={gn_ratio_arr.max():.4f}")

    print(f"\n  Cosine similarity: mean={cos_arr.mean():.4f}  median={np.median(cos_arr):.4f}  "
          f"std={cos_arr.std():.4f}")
    print(f"  Min: {cos_arr.min():.4f}  Max: {cos_arr.max():.4f}")
    neg_frac = (cos_arr < 0).mean()
    low_frac = (cos_arr < 0.3).mean()
    print(f"  cos < 0:  {neg_frac:.2%}")
    print(f"  cos < 0.3: {low_frac:.2%}")

    # Per-layer summary
    print("\nPer-layer gradient stats:")
    print(f"  {'layer':<12} {'cos':>8} {'||grad_act||':>14} {'||grad_dyn||':>14} {'ratio':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*14} {'-'*8}")
    sorted_layers = sorted(per_layer_cos.keys(), key=lambda x: (x.split(".")[0] != "blocks", x))
    for lk in sorted_layers:
        c = np.mean(per_layer_cos[lk])
        an = np.mean(per_layer_act_norm[lk])
        dn = np.mean(per_layer_dyn_norm[lk])
        r = dn / (an + 1e-12)
        print(f"  {lk:<12} {c:8.4f} {an:14.6f} {dn:14.6f} {r:8.4f}")

    # Effective video grad ratio for candidate lambdas
    print("\nEffective video grad ratio for candidate lambda_video:")
    print(f"  {'lambda':>10} {'eff_ratio':>12} {'description'}")
    print(f"  {'-'*10} {'-'*12} {'-'*40}")
    candidates = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10]
    for lam in candidates:
        eff = lam * mean_gn_ratio
        desc = ""
        if eff < 0.03:
            desc = "<3% — negligible, likely same as R1"
        elif eff < 0.10:
            desc = "3-10% — auxiliary signal, unlikely to dominate"
        elif eff < 0.30:
            desc = "10-30% — moderate, may help or hurt"
        else:
            desc = ">30% — likely too strong, may degrade policy"
        print(f"  {lam:10.4f} {eff:12.4f} {'— ' + desc}")

    # Save
    output = {
        "checkpoint": args.checkpoint,
        "num_batches": args.num_batches,
        "loss_scale": {
            "action_loss_mean": float(act_arr.mean()),
            "action_loss_std": float(act_arr.std()),
            "dynamics_loss_mean": float(dyn_arr.mean()),
            "dynamics_loss_std": float(dyn_arr.std()),
            "ratio_dyn_over_act_mean": float(ratio_arr.mean()),
            "ratio_dyn_over_act_median": float(np.median(ratio_arr)),
        },
        "gradient_stats": {
            "grad_norm_action_mean": float(act_gn_arr.mean()),
            "grad_norm_dynamics_mean": float(dyn_gn_arr.mean()),
            "grad_norm_ratio_mean": float(gn_ratio_arr.mean()),
            "grad_norm_ratio_median": float(np.median(gn_ratio_arr)),
            "cosine_similarity_mean": float(cos_arr.mean()),
            "cosine_similarity_median": float(np.median(cos_arr)),
            "cosine_negative_fraction": float(neg_frac),
            "cosine_below_0_3_fraction": float(low_frac),
        },
        "per_layer": {lk: {
            "cos": float(np.mean(per_layer_cos[lk])),
            "grad_norm_action": float(np.mean(per_layer_act_norm[lk])),
            "grad_norm_dynamics": float(np.mean(per_layer_dyn_norm[lk])),
            "ratio": float(np.mean(per_layer_dyn_norm[lk]) / (np.mean(per_layer_act_norm[lk]) + 1e-12)),
        } for lk in sorted_layers},
        "effective_ratio_table": {str(lam): float(lam * mean_gn_ratio) for lam in candidates},
    }

    out_path = f"outputs/diag_loss_grad_scale_{Path(args.checkpoint).parent.name}.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
