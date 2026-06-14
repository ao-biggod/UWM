#!/usr/bin/env python3
"""Gradient conflict diagnostics for Cross-Attention UWM joint training.

Loads a cross_attn joint checkpoint, runs N batches, and reports:
  1. Per-layer self-attention mass: action→video, register→video
  2. Cosine similarity between grad(action_loss) and grad(dynamics_loss) per param
  3. Gradient norms per layer

Usage:
  cd /root/autodl-tmp/UWM_pushT
  source /root/miniconda3/bin/activate robodiff
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH=unified-world-model-main:$PYTHONPATH

  python scripts/diagnose_gradient_conflict.py \
    --checkpoint outputs/uwm_pusht_crossattn_joint_C/latest.pt \
    --num-batches 32 --batch-size 16 --device cuda:0 \
    --output outputs/diag_crossattn_joint.json
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


def process_batch(batch, obs_horizon, action_horizon, device):
    action_start = obs_horizon - 1
    action_end = action_start + action_horizon
    curr_obs = {k: v[:, :action_start + 1].to(device) for k, v in batch["obs"].items()}
    next_obs = {k: v[:, action_end:].to(device) for k, v in batch["obs"].items()}
    actions = batch["action"][:, action_start:action_end].to(device)
    return curr_obs, next_obs, actions


def build_model_from_ckpt(device, ckpt_path):
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
        conditioning_type=cond_type,
        dp_only=dp_only,
        dynamics_loss_weight=dyn_weight,
        self_attn_mask=sa_mask,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    del ckpt
    return model.to(device)


def _capture_attention_weights(module, input, output):
    """Forward hook: compute and store self-attention weights."""
    x = input[0]  # already modulated
    B, N, D = x.shape
    num_heads = module.num_heads
    head_dim = D // num_heads

    qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    # Apply mask if provided (for policy_protect variants)
    # attn_mask is passed as a positional arg to self.attn()
    attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]
    module._captured_attn = attn.detach().cpu()


def patch_model_for_attention_capture(model):
    """Patch all self-attention layers to capture weights. Returns cleanup fn."""
    patched = []

    for block in model.noise_pred_net.blocks:
        attn_module = block.attn
        handle = attn_module.register_forward_hook(_capture_attention_weights)
        patched.append((attn_module, handle))

    def cleanup():
        for attn_module, handle in patched:
            handle.remove()
            if hasattr(attn_module, "_captured_attn"):
                del attn_module._captured_attn

    return cleanup


def collect_attention_mass(model):
    """Collect per-layer attention mass from captured weights."""
    results = []
    for i, block in enumerate(model.noise_pred_net.blocks):
        attn = block.attn._captured_attn  # [B, num_heads, N, N]
        avg = attn.mean(dim=[0, 1])  # [N, N]

        a_start, a_end = model.noise_pred_net.action_inds
        v_start, v_end = model.noise_pred_net.next_obs_inds
        action_to_video = avg[a_start:a_end, v_start:v_end].mean().item()
        video_to_video = avg[v_start:v_end, v_start:v_end].mean().item()
        register_to_video = avg[v_end:, v_start:v_end].mean().item()
        action_to_action = avg[a_start:a_end, a_start:a_end].mean().item()
        N = avg.shape[0]

        results.append({
            "layer": i,
            "action_to_video": action_to_video,
            "video_to_video": video_to_video,
            "register_to_video": register_to_video,
            "action_to_action": action_to_action,
            "num_tokens": N,
        })
    return results


def collect_gradient_stats(model, batch_tuple, num_batches, device):
    """Collect per-param gradient cosine similarity and norms over batches."""
    obs_dict, next_obs_dict, actions = batch_tuple

    # Identify shared backbone params (all noise_pred_net params)
    param_names = []
    params = []
    for name, p in model.named_parameters():
        if "noise_pred_net" in name and p.requires_grad:
            param_names.append(name)
            params.append(p)

    all_cos_sims = defaultdict(list)
    all_action_norms = defaultdict(list)
    all_dynamics_norms = defaultdict(list)

    from torch.utils.data import DataLoader
    from datasets.pusht import make_pusht_dataset

    dataset, _ = make_pusht_dataset(
        name="pusht_diag", zarr_path=zarr_path_global,
        shape_meta={"obs": {"image": {"shape": [96,96,3], "type": "rgb"},
                            "agent_pos": {"shape": [2], "type": "low_dim"}},
                    "action": {"shape": [2]}},
        seq_len=19, val_ratio=0.02, max_train_episodes=90, seed=42,
        normalize_action=True, normalize_lowdim=True,
    )
    loader = DataLoader(dataset, batch_size=bs_global, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)
    data_iter = iter(loader)

    model.train()
    for batch_idx in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        curr_obs, next_obs, actions = process_batch(batch, 2, 16, device)

        # --- action_loss grads ---
        model.zero_grad()
        loss_act, info_act = _compute_losses_separately(model, curr_obs, next_obs, actions)
        grad_action = torch.autograd.grad(
            loss_act, params, retain_graph=True, allow_unused=True
        )

        # --- dynamics_loss grads ---
        model.zero_grad()
        loss_dyn, info_dyn = _compute_losses_separately(model, curr_obs, next_obs, actions)
        grad_dynamics = torch.autograd.grad(
            loss_dyn, params, retain_graph=False, allow_unused=True
        )

        model.zero_grad()

        for i, name in enumerate(param_names):
            ga = grad_action[i]
            gd = grad_dynamics[i]
            if ga is None or gd is None:
                continue
            ga_f = ga.detach().flatten()
            gd_f = gd.detach().flatten()
            cos_sim = (ga_f @ gd_f) / (ga_f.norm() * gd_f.norm() + 1e-12)
            all_cos_sims[name].append(cos_sim.item())
            all_action_norms[name].append(ga_f.norm().item())
            all_dynamics_norms[name].append(gd_f.norm().item())

        if batch_idx < 5 or batch_idx % 10 == 0:
            mean_cos = np.mean([np.mean(v) for v in all_cos_sims.values()])
            print(f"  batch {batch_idx:3d}: mean_cos={mean_cos:.4f}  "
                  f"act_loss={loss_act.item():.4f}  dyn_loss={loss_dyn.item():.4f}")

    return dict(all_cos_sims), dict(all_action_norms), dict(all_dynamics_norms)


def _compute_losses_separately(model, curr_obs, next_obs, actions):
    """Compute action_loss and dynamics_loss separately using the model's internal logic."""
    batch_size, device = actions.shape[0], actions.device

    # Encode observations
    if model.conditioning_type == "cross_attn":
        obs_memory = model.obs_encoder.encode_obs_memory(curr_obs)
        next_obs = model.obs_encoder.encode_next_obs(next_obs)
        obs_global = None
    else:
        obs_global, next_obs = model.obs_encoder.encode_curr_and_next_obs(curr_obs, next_obs)
        obs_memory = None

    # Action noise
    action_noise = torch.randn_like(actions)
    action_t = torch.randint(0, model.num_train_steps, (batch_size,), device=device).long()
    noisy_action = model.noise_scheduler.add_noise(actions, action_noise, action_t)

    # Video noise
    next_obs_noise = torch.randn_like(next_obs)
    next_obs_t = torch.randint(0, model.num_train_steps, (batch_size,), device=device).long()
    noisy_next_obs = model.noise_scheduler.add_noise(next_obs, next_obs_noise, next_obs_t)

    action_noise_pred, next_obs_noise_pred = model.noise_pred_net(
        obs_global, noisy_action, action_t, noisy_next_obs, next_obs_t, obs_memory
    )
    action_loss = F.mse_loss(action_noise_pred, action_noise)
    dynamics_loss = F.mse_loss(next_obs_noise_pred, next_obs_noise)

    return action_loss, dynamics_loss


def aggregate_by_layer(param_stats, model):
    """Aggregate per-parameter stats by layer."""
    layer_stats = defaultdict(lambda: {"cos_sim": [], "action_norm": [], "dynamics_norm": []})

    for name, values in param_stats[0].items():
        # Extract layer key: "noise_pred_net.blocks.N.xxx"
        parts = name.split(".")
        if "blocks" in parts:
            idx = parts.index("blocks")
            if idx + 1 < len(parts):
                layer_key = f"blocks.{parts[idx+1]}"
            else:
                layer_key = "other"
        else:
            layer_key = "other"

        layer_stats[layer_key]["cos_sim"].extend([float(np.mean(values))])
        if name in param_stats[1]:
            layer_stats[layer_key]["action_norm"].extend([float(np.mean(param_stats[1][name]))])
        if name in param_stats[2]:
            layer_stats[layer_key]["dynamics_norm"].extend([float(np.mean(param_stats[2][name]))])

    return {k: {sk: float(np.mean(sv)) if sv else 0.0 for sk, sv in v.items()}
            for k, v in layer_stats.items()}


# Globals for dataset access in collect fn
zarr_path_global = "/root/autodl-tmp/UWM_pushT/diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr"
bs_global = 16


def main():
    global zarr_path_global, bs_global

    parser = argparse.ArgumentParser(description="Gradient conflict diagnostics")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--zarr-path", type=str,
                        default=zarr_path_global)
    args = parser.parse_args()

    device = torch.device(args.device)
    bs_global = args.batch_size
    zarr_path_global = args.zarr_path

    print("=" * 60)
    print("Gradient Conflict Diagnostics")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  num_batches: {args.num_batches}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  device: {device}")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    model = build_model_from_ckpt(device, args.checkpoint)
    model.eval()
    print(f"  conditioning_type: {model.conditioning_type}")
    print(f"  dp_only: {model.dp_only}")

    if model.dp_only or model.conditioning_type != "cross_attn":
        print("ERROR: This script only works with cross_attn joint (non-dp_only) checkpoints.")
        sys.exit(1)

    # Attention diagnostics (single batch)
    print("\n[2/3] Attention mass diagnostics...")
    cleanup = patch_model_for_attention_capture(model)
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
    batch = next(iter(loader))
    curr_obs, next_obs, actions = process_batch(batch, 2, 16, device)

    with torch.no_grad():
        model(curr_obs, next_obs, actions)

    attn_mass = collect_attention_mass(model)
    cleanup()

    for entry in attn_mass:
        print(f"  Layer {entry['layer']:2d}: act→vid={entry['action_to_video']:.4f}  "
              f"reg→vid={entry['register_to_video']:.4f}  "
              f"vid→vid={entry['video_to_video']:.4f}  "
              f"act→act={entry['action_to_action']:.4f}")

    # Gradient diagnostics
    print(f"\n[3/3] Gradient diagnostics ({args.num_batches} batches)...")
    cos_sims, act_norms, dyn_norms = collect_gradient_stats(
        model, (curr_obs, next_obs, actions), args.num_batches, device
    )

    # Summary stats
    all_cos = [np.mean(v) for v in cos_sims.values()]
    print(f"\n  Gradient cosine similarity (mean across all params): {np.mean(all_cos):.4f}")
    print(f"  Min: {np.min(all_cos):.4f}  Max: {np.max(all_cos):.4f}  Median: {np.median(all_cos):.4f}")

    layer_stats = aggregate_by_layer((cos_sims, act_norms, dyn_norms), model)
    print(f"\n  Per-layer summary:")
    for layer_key in sorted(layer_stats.keys(), key=lambda x: (x.split(".")[0] != "blocks", x)):
        ls = layer_stats[layer_key]
        print(f"    {layer_key}: cos={ls['cos_sim']:.4f}  "
              f"||grad_act||={ls['action_norm']:.4f}  "
              f"||grad_dyn||={ls['dynamics_norm']:.4f}")

    # Save
    output = {
        "checkpoint": args.checkpoint,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "attention_mass": attn_mass,
        "gradient_cosine_similarity": {k: float(np.mean(v)) for k, v in cos_sims.items()},
        "gradient_action_norm": {k: float(np.mean(v)) for k, v in act_norms.items()},
        "gradient_dynamics_norm": {k: float(np.mean(v)) for k, v in dyn_norms.items()},
        "summary": {
            "mean_cos_sim": float(np.mean(all_cos)),
            "min_cos_sim": float(np.min(all_cos)),
            "max_cos_sim": float(np.max(all_cos)),
            "median_cos_sim": float(np.median(all_cos)),
            "per_layer": layer_stats,
        },
    }

    out_path = args.output or f"outputs/diag_gradient_conflict_{Path(args.checkpoint).stem}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
