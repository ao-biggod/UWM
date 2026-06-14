#!/usr/bin/env python3
"""Ablation training: progressively align DP-only recipe toward official DP.

R1: +EMA
R2: +EMA + cosine LR + warmup
R3: +EMA + cosine LR + warmup + min/max normalizer + clip_sample=True

Config C eval (norm_agent_pos=True, clip_sample=False) is used for ALL evaluations.
Only training recipe changes across runs.

Usage:
  # R1: baseline + EMA
  python scripts/ablate_dp_recipe.py --run r1 --num-steps 20000 --device cuda:0

  # R2: + cosine LR + warmup
  python scripts/ablate_dp_recipe.py --run r2 --num-steps 20000 --device cuda:0

  # R3: + min/max normalizer + clip_sample=True
  python scripts/ablate_dp_recipe.py --run r3 --num-steps 20000 --device cuda:0

  # Full eval after training:
  python scripts/ablate_dp_recipe.py --run r1 --eval-only --eval-ckpt outputs/ablate_r1/latest.pt
"""
import argparse, json, os, sys, time, math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def collate_fn(batch):
    if isinstance(batch[0], dict):
        return {k: collate_fn([b[k] for b in batch]) for k in batch[0]}
    return torch.stack(batch)


def build_dataset(zarr_path, normalize_action_minmax=False):
    from datasets.pusht import make_pusht_dataset
    shape_meta = {
        "obs": {
            "image": {"shape": [96, 96, 3], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }
    return make_pusht_dataset(
        name="pusht_train", zarr_path=zarr_path, shape_meta=shape_meta,
        seq_len=19, val_ratio=0.02, max_train_episodes=90, seed=42,
        normalize_action=not normalize_action_minmax,  # False for min/max → raw actions
        normalize_lowdim=True,
    )


class EMAModel(nn.Module):
    """Exponential Moving Average for model parameters."""

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self._register(model)

    def _register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_to(self, model):
        """Copy EMA weights into model temporarily."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {"decay": self.decay, "shadow": {k: v.cpu().clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        for k, v in state["shadow"].items():
            self.shadow[k] = v


def build_model(device, clip_sample=True):
    from models.dp import ImageDiffusionPolicy, ImageObservationEncoder
    from models.dp import TransformerNoisePredictionNet

    shape_meta = {
        "obs": {"image": {"shape": [96, 96, 3], "type": "rgb"},
                "agent_pos": {"shape": [2], "type": "low_dim"}},
        "action": {"shape": [2]},
    }
    obs_encoder = ImageObservationEncoder(
        shape_meta=shape_meta, num_frames=2, embed_dim=768,
        resize_shape=None, crop_shape=None, random_crop=False,
        color_jitter=None, imagenet_norm=False,
        pretrained_weights=None, use_low_dim=True, use_language=False,
    )
    model = ImageDiffusionPolicy(
        action_len=16, action_dim=2, obs_encoder=obs_encoder,
        noise_pred_net=partial(
            TransformerNoisePredictionNet, input_len=16, input_dim=2,
            timestep_embed_dim=256, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True),
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2", clip_sample=clip_sample,
    )
    return model.to(device)


def save_checkpoint(model, ema_model, train_set, step, save_path, run_id, lowdim_normalizer=None):
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    an = train_set.action_normalizer
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "model": state,
        "step": step,
        "run": run_id,
        "action_normalizer": {"scale": np.asarray(an.scale).tolist(),
                               "offset": np.asarray(an.offset).tolist()},
    }
    if ema_model is not None:
        ckpt["ema_model"] = ema_model.state_dict()
    if lowdim_normalizer is not None:
        ckpt["lowdim_normalizer"] = lowdim_normalizer
    torch.save(ckpt, save_path)
    return ckpt


def run_val(model, val_set, device, obs_horizon, action_horizon):
    loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.eval()
    total_loss, n = 0.0, 0
    max_eval = min(20, len(loader))
    for i, batch in enumerate(loader):
        if i >= max_eval:
            break
        curr_obs = {k: v[:, :obs_horizon].to(device) for k, v in batch["obs"].items()}
        action_target = batch["action"][:, obs_horizon - 1 : obs_horizon - 1 + action_horizon].to(device)
        with torch.no_grad():
            loss = model(curr_obs, action_target)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / n


# ─── Deterministic Eval (same as before, config C) ───

def eval_deterministic(model, device, action_scale, action_offset,
                       ap_scale, ap_offset, seeds, label):
    import random
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    def seed_everything(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    model.eval()
    scores = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        seed_everything(seed)
        env = MultiStepWrapper(PushTImageEnv(legacy=True), n_obs_steps=2, n_action_steps=8)
        env.seed(seed)
        obs = env.reset()
        rewards = []; done = False; step = 0

        while not done and step < 300:
            img = obs["image"]; agent_pos = obs["agent_pos"]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if isinstance(agent_pos, np.ndarray):
                agent_pos = torch.from_numpy(agent_pos)
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
            rewards.append(float(reward))
            done = bool(np.all(done)); step += 1

        max_r = float(max(rewards)) if rewards else 0.0
        scores.append(max_r)
        if i < 5 or i % 10 == 0 or i == len(seeds) - 1:
            print(f"  [{label}] Ep {i:3d} (seed={seed}): {max_r:.4f}", flush=True)

    elapsed = time.time() - t0
    arr = np.array(scores)
    print(f"  [{label}] {len(seeds)}eps in {elapsed:.0f}s: mean={arr.mean():.4f} "
          f"median={np.median(arr):.4f} ep>0.5={(arr>0.5).sum()}/{len(seeds)}")
    return scores, arr


def get_lowdim_normalizer(ckpt, zarr_path):
    import zarr
    if "lowdim_normalizer" in ckpt:
        ln = ckpt["lowdim_normalizer"]
        if "agent_pos" in ln:
            return np.array(ln["agent_pos"]["scale"]), np.array(ln["agent_pos"]["offset"])
    z = zarr.open(zarr_path, "r")
    ep_ends = z["meta/episode_ends"][:]
    state = z["data/state"][:ep_ends[89], :2]
    scale = (state.max(axis=0) - state.min(axis=0)) / 2.0
    offset = (state.max(axis=0) + state.min(axis=0)) / 2.0
    return scale, offset


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="DP recipe ablation R1-R3")
    parser.add_argument("--run", type=str, required=True,
                        choices=["r1", "r2", "r3"],
                        help="r1=EMA, r2=EMA+cosineLR+warmup, r3=minmax+EMA+cosineLR")
    parser.add_argument("--zarr-path", type=str,
                        default="diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-ckpt", type=str, default=None)
    parser.add_argument("--eval-ema", action="store_true", help="Use EMA weights for eval")
    parser.add_argument("--n-eval-eps", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=100000)
    args = parser.parse_args()

    device = torch.device(args.device)
    obs_horizon, action_horizon = 2, 16
    eval_seeds = list(range(args.seed_start, args.seed_start + args.n_eval_eps))

    if args.output_dir is None:
        args.output_dir = f"outputs/ablate_{args.run}"
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Config per run ───

    if args.run == "r1":
        use_ema = True; ema_decay = 0.9999
        use_cosine_lr = False; warmup_steps = 0
        normalize_action_minmax = False; clip_sample_train = True
        cfg_desc = "R1: EMA decay=0.9999"
    elif args.run == "r2":
        use_ema = True; ema_decay = 0.9999
        use_cosine_lr = True; warmup_steps = 500
        normalize_action_minmax = False; clip_sample_train = True
        cfg_desc = "R2: EMA + cosine LR + warmup 500"
    else:  # r3
        use_ema = True; ema_decay = 0.9999
        use_cosine_lr = True; warmup_steps = 500
        normalize_action_minmax = True; clip_sample_train = True
        cfg_desc = "R3: EMA + cosine LR + warmup + min/max norm + clip_sample=True"

    print("=" * 60)
    print(f"DP Recipe Ablation: {cfg_desc}")
    print(f"  Steps: {args.num_steps}, Batch: {args.batch_size}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    # ─── Eval-only mode ───

    if args.eval_only:
        print("\n[Eval-only mode] Loading checkpoint...")
        ckpt = torch.load(args.eval_ckpt, map_location=device)
        model = build_model(device, clip_sample=False)  # Config C: clip_sample=False
        model.load_state_dict(ckpt["model"])
        model.eval()

        if args.eval_ema and "ema_model" in ckpt:
            ema = EMAModel(model, decay=ckpt["ema_model"]["decay"])
            ema.load_state_dict(ckpt["ema_model"])
            ema.apply_to(model)
            print("  Using EMA weights")

        an = ckpt["action_normalizer"]
        action_scale = np.array(an["scale"])
        action_offset = np.array(an["offset"])
        ap_scale, ap_offset = get_lowdim_normalizer(ckpt, args.zarr_path)

        print(f"  action_scale={action_scale}  action_offset={action_offset}")
        print(f"  ap_scale={ap_scale}  ap_offset={ap_offset}")

        label = "EMA" if args.eval_ema else "raw"
        scores, arr = eval_deterministic(model, device, action_scale, action_offset,
                                         ap_scale, ap_offset, eval_seeds, f"{args.run}_{label}")

        eval_dir = os.path.join(args.output_dir, f"eval_{label}")
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "per_episode_scores.csv"), "w") as f:
            f.write("ep,seed,score\n")
            for i, s in enumerate(eval_seeds):
                f.write(f"{i},{s},{scores[i]:.6f}\n")
        with open(os.path.join(eval_dir, "summary.json"), "w") as f:
            json.dump({"mean": float(arr.mean()), "median": float(np.median(arr)),
                       "std": float(arr.std()), "ep_gt_05": int((arr > 0.5).sum()),
                       "seeds": eval_seeds, "run": args.run, "label": label}, f, indent=2)
        return

    # ─── Training mode ───

    print("\n[1/5] Creating dataset...")
    train_set, val_set = build_dataset(args.zarr_path, normalize_action_minmax)

    if normalize_action_minmax:
        # Re-compute normalizer from data using min/max
        import zarr
        z = zarr.open(args.zarr_path, "r")
        ep_ends = z["meta/episode_ends"][:]
        actions = z["data/action"][:ep_ends[89]]
        # MinMax: map to [-1, 1]
        a_min = actions.min(axis=0)
        a_max = actions.max(axis=0)
        a_scale = (a_max - a_min) / 2.0
        a_offset = (a_max + a_min) / 2.0
        class MinMaxNormalizer:
            def __init__(self, scale, offset):
                self.scale = scale
                self.offset = offset
            def normalize(self, x):
                return (x - self.offset) / self.scale
            def unnormalize(self, x):
                return x * self.scale + self.offset
        train_set.action_normalizer = MinMaxNormalizer(a_scale, a_offset)
        train_set.normalize_action = True
        print(f"  MinMax action normalizer: scale={a_scale} offset={a_offset}")

    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}, Batches: {len(loader)}")

    print("\n[2/5] Creating model...")
    model = build_model(device, clip_sample=clip_sample_train)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params / 1e6:.1f}M")

    # EMA
    ema_model = EMAModel(model, decay=ema_decay) if use_ema else None

    # Optimizer with cosine LR + warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6,
                                   betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    if use_cosine_lr:
        base_lr = 1e-4
        # Warmup + cosine decay
        def get_lr(step):
            if step < warmup_steps:
                return base_lr * (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, args.num_steps - warmup_steps)
            return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(0)
        print(f"  LR schedule: cosine + warmup {warmup_steps}")
    else:
        print(f"  LR: 1e-4 constant")

    # Log
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    log_file = open(log_path, "a")

    print(f"\n[3/5] Training ({args.num_steps} steps)...")
    data_iter = iter(loader)
    t0 = time.time()

    for step in range(args.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        curr_obs = {k: v[:, :obs_horizon].to(device) for k, v in batch["obs"].items()}
        action_target = batch["action"][:, obs_horizon - 1 : obs_horizon - 1 + action_horizon].to(device)

        # Update LR
        if use_cosine_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = get_lr(step)

        model.train()
        loss = model(curr_obs, action_target)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema_model is not None:
            ema_model.update(model)

        log_entry = {"step": step, "loss": loss.item(),
                     "lr": optimizer.param_groups[0]["lr"]}
        log_file.write(json.dumps(log_entry) + "\n")

        if step > 0 and args.eval_every > 0 and step % args.eval_every == 0:
            val_loss = run_val(model, val_set, device, obs_horizon, action_horizon)
            print(f"\n[Eval] step={step}: val_loss={val_loss:.4f}")

        if step > 0 and args.save_every > 0 and step % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_step{step:07d}.pt")
            save_checkpoint(model, ema_model, train_set, step, save_path, args.run)
            latest_path = os.path.join(args.output_dir, "latest.pt")
            save_checkpoint(model, ema_model, train_set, step, latest_path, args.run)
            print(f"\n[Save] step={step}")

        if step % 100 == 0 or step < 10:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step {step:6d}: loss={loss.item():.4f} lr={lr:.2e} {sps:.1f}s/s")

    # Final save
    print(f"\n[4/5] Final save...")
    final_path = os.path.join(args.output_dir, "latest.pt")
    save_checkpoint(model, ema_model, train_set, args.num_steps - 1, final_path, args.run)
    log_file.close()

    # Summary
    elapsed = time.time() - t0
    print(f"\n[5/5] Training complete. {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {args.output_dir}")

    # ─── Post-training eval: raw + EMA ───

    print(f"\n[Post-train eval] 50-ep deterministic, seeds {eval_seeds[0]}-{eval_seeds[-1]}")

    an = train_set.action_normalizer
    action_scale = np.asarray(an.scale)
    action_offset = np.asarray(an.offset)

    # Get agent_pos normalizer from dataset
    if hasattr(train_set, 'lowdim_normalizer'):
        ap_scale = np.array(train_set.lowdim_normalizer["agent_pos"]["scale"])
        ap_offset = np.array(train_set.lowdim_normalizer["agent_pos"]["offset"])
    else:
        ap_scale, ap_offset = get_lowdim_normalizer({}, args.zarr_path)

    # Config C model (clip_sample=False for eval)
    model_eval = build_model(device, clip_sample=False)
    model_eval.load_state_dict({k: v for k, v in model.state_dict().items()})
    model_eval.eval()

    # Raw weights
    scores_raw, arr_raw = eval_deterministic(model_eval, device, action_scale, action_offset,
                                             ap_scale, ap_offset, eval_seeds, f"{args.run}_raw")

    # EMA weights
    if ema_model is not None:
        model_ema = build_model(device, clip_sample=False)
        model_ema.load_state_dict({k: v for k, v in model.state_dict().items()})
        ema_model.apply_to(model_ema)
        model_ema.eval()
        scores_ema, arr_ema = eval_deterministic(model_ema, device, action_scale, action_offset,
                                                  ap_scale, ap_offset, eval_seeds, f"{args.run}_EMA")
    else:
        arr_ema = None

    # Save results
    results = {
        "run": args.run, "config": cfg_desc,
        "seeds": eval_seeds,
        "raw": {"mean": float(arr_raw.mean()), "median": float(np.median(arr_raw)),
                "std": float(arr_raw.std()), "ep_gt_05": int((arr_raw > 0.5).sum()),
                "scores": [float(x) for x in scores_raw]},
    }
    if arr_ema is not None:
        results["ema"] = {"mean": float(arr_ema.mean()), "median": float(np.median(arr_ema)),
                          "std": float(arr_ema.std()), "ep_gt_05": int((arr_ema > 0.5).sum()),
                          "scores": [float(x) for x in scores_ema]}

    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*60}")
    print("RESULTS (same 50 seeds 100000-100049)")
    print(f"{'='*60}")
    print(f"\n{'Model':<25} {'Mean':>8} {'Median':>8} {'Std':>8} {'ep>0.5':>10}")
    print("-" * 58)
    print(f"  {'Official DP image':<23} {'0.9490':>8} {'1.0000':>8} {'0.1553':>8} {'49/50':>10}")
    print(f"  {'UWM joint C':<23} {'0.4342':>8} {'0.4057':>8} {'0.3850':>8} {'20/50':>10}")
    print(f"  {'DP-only C (baseline)':<23} {'0.3912':>8} {'0.3909':>8} {'0.3456':>8} {'17/50':>10}")
    print(f"  {'---':<23} {'---':>8} {'---':>8} {'---':>8} {'---':>10}")
    print(f"  {args.run + ' raw':<23} {arr_raw.mean():8.4f} {np.median(arr_raw):8.4f} "
          f"{arr_raw.std():8.4f} {(arr_raw > 0.5).sum():>8}/{len(eval_seeds)}")
    if arr_ema is not None:
        print(f"  {args.run + ' EMA':<23} {arr_ema.mean():8.4f} {np.median(arr_ema):8.4f} "
              f"{arr_ema.std():8.4f} {(arr_ema > 0.5).sum():>8}/{len(eval_seeds)}")

    print(f"\nSaved: {os.path.join(args.output_dir, 'eval_results.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
