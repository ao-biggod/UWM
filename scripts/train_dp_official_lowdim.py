#!/usr/bin/env python3
"""Train DP official lowdim policy with 20D keypoint obs, for bridge comparison.

Uses official DP components: PushTLowdimDataset, DiffusionTransformerLowdimPolicy,
LinearNormalizer, EMAModel. Minimal training loop, not Hydra.
"""
import sys, os, time, copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque
from omegaconf import OmegaConf
import hydra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))

from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler


def main():
    device = torch.device("cuda:0")

    # ---- Load task config ----
    task_cfg = OmegaConf.load("diffusion_policy-main/diffusion_policy/config/task/pusht_lowdim.yaml")
    obs_dim = task_cfg.obs_dim  # 20
    action_dim = task_cfg.action_dim  # 2
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8

    print(f"DP Official Lowdim Training")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, horizon={horizon}")
    print(f"  n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")

    # ---- Dataset ----
    from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
    dataset = PushTLowdimDataset(
        zarr_path="diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr",
        horizon=horizon,
        pad_before=n_obs_steps - 1,
        pad_after=n_action_steps - 1,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=90,
    )
    print(f"Train samples: {len(dataset)}")
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    # Normalize from dataset stats
    normalizer = dataset.get_normalizer()
    print(f"Normalizer stats loaded")

    # ---- Model ----
    from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
    from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon")

    transformer = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        cond_dim=obs_dim,
        n_layer=8, n_head=4, n_emb=256,
        p_drop_emb=0.0, p_drop_attn=0.01,
        causal_attn=True, time_as_cond=True,
    )

    raw_model = DiffusionTransformerLowdimPolicy(
        model=transformer,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=100,
        obs_as_cond=True,
        pred_action_steps_only=False,
    ).to(device)
    raw_model.set_normalizer(normalizer)
    raw_model.normalizer.to(device)

    # Patch compute_loss to fix device issues
    orig_compute_loss = raw_model.compute_loss
    def patched_compute_loss(batch):
        # Ensure all batch values are tensors on device
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            device_batch[k] = v
        # Ensure normalizer is on device
        if hasattr(raw_model.normalizer, 'to'):
            pass  # already done
        if hasattr(raw_model.mask_generator, 'to'):
            pass  # already done
        return orig_compute_loss(device_batch)
    raw_model.compute_loss = patched_compute_loss
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M")

    # EMA
    ema_model = copy.deepcopy(raw_model)
    ema_model.eval()
    ema_model.requires_grad_(False)
    ema = EMAModel(
        model=ema_model, update_after_step=0, inv_gamma=1.0,
        power=0.75, min_value=0.0, max_value=0.9999)

    # Optimizer + LR scheduler
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-4, weight_decay=1e-6)
    num_epochs = 100
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=total_steps)

    print(f"Training: {num_epochs} epochs x {steps_per_epoch} steps = {total_steps} steps")
    print(f"{'='*60}")

    t0 = time.time()
    global_step = 0
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in train_loader:
            raw_model.train()
            # Convert numpy arrays to tensors and move to device
            batch = {
                k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v).to(device)
                if isinstance(v, (torch.Tensor, np.ndarray)) else v
                for k, v in batch.items()
            }
            loss = raw_model.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            ema.step(raw_model)
            epoch_losses.append(loss.item())
            global_step += 1

        if epoch % 10 == 0 or epoch < 5:
            elapsed = time.time() - t0
            lr = lr_scheduler.get_last_lr()[0]
            print(f"  epoch {epoch:3d}: loss={np.mean(epoch_losses):.6f}  lr={lr:.2e}  "
                  f"ema_decay={ema.decay:.4f}  {elapsed:.1f}s", flush=True)

        # Save checkpoint periodically
        if epoch % 20 == 0:
            ckpt_path = f"artifacts_keep/dp_official_lowdim/checkpoint_epoch_{epoch:03d}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "raw_model": raw_model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "normalizer": normalizer,
            }, ckpt_path)
            print(f"  [Saved: {ckpt_path}]", flush=True)

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Final save
    os.makedirs("artifacts_keep/dp_official_lowdim", exist_ok=True)
    final_path = "artifacts_keep/dp_official_lowdim/latest.pt"
    torch.save({
        "epoch": num_epochs - 1, "global_step": global_step,
        "raw_model": raw_model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "normalizer": normalizer,
    }, final_path)
    print(f"Final model saved: {final_path}")

    # ---- Eval ----
    print(f"\n{'='*60}")
    print(f"Eval: DP official lowdim + PushTKeypointsRunner")
    print(f"{'='*60}")

    from diffusion_policy.env_runner.pusht_keypoints_runner import PushTKeypointsRunner

    eval_policy = ema_model
    eval_policy.eval()

    env_runner = PushTKeypointsRunner(
        keypoint_visible_rate=1.0,
        n_train=0, n_train_vis=0,
        n_test=50, n_test_vis=0,
        legacy_test=True,
        train_start_seed=0,
        test_start_seed=100000,
        max_steps=300,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        n_latency_steps=0,
        fps=10,
        agent_keypoints=False,
        past_action=False,
        n_envs=None,
        output_dir="artifacts_keep/dp_official_lowdim/",
    )

    eval_result = env_runner.run(eval_policy)
    print(f"\n  Eval result: {eval_result}")

    # Save eval
    import json
    with open("artifacts_keep/dp_official_lowdim/eval_result.json", "w") as f:
        json.dump(eval_result if isinstance(eval_result, dict) else str(eval_result), f, indent=2)


if __name__ == "__main__":
    main()
