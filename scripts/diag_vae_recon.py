#!/usr/bin/env python3
"""Minimal SDXL-VAE reconstruction test on PushT images."""
import os, sys
import numpy as np
import torch
import zarr
from PIL import Image

sys.path.insert(0, "unified-world-model-main")

# 1. Load 20 frames from PushT zarr
z = zarr.open("diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr", "r")
all_imgs = z["data/img"][:]  # (N, 96, 96, 3) uint8
print(f"Dataset images: {all_imgs.shape}, dtype={all_imgs.dtype}, range=[{all_imgs.min()}, {all_imgs.max()}]")

# Take 20 frames spread across different episodes
n_per_ep = 5
n_ep = 4
ep_ends = z["meta/episode_ends"][:]
ep_starts = np.concatenate([[0], ep_ends[:-1]])
indices = []
for i in range(min(n_ep, len(ep_starts))):
    s, e = ep_starts[i], ep_ends[i]
    if e - s >= n_per_ep:
        idxs = np.linspace(s, e - 1, n_per_ep, dtype=int)
    else:
        idxs = np.arange(s, e)
    indices.extend(idxs[:n_per_ep])
indices = indices[:20]
print(f"Sampled {len(indices)} frames from episodes")

# 2. Load SDXL-VAE
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", local_files_only=True)
vae.eval()
device = torch.device("cuda:0")
vae = vae.to(device)
print(f"VAE loaded, device={device}")

# 3. Encode-decode each image
out_dir = "outputs/diagnostics/vae_recon_pusht"
os.makedirs(out_dir, exist_ok=True)

for i, idx in enumerate(indices):
    img_uint8 = all_imgs[idx]  # (96, 96, 3) uint8
    img_float = img_uint8.astype(np.float32) / 255.0  # [0, 1]

    # VAE expects (B, C, H, W) in [-1, 1]
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 96, 96)
    img_tensor = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()
        recon = vae.decode(latents).sample  # (1, 3, H', W')
        recon = (recon + 1.0) / 2.0  # [-1,1] -> [0,1]
        recon = torch.clamp(recon, 0, 1)

    # Compute metrics
    recon_np = recon[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    mse = ((img_float - recon_np) ** 2).mean()
    psnr = -10 * np.log10(mse) if mse > 0 else 99

    # Handle possible size mismatch (VAE may output slightly different size)
    h_in, w_in = img_float.shape[:2]
    h_out, w_out = recon_np.shape[:2]
    if h_in != h_out or w_in != w_out:
        from PIL import Image as PILImage
        recon_pil = PILImage.fromarray((recon_np * 255).astype(np.uint8))
        recon_pil = recon_pil.resize((w_in, h_in), PILImage.LANCZOS)
        recon_np = np.array(recon_pil).astype(np.float32) / 255.0
        mse = ((img_float - recon_np) ** 2).mean()
        psnr = -10 * np.log10(mse) if mse > 0 else 99

    # Side-by-side image
    side = np.concatenate([img_float, recon_np], axis=1)  # (H, 2*W, 3)
    side_uint8 = (side * 255).astype(np.uint8)
    Image.fromarray(side_uint8).save(f"{out_dir}/frame_{i:02d}_ep{idx:05d}.png")

    latent_shape = latents.shape
    print(f"  [{i:2d}] idx={idx:5d}  PSNR={psnr:.2f}dB  latent={list(latent_shape)}  "
          f"recon range=[{recon_np.min():.3f}, {recon_np.max():.3f}]  {'RESIZED' if h_in != h_out else ''}")

print(f"\nSaved {len(indices)} comparison images to {out_dir}/")
print("Done.")
