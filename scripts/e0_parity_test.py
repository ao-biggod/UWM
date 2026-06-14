#!/usr/bin/env python3
"""E0-P: Policy adapter parity test — first call action alignment."""
import sys, os, torch, numpy as np, dill
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))

os.makedirs("outputs/e0_parity", exist_ok=True)

# ========= Load policy =========
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

CKPT = "outputs/e0_lowdim_official_full/checkpoints/epoch=0190-test_mean_score=1.000.ckpt"
SEED = 100000
DEVICE = "cuda:0"

ckpt = torch.load(CKPT, map_location="cpu")
ema_sd = ckpt["state_dicts"]["ema_model"]

noise_scheduler = DDPMScheduler(
    num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
    beta_start=0.0001, beta_end=0.02, variance_type="fixed_small",
    clip_sample=True, prediction_type="epsilon")
transformer = TransformerForDiffusion(
    input_dim=2, output_dim=2, horizon=16, n_obs_steps=2, cond_dim=20,
    n_layer=8, n_head=4, n_emb=256, p_drop_emb=0.0, p_drop_attn=0.01,
    causal_attn=True, time_as_cond=True, obs_as_cond=True)
policy = DiffusionTransformerLowdimPolicy(
    model=transformer, noise_scheduler=noise_scheduler,
    horizon=16, obs_dim=20, action_dim=2,
    n_action_steps=8, n_obs_steps=2,
    num_inference_steps=100, obs_as_cond=True,
    pred_action_steps_only=False)
policy.load_state_dict(ema_sd)
policy.eval()
policy.to(DEVICE)

# ========= E0-P2: Official runner first call =========
print("=" * 60)
print("E0-P2: Official PushTKeypointsRunner — first policy call dump")
print("=" * 60)

from diffusion_policy.env_runner.pusht_keypoints_runner import PushTKeypointsRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.common.pytorch_util import dict_apply

# Replicate what PushTKeypointsRunner does for a single env
env_n_obs_steps = 2  # n_obs_steps + n_latency_steps
env = MultiStepWrapper(
    PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0),
    n_obs_steps=env_n_obs_steps,
    n_action_steps=8)

env.seed(SEED)
obs = env.reset()  # shape (n_obs_steps, 40) = (2, 40)

print(f"obs.shape from reset: {obs.shape}")  # should be (2, 40)

Do = obs.shape[-1] // 2  # 20
np_obs_dict = {
    'obs': obs[...,:2,:Do].astype(np.float32),  # (1, 2, 20)
    'obs_mask': obs[...,:2,Do:] > 0.5,
}
print(f"np_obs_dict['obs'].shape: {np_obs_dict['obs'].shape}")  # (2, 20) no batch dim

# Add batch dim (as runner does via env wrapper)
obs_dict = dict_apply(np_obs_dict,
    lambda x: torch.from_numpy(x).unsqueeze(0).to(device=DEVICE))
print(f"obs_dict['obs'].shape (with batch): {obs_dict['obs'].shape}")  # (1, 2, 20)

torch.manual_seed(42)
with torch.no_grad():
    result = policy.predict_action(obs_dict)

action_official = result['action'][0].detach().cpu().numpy()  # (8, 2)
print(f"action_official.shape: {action_official.shape}")
print(f"action_official[0]: {action_official[0]}")
print(f"action_official range: [{action_official.min():.1f}, {action_official.max():.1f}]")

# Execute first action to get next obs (for completeness)
next_obs, _, _, _ = env.step(action_official)  # shape (8, 2) as expected by wrapper
print(f"next_obs.shape: {next_obs.shape}")

# Save
np.savez("outputs/e0_parity/official_first_call.npz",
    seed=SEED,
    obs_dict_keys=list(np_obs_dict.keys()),
    obs_shape=obs.shape,
    obs_raw=obs,
    obs_input_shape=np_obs_dict['obs'].shape,
    obs_input_raw=np_obs_dict['obs'],
    action_return_shape=action_official.shape,
    action_return_raw=action_official,
    executed_actions=action_official,
    first_env_obs_after_reset=obs,
)
print("Saved: outputs/e0_parity/official_first_call.npz")

# ========= E0-P3: Our fixed-buffer first call =========
print(f"\n{'=' * 60}")
print("E0-P3: Our fixed-buffer — first policy call dump")
print("=" * 60)

# Use the exact same env setup as official
env2 = MultiStepWrapper(
    PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0),
    n_obs_steps=env_n_obs_steps,
    n_action_steps=8)
env2.seed(SEED)
obs2 = env2.reset()

print(f"obs.shape: {obs2.shape}")

# Build obs dict EXACTLY as official runner does
Do2 = obs2.shape[-1] // 2
our_np_obs = {
    'obs': obs2[...,:2,:Do2].astype(np.float32),
}
our_obs_dict = dict_apply(our_np_obs,
    lambda x: torch.from_numpy(x).unsqueeze(0).to(device=DEVICE))

torch.manual_seed(42)
with torch.no_grad():
    our_result = policy.predict_action(our_obs_dict)

our_action = our_result['action'][0].detach().cpu().numpy()
print(f"our_action.shape: {our_action.shape}")
print(f"our_action[0]: {our_action[0]}")

np.savez("outputs/e0_parity/our_first_call.npz",
    seed=SEED,
    obs_dict_keys=list(our_np_obs.keys()),
    obs_shape=obs2.shape,
    obs_raw=obs2,
    obs_input_shape=our_np_obs['obs'].shape,
    obs_input_raw=our_np_obs['obs'],
    action_return_shape=our_action.shape,
    action_return_raw=our_action,
    executed_actions=our_action,
)
print("Saved: outputs/e0_parity/our_first_call.npz")

# ========= E0-P4: Numerical comparison =========
print(f"\n{'=' * 60}")
print("E0-P4: Numerical alignment check")
print("=" * 60)

official = np.load("outputs/e0_parity/official_first_call.npz")
our = np.load("outputs/e0_parity/our_first_call.npz")

obs_diff = np.abs(official["obs_raw"] - our["obs_raw"]).max()
obs_input_diff = np.abs(official["obs_input_raw"] - our["obs_input_raw"]).max()
action_diff = np.abs(official["action_return_raw"] - our["action_return_raw"]).max()
exec_diff = np.abs(official["executed_actions"] - our["executed_actions"]).max()

print(f"  obs_raw max_abs_diff:          {obs_diff:.10f}")
print(f"  obs_input max_abs_diff:        {obs_input_diff:.10f}")
print(f"  action_return max_abs_diff:    {action_diff:.10f}")
print(f"  executed_actions max_abs_diff: {exec_diff:.10f}")

if obs_diff < 1e-5 and action_diff < 1e-4:
    print(f"\n  >>> PARITY PASSED <<<")
else:
    print(f"\n  >>> PARITY FAILED <<<")
