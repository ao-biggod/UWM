# Code Map

## Diffusion Policy — PushT Pipeline

```
train.py                                        # Hydra entrypoint, config_path=diffusion_policy/config
  |
  +-> hydra.utils.get_class(cfg._target_)       # e.g. TrainDiffusionUnetHybridWorkspace
  |
  +-> workspace.run()
        |
        +-> dataset = hydra.utils.instantiate(cfg.task.dataset)
        |     |
        |     +-> PushTImageDataset (diffusion_policy/dataset/pusht_image_dataset.py)
        |           |
        |           +-> ReplayBuffer.copy_from_path(zarr_path)  (diffusion_policy/common/replay_buffer.py)
        |           +-> SequenceSampler(replay_buffer, horizon, ...)  (diffusion_policy/common/sampler.py)
        |           +-> __getitem__ -> {'obs': {'image': (T,3,96,96), 'agent_pos': (T,2)}, 'action': (T,2)}
        |
        +-> dataloader = DataLoader(dataset, ...)
        |
        +-> model = hydra.utils.instantiate(cfg.policy)
        |     |
        |     +-> DiffusionUnetHybridImagePolicy (diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py)
        |           |
        |           +-> DiffusionUNet (diffusion_policy/model/diffusion/...)
        |           +-> noise_scheduler (DDPMScheduler from diffusers)
        |
        +-> env_runner = hydra.utils.instantiate(cfg.task.env_runner)
        |     |
        |     +-> PushTImageRunner (diffusion_policy/env_runner/pusht_image_runner.py)
        |           |
        |           +-> MultiStepWrapper(VideoRecordingWrapper(PushTImageEnv))
        |           |     +-> PushTImageEnv (diffusion_policy/env/pusht/pusht_image_env.py)
        |           |           +-> PushTEnv (diffusion_policy/env/pusht/pusht_env.py) [PyMunk physics]
        |           +-> run(policy) -> log_data {test_mean_score, sim_video, ...}
        |
        +-> training loop:
              +-> batch = next(dataloader)
              +-> loss = model.compute_loss(batch)
              +-> optimizer.step()
              +-> if rollout_every: log_data = env_runner.run(policy)
              +-> checkpoint
```

**关键文件路径对应：**

| 节点 | 文件 |
|------|------|
| Entrypoint | `diffusion_policy-main/train.py` |
| Config (image) | `diffusion_policy-main/image_pusht_diffusion_policy_cnn.yaml` |
| Config (task, image) | `diffusion_policy-main/diffusion_policy/config/task/pusht_image.yaml` |
| Config (task, lowdim) | `diffusion_policy-main/diffusion_policy/config/task/pusht_lowdim.yaml` |
| Config (workspace) | `diffusion_policy-main/diffusion_policy/config/train_diffusion_transformer_lowdim_pusht_workspace.yaml` |
| Dataset (image) | `diffusion_policy-main/diffusion_policy/dataset/pusht_image_dataset.py` |
| Dataset (lowdim) | `diffusion_policy-main/diffusion_policy/dataset/pusht_dataset.py` |
| Dataset base | `diffusion_policy-main/diffusion_policy/dataset/base_dataset.py` |
| ReplayBuffer | `diffusion_policy-main/diffusion_policy/common/replay_buffer.py` (zarr backend) |
| SequenceSampler | `diffusion_policy-main/diffusion_policy/common/sampler.py` |
| Env (base) | `diffusion_policy-main/diffusion_policy/env/pusht/pusht_env.py` |
| Env (image) | `diffusion_policy-main/diffusion_policy/env/pusht/pusht_image_env.py` |
| Env (keypoints) | `diffusion_policy-main/diffusion_policy/env/pusht/pusht_keypoints_env.py` |
| EnvRunner (image) | `diffusion_policy-main/diffusion_policy/env_runner/pusht_image_runner.py` |
| EnvRunner (keypoints) | `diffusion_policy-main/diffusion_policy/env_runner/pusht_keypoints_runner.py` |
| Workspace (image) | `diffusion_policy-main/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py` |
| Workspace (lowdim) | `diffusion_policy-main/diffusion_policy/workspace/train_diffusion_transformer_lowdim_workspace.py` |
| Workspace base | `diffusion_policy-main/diffusion_policy/workspace/base_workspace.py` |
| Demo/数据采集 | `diffusion_policy-main/demo_pusht.py` |

## Unified World Model — Training Pipeline

```
experiments/uwm/train.py                        # Hydra entrypoint, config_path=../../configs
  |                                               # config_name=train_uwm.yaml
  +-> mp.spawn(train, nprocs=world_size)
        |
        +-> init_distributed(rank, world_size)    # experiments/utils.py
        +-> init_wandb(config)                     # experiments/utils.py
        |
        +-> train_set, val_set = instantiate(config.dataset)
        |     |
        |     +-> make_robomimic_dataset()  (datasets/robomimic/__init__.py)
        |     |     |
        |     |     +-> RobomimicDataset (datasets/robomimic/dataset.py)
        |     |           |
        |     |           +-> glob_all(hdf5_path_globs)  (datasets/utils/file_utils.py)
        |     |           +-> CompressedTrajectoryBuffer (datasets/utils/buffer.py) [zarr backend]
        |     |           +-> TrajectorySampler (datasets/utils/sampler.py)
        |     |           +-> unflatten_obs()  (datasets/utils/obs_utils.py)
        |     |           +-> __getitem__ -> {'obs.image': ..., 'action': ...}
        |     |                             -> unflatten -> {'obs': {'image': ...}, 'action': ...}
        |     |
        |     +-> 或 make_droid_dataset()  (datasets/droid/__init__.py)
        |
        +-> train_loader, val_loader = make_distributed_data_loader()
        |     |                              (datasets/utils/loader.py)
        |     +-> DistributedSampler + DataLoader
        |
        +-> model = instantiate(config.model)
        |     |
        |     +-> UnifiedWorldModel (models/uwm/uwm.py)
        |           |
        |           +-> UWMObservationEncoder (models/uwm/obs_encoder.py)
        |           |     +-> VideoTransform (models/common/transforms.py)
        |           |     +-> ViTImageEncoder or ResNetImageEncoder (models/common/vision.py)
        |           |     +-> VAEDownsample (models/common/transforms.py)
        |           |     +-> CLIPTextEncoder (models/common/language.py)
        |           |
        |           +-> DualNoisePredictionNet (models/uwm/uwm.py)
        |           |     +-> MultiViewVideoPatchifier
        |           |     +-> AdaLNAttentionBlock x depth
        |           |     +-> DualTimestepEncoder
        |           |
        |           +-> DDIMScheduler (from diffusers)
        |
        +-> optimizer = AdamW
        +-> scheduler = get_scheduler()  (from diffusers.optimization)
        +-> scaler = GradScaler (AMP)
        |
        +-> model = DistributedDataParallel(model)
        |
        +-> training loop:
              +-> batch = next(train_loader)
              +-> train_one_step()
              |     |
              |     +-> process_batch(batch, obs_horizon, action_horizon, device)
              |     |     |
              |     |     +-> curr_obs = {k: v[:, :action_start+1]}  # first To frames
              |     |     +-> next_obs = {k: v[:, action_end:]}      # last Tf frames
              |     |     +-> action = batch["action"][:, action_start:action_end]  # Ta frames
              |     |
              |     +-> loss, info = model(curr_obs, next_obs, action)
              |           |
              |           +-> encode_curr_and_next_obs(curr_obs, next_obs)
              |           |     +-> apply_transform([curr_obs, next_obs])  # augment images
              |           |     +-> img_encoder(curr_imgs)                 # encode to features
              |           |     +-> concat lowdim features
              |           |     +-> concat language features
              |           |     +-> apply_vae(next_imgs)                   # encode to latents
              |           |     +-> return curr_feats, next_latents
              |           |
              |           +-> noise_pred_net(curr_feats, noisy_action, t1, noisy_next_obs, t2)
              |           +-> action_loss = MSE(noise_pred, noise)
              |           +-> dynamics_loss = MSE(obs_noise_pred, obs_noise)
              |           +-> loss = action_loss + dynamics_loss
              |
              +-> scaler.scale(loss).backward()
              +-> scaler.step(optimizer)
              +-> scaler.update()
              +-> scheduler.step()
              |
              +-> maybe_evaluate() -> eval_one_epoch() -> val loss + samples
              +-> maybe_save_checkpoint()

experiments/uwm/train_robomimic.py               # Robomimic finetune entrypoint
  |                                               # config_name=train_uwm_robomimic.yaml
  +-> Same structure as train.py, plus:
        +-> maybe_collect_rollout()
              +-> collect_rollout()
                    +-> make_robomimic_env()  (environments/robomimic/__init__.py)
                    |     +-> RoboMimicEnvWrapper (environments/robomimic/wrappers.py)
                    +-> model.sample(obs)  # sample action
                    +-> env.step(action)   # execute action
                    +-> env.get_video()    # record video
                    +-> return success_rate, video
```

**关键文件路径对应：**

| 节点 | 文件 |
|------|------|
| Entrypoint (pretrain) | `experiments/uwm/train.py` |
| Entrypoint (finetune) | `experiments/uwm/train_robomimic.py` |
| Entrypoint (eval) | `experiments/uwm/eval_robomimic.py` |
| Config (train, pretrain) | `configs/train_uwm.yaml` → `configs/template_train.yaml` |
| Config (train, finetune) | `configs/train_uwm_robomimic.yaml` → `configs/template_train_robomimic.yaml` |
| Config (model) | `configs/model/uwm.yaml` |
| Config (dataset template) | `configs/dataset/template_robomimic.yaml` |
| Dataset factory | `datasets/robomimic/__init__.py` |
| Dataset class | `datasets/robomimic/dataset.py` |
| Buffer (zarr) | `datasets/utils/buffer.py` |
| TrajectorySampler | `datasets/utils/sampler.py` |
| DataLoader | `datasets/utils/loader.py` |
| unflatten_obs | `datasets/utils/obs_utils.py` |
| UWM Model | `models/uwm/uwm.py` |
| UWM ObsEncoder | `models/uwm/obs_encoder.py` |
| Vision backbone | `models/common/vision.py` |
| Transforms | `models/common/transforms.py` |
| Language encoder | `models/common/language.py` |
| Env factory | `environments/robomimic/__init__.py` |
| Env wrapper | `environments/robomimic/wrappers.py` |
| Utils (distributed) | `experiments/utils.py` |
| Normalizer | `datasets/utils/normalizer.py` |

## process_batch — 时序切分图示

```
dataset 返回的序列（长度 = num_frames = obs_horizon + action_len + obs_horizon - 1 = 11）:

索引:  0     1     2     3     4     5     6     7     8     9    10
obs:  [o0]  [o1]  [o2]  [o3]  [o4]  [o5]  [o6]  [o7]  [o8]  [o9]  [o10]
act:  [a0]  [a1]  [a2]  [a3]  [a4]  [a5]  [a6]  [a7]  [a8]  [a9]  [a10]

process_batch 切分 (obs_horizon=2, action_len=8):
  action_start = 1
  action_end   = 1 + 8 = 9

  curr_obs:  [o0, o1]             # [:2]   — 条件观测
  action:    [a1...a8]            # [1:9]  — 预测的动作序列
  next_obs:  [o9, o10]            # [9:]   — 未来观测（用于 dynamics loss）

语义: 看到 o0,o1, 预测 a1..a8, 结果观测为 o9,o10
```

## UWM Model Forward — Tensor Shape 追踪

```
forward(self, obs_dict, next_obs_dict, action, action_mask=None):
  B = batch_size

  # 1. Encode
  obs_dict:  {'image': (B, 2, 96, 96, 3), 'agent_pos': (B, 2, 2)}
  next_obs_dict: {'image': (B, 2, 96, 96, 3), 'agent_pos': (B, 2, 2)}
  action:    (B, 8, 2)  # action_len=8, action_dim=2

  curr_feats, next_latents = encode_curr_and_next_obs(obs_dict, next_obs_dict)
    # apply_transform: 统一 transform, images -> (B, 1, 3, 2, 96, 96)  [V=1, C=3, T=2, H, W]
    # img_encoder:      -> (B, V*T*D) = (B, 1*2*768) = (B, 1536)
    # lowdim concat:    agent_pos (B, 2, 2) -> flatten (B, 4)
    # curr_feats:       (B, 1536 + 4) = (B, 1540)
    # apply_vae:        next_imgs -> latent (B, 1, 4, 1, 12, 12)  [VAE downsampling]

  # 2. Noise prediction
  action_noise_pred, next_obs_noise_pred = noise_pred_net(
      curr_feats, noisy_action, action_t, noisy_next_obs, next_obs_t)
    # action_noise_pred:     (B, 8, 2)
    # next_obs_noise_pred:   (B, 1, 4, 1, 12, 12)

  # 3. Loss
  action_loss = MSE(action_noise_pred, action_noise)
  dynamics_loss = MSE(next_obs_noise_pred, next_obs_noise)
  loss = action_loss + dynamics_loss
```
