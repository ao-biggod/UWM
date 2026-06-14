# Project Context

## 项目目标
复现 Diffusion Policy 的 PushT baseline，然后将 PushT task 接入 Unified World Model (UWM)，
最终实现 UWM on PushT 的训练与评估。

## 已有仓库

| 仓库 | 路径 | 大小 | 说明 |
|------|------|------|------|
| diffusion_policy-main | `D:\UWM_pushT\diffusion_policy-main\` | 18.4 MB, 427 files | Diffusion Policy 官方实现，含 PushT 完整 pipeline |
| unified-world-model-main | `D:\UWM_pushT\unified-world-model-main\` | 316 KB, 133 files | UWM 官方实现，含 DROID/LIBERO/Robomimic 训练 pipeline |

## 复现顺序

1. **Phase 0**: 代码地图 — 完成（见 CODE_MAP.md）
2. **Phase 1**: 复现 Diffusion Policy PushT baseline（在 AutoDL/Linux 上）
3. **Phase 2**: 实现 PushT dataset for UWM
4. **Phase 3**: 实现 PushT EnvRunner / Evaluator for UWM
5. **Phase 4**: 接入 UWM training config
6. **Phase 5**: 10-step smoke test
7. **Phase 6**: 正式训练
8. **Phase 7**: 评估和对比

详细计划见 REPRO_PLAN.md。

## 禁止事项

- 不要修改 UWM 核心训练代码（experiments/uwm/train.py）
- 不要修改 UWM 模型代码（models/uwm/uwm.py, obs_encoder.py）
- 不要修改 UWM 已有的 dataset 类（datasets/robomimic/dataset.py）
- 只能新增 UWM 的 PushT dataset 和 config，或在现有文件中做最小修改

## 约定的 Batch Key

基于 UWM `process_batch()` 函数（`experiments/uwm/train.py:23-33`）和现有 Robomimic dataset 约定：

```text
batch = {
    'obs': {
        'image':     Tensor[B, num_frames, C, H, W],  # uint8, HWC in shape_meta, 实际由 transform 处理
        'agent_pos': Tensor[B, num_frames, 2],         # float32, xy 坐标
    },
    'action': Tensor[B, num_frames, 2],                # float32
}
```

其中 `num_frames = obs_horizon + action_len + obs_horizon - 1`。

## 约定的 Tensor Shape

```text
                                  DP PushT          UWM PushT (目标)
obs/image (raw):                  [To, 3, 96, 96]   [B, num_frames, H, W, C]
obs/agent_pos (raw):              [To, 2]           [B, num_frames, 2]
action (raw):                     [Ta, 2]           [B, num_frames, 2]
obs/image (after process_batch):   —                [B, To, H, W, C]  (curr_obs), [B, Tf, H, W, C] (next_obs)
obs/agent_pos (after process_batch): —              [B, To, 2]  (curr_obs), [B, Tf, 2] (next_obs)
action (after process_batch):      —                [B, Ta, 2]

To = obs_num_frames = 2
Ta = action_len = 8 (与 DP n_action_steps 一致)
Tf = obs_num_frames = 2
num_frames = To + Ta + Tf - 1 = 2 + 8 + 2 - 1 = 11
              (待确认：实际可能是 2+8+1=11 或跟随 UWM 默认值 19)

image_size = 96x96
action_dim = 2
```

### 与 UWM 默认值的差异

| 参数 | UWM 默认 (Robomimic/DROID) | DP PushT | UWM PushT (建议) |
|------|--------------------------|----------|-----------------|
| obs_num_frames (To) | 2 | 2 | 2 |
| action_len (Ta) | 16 | 8 | 8 |
| image_size | 84x84 或 224x224 | 96x96 | 96x96 |
| action_dim | 7 或 10 | 2 | 2 |
| num_frames (seq_len) | 19 | 16 | 11 |
| obs type | rgb + low_dim | rgb + agent_pos | rgb + agent_pos |

## 平台说明

Windows 本地环境不适合训练（无 CUDA、无 robomimic 环境）。
所有数据准备、训练、评估步骤都应在 **AutoDL 或 Linux 服务器** 上执行。
配置路径中硬编码的 `/home/ubuntu/...` 路径需要改为 AutoDL 实际路径。

## UWM process_batch 工作机制 (关键)

```python
# experiments/uwm/train.py:23-33
def process_batch(batch, obs_horizon, action_horizon, device):
    action_start = obs_horizon - 1          # = 2-1 = 1
    action_end = action_start + action_horizon  # = 1+action_len
    curr_obs = {k: v[:, : action_start + 1] for k, v in batch["obs"].items()}  # [:, :2]
    next_obs = {k: v[:, action_end:] for k, v in batch["obs"].items()}         # [:, action_end:]
    actions = batch["action"][:, action_start:action_end]                       # [:, 1:1+action_len]
    return curr_obs, next_obs, actions
```

即从连续序列中切出：
- `curr_obs`: 前 To 帧
- `action`: 中间 Ta 帧(从第 1 帧后开始)
- `next_obs`: 最后 Tf 帧

这要求 dataset 返回的 `obs` 和 `action` 在时间轴上对齐，且重叠一帧（curr_obs 的最后一帧 = action 的第一帧对应时刻）。

## 已确认的关键文件

### Diffusion Policy 侧
- `train.py` — Hydra entrypoint，`config_path=diffusion_policy/config`
- `diffusion_policy/config/task/pusht_image.yaml` — PushT 任务配置
- `diffusion_policy/config/train_diffusion_transformer_lowdim_pusht_workspace.yaml` — lowdim 训练配置
- `image_pusht_diffusion_policy_cnn.yaml` — image 训练配置（单文件版）
- `diffusion_policy/dataset/pusht_image_dataset.py` — PushTImageDataset
- `diffusion_policy/dataset/pusht_dataset.py` — PushTLowdimDataset
- `diffusion_policy/env/pusht/pusht_env.py` — PushT gym env (PyMunk 物理)
- `diffusion_policy/env/pusht/pusht_image_env.py` — PushTImageEnv
- `diffusion_policy/env/pusht/pusht_keypoints_env.py` — PushTKeypointsEnv
- `diffusion_policy/env_runner/pusht_image_runner.py` — PushTImageRunner
- `diffusion_policy/env_runner/pusht_keypoints_runner.py` — PushTKeypointsRunner
- `diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py` — image workspace
- `diffusion_policy/workspace/train_diffusion_transformer_lowdim_workspace.py` — lowdim workspace
- `demo_pusht.py` — 数据采集脚本
- `diffusion_policy/common/replay_buffer.py` — ReplayBuffer (zarr 后端)

### UWM 侧
- `experiments/uwm/train.py` — UWM 预训练入口（Hydra: train_uwm.yaml）
- `experiments/uwm/train_robomimic.py` — UWM Robomimic 微调入口（Hydra: train_uwm_robomimic.yaml）
- `experiments/uwm/eval_robomimic.py` — Robomimic 评估脚本
- `experiments/uwm/eval_droid.py` — DROID 评估脚本
- `experiments/utils.py` — 分布式训练工具
- `models/uwm/uwm.py` — UWM 模型 (UnifiedWorldModel)
- `models/uwm/obs_encoder.py` — UWM 观测编码器 (UWMObservationEncoder)
- `datasets/robomimic/dataset.py` — RobomimicDataset（模板）
- `datasets/robomimic/__init__.py` — make_robomimic_dataset
- `datasets/utils/buffer.py` — CompressedTrajectoryBuffer (zarr)
- `datasets/utils/sampler.py` — TrajectorySampler
- `datasets/utils/loader.py` — make_distributed_data_loader
- `datasets/utils/obs_utils.py` — unflatten_obs
- `datasets/utils/normalizer.py` — LinearNormalizer
- `configs/template_train.yaml` — 预训练模板
- `configs/template_train_robomimic.yaml` — 微调训练模板
- `configs/model/uwm.yaml` — UWM 模型配置
- `configs/dataset/template_robomimic.yaml` — Robomimic dataset 模板
- `environments/robomimic/wrappers.py` — RoboMimicEnvWrapper
- `environments/robomimic/__init__.py` — make_robomimic_env

## PushT 数据来源

DP README 和代码中指向的数据文件：
- `pusht_cchi_v7_replay.zarr` — PushT 演示数据集
- DP demo_pusht.py 用于采集新数据
- 数据格式：zarr，包含 keys: `img` (T,H,W,C), `state` (T,5), `action` (T,2), `keypoint` (T,9,2), `n_contacts` (T,1)
- 图片尺寸：96x96 RGB
- 动作空间：2D (agent target position in [0, 512])
