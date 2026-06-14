# Reproduction Plan: PushT → UWM

## Phase 0: 代码地图 ✓

- **目标**: 理解两个仓库的结构、调用链、数据格式
- **输入**: 两个 repo 的所有源文件
- **输出**: CODE_MAP.md, PROJECT_CONTEXT.md
- **涉及文件**: 见 CODE_MAP.md
- **验收标准**: 能画出完整的调用链图
- **风险**: 无

## Phase 1: 复现 Diffusion Policy PushT Baseline

### 目标
在 AutoDL 或 Linux 服务器上跑通 Diffusion Policy 的 PushT 训练和评估，
得到 baseline 分数（mean_score 约 0.8-0.9）。

### 输入
- `diffusion_policy-main/` 全部代码
- PushT 数据集 `pusht_cchi_v7_replay.zarr`

### 输出
- 训练好的 DP checkpoint
- WandB 日志中的 `test_mean_score` 指标
- 确认 DP pipeline 正常工作

### 涉及文件
- `diffusion_policy-main/train.py`
- `diffusion_policy-main/image_pusht_diffusion_policy_cnn.yaml`
- `diffusion_policy-main/diffusion_policy/config/task/pusht_image.yaml`
- `diffusion_policy-main/diffusion_policy/dataset/pusht_image_dataset.py`
- `diffusion_policy-main/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py`
- `diffusion_policy-main/diffusion_policy/env_runner/pusht_image_runner.py`
- `diffusion_policy-main/diffusion_policy/env/pusht/pusht_image_env.py`

### 验收标准
1. 训练至少 100 epochs 不报错
2. `test_mean_score` 在 WandB 中可见
3. 评估视频显示 T 块被推到目标区域

### 可能风险
- conda 环境依赖冲突（conda_environment.yaml）
- zarr 数据路径需调整为实际路径
- Windows 本地无法运行，必须在 Linux 上
- PyMunk/pygame 依赖在 headless 服务器上需额外配置

## Phase 2: 实现 PushT Dataset for UWM

### 目标
创建 UWM 兼容的 PushT dataset，能通过 UWM 的 dataloader 正确加载数据。

### 输入
- DP 的 `pusht_image_dataset.py` 和 `ReplayBuffer`
- UWM 的 `RobomimicDataset` 代码（作为模板）
- UWM 的 `CompressedTrajectoryBuffer` 和 `TrajectorySampler`
- PushT zarr 数据 `pusht_cchi_v7_replay.zarr`

### 输出
- `datasets/pusht/__init__.py` — `make_pusht_dataset()` 工厂函数
- `datasets/pusht/dataset.py` — `PushtDataset` 类
- `configs/dataset/pusht_image.yaml` — PushT dataset 配置

### 涉及文件
- **新增**: `datasets/pusht/__init__.py`
- **新增**: `datasets/pusht/dataset.py`
- **新增**: `configs/dataset/pusht_image.yaml`
- **参考**: `datasets/robomimic/dataset.py`
- **参考**: `diffusion_policy-main/diffusion_policy/dataset/pusht_image_dataset.py`
- **参考**: `diffusion_policy-main/diffusion_policy/common/replay_buffer.py`

### 验收标准
1. `instantiate(config.dataset)` 返回 `(train_set, val_set)`
2. `train_set[0]` 返回格式为 `{'obs.image': ..., 'obs.agent_pos': ..., 'action': ...}`，经过 `unflatten_obs` 后匹配 `process_batch` 期望
3. `TrainerSampler` 正确跨 episode 采样
4. 数据形状：`obs.image: (num_frames, 96, 96, 3) uint8`, `obs.agent_pos: (num_frames, 2) float32`, `action: (num_frames, 2) float32`

### 可能风险
- DP 的 ReplayBuffer 存储格式与 UWM 的 CompressedTrajectoryBuffer 格式不同
  - DP: zarr 直接存储整个 episode 的 numpy arrays
  - UWM: zarr + CompressedTrajectoryBuffer 存储 flattened timeline
- 时序对齐：`obs` 和 `action` 在时间维度上的对齐方式需与 `process_batch` 一致
- `num_frames` 的选取（11 vs 19）

## Phase 3: 实现 PushT EnvRunner / Evaluator

### 目标
创建 PushT 环境的 rollout / evaluation wrapper，用于 UWM 训练中的在线评估。

### 输入
- DP 的 PushT gym env
- UWM 的 `RoboMimicEnvWrapper`
- UWM 的 `make_robomimic_env`

### 输出
- `environments/pusht/__init__.py` — `make_pusht_env()` 工厂函数
- `environments/pusht/wrappers.py` — `PushTEnvWrapper`

### 涉及文件
- **新增**: `environments/pusht/__init__.py`
- **新增**: `environments/pusht/wrappers.py`
- **参考**: `environments/robomimic/wrappers.py` (RoboMimicEnvWrapper)
- **参考**: `environments/robomimic/__init__.py` (make_robomimic_env)
- **复用**: `diffusion_policy-main/diffusion_policy/env/pusht/pusht_image_env.py`
- **复用**: `diffusion_policy-main/diffusion_policy/env/pusht/pusht_env.py`

### 验收标准
1. `make_pusht_env()` 能创建 PushT 环境
2. `env.reset()` 返回 `{'image': (obs_horizon, 96, 96, 3), 'agent_pos': (obs_horizon, 2)}`
3. `env.step(action)` 正确执行 action sequence
4. 能录制 rollout video
5. 返回 success metric（coverage rate）

### 可能风险
- `pusht_env.py` 依赖 `pymunk`、`pygame`、`shapely`、`skimage`（需在 Linux 上安装）
- 评估标准需与 DP 一致（覆盖率达到 95% 即为成功）
- `unified-world-model-main` 没有 `diffusion_policy` 的依赖，需要决定如何引用

## Phase 4: 接入 UWM Training Config

### 目标
创建 Hydra config，使 UWM 能加载 PushT dataset + PushT env。

### 输入
- Phase 2 的 `PushtDataset`
- Phase 3 的 `PushTEnvWrapper`
- UWM 的 config 模板

### 输出
- `configs/dataset/pusht_image.yaml` — 完整 dataset 配置
- `configs/train_uwm_pusht.yaml` — UWM on PushT 训练入口
- `configs/train_uwm_pusht_finetune.yaml` — 从 DROID 预训练模型微调的入口（可选）

### 涉及文件
- **新增**: `configs/train_uwm_pusht.yaml`
- **修改/新增**: `configs/dataset/pusht_image.yaml`
- **参考**: `configs/train_uwm_robomimic.yaml`
- **参考**: `configs/template_train_robomimic.yaml`

### 验收标准
1. `python experiments/uwm/train_robomimic.py --config-name train_uwm_pusht` 能启动（即使没有 GPU）
2. Hydra 不报 config 解析错误
3. dataset 和 model 都能 `instantiate` 成功

### 可能风险
- Hydra 的 `defaults` 覆盖机制需要仔细测试
- `shape_meta` 格式必须与 `obs_encoder` 期望一致
- `model.obs_encoder.use_low_dim` 需要设为 True（因为 PushT 有 agent_pos）
- `model.obs_encoder.use_language` 需要设为 False（PushT 无语言）

## Phase 5: 10-Step Smoke Test

### 目标
用极小的训练步数（10 steps）验证整个 pipeline 无语法错误、无 shape 不匹配。

### 输入
- Phase 2 的 dataset
- Phase 3 的 env wrappers（可选，此阶段可跳过 rollout）
- Phase 4 的 config
- PushT zarr 数据

### 输出
- 确认 `model.forward()` 不报 shape error
- 确认 `process_batch()` 切分正确
- 确认 loss 正常下降

### 涉及文件
- `experiments/uwm/train.py` 或 `experiments/uwm/train_robomimic.py`
- 新增的 pusht dataset 和 config

### 验收标准
1. 10 steps 内无 CUDA/CPU shape error
2. `action_loss` 和 `dynamics_loss` 有数值且非 NaN
3. 不需要 rollout — 仅验证训练 loop

### 可能风险
- shape_meta 中 image shape 格式 (H,W,C) vs (C,H,W) 混淆
- `process_batch` 中 seq_len 不够导致 index out of range
- UWM 默认的 `resize_shape: [240, 320]` + `crop_shape: [224, 224]` 对 96x96 图像不合适
- VAE downsampling 对 96x96 输入可能出错
- `num_frames` 不足导致 `action_end + To > seq_len`

## Phase 6: 正式训练

### 目标
在 AutoDL 上完整训练 UWM on PushT。

### 输入
- Phase 5 验证通过的代码
- PushT zarr 数据
- (可选) DROID 预训练 checkpoint

### 输出
- UWM PushT 训练 checkpoint
- WandB 训练曲线

### 涉及文件
- 所有 Phase 2-4 的文件
- `experiments/uwm/train.py` 或 `train_robomimic.py`

### 验收标准
1. 完整训练不报错
2. 训练 loss 收敛
3. WandB 日志完整

### 可能风险
- GPU 显存不足（UWM 模型较大，batch_size 需调整）
- 预训练 checkpoint 的 obs_encoder 与 PushT shape_meta 不匹配

## Phase 7: 评估和对比

### 目标
评估 UWM on PushT 的性能，与 DP baseline 对比。

### 输入
- Phase 6 的 checkpoint
- PushT eval environment

### 输出
- UWM on PushT 的 success rate / coverage metric
- 与 DP baseline 的对比表

### 涉及文件
- `experiments/uwm/eval_robomimic.py`（需修改或新建 pushT eval）
- OR 新建 `experiments/uwm/eval_pusht.py`
- Phase 3 的 PushT env wrapper

### 验收标准
1. 评估脚本正常运行
2. 有可量化的对比指标
3. 有 rollout 视频

### 可能风险
- UWM 的 `sample()` 方法输出 action shape 需与 PushT env 期望匹配
- 评估标准需与 DP 完全一致才能公平对比
