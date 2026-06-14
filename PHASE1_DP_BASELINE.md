# Phase 1: Diffusion Policy PushT Baseline 复现

> 目标：在 AutoDL/Linux 上复现 Diffusion Policy 的 PushT image baseline。

---

## 1. 调用链全景

```
train.py
  └─ hydra.main(config_path=diffusion_policy/config)
      └─ config: image_pusht_diffusion_policy_cnn.yaml
          ├─ _target_: TrainDiffusionUnetHybridWorkspace
          ├─ task: pusht_image
          │   ├─ dataset: PushTImageDataset
          │   │   └─ ReplayBuffer → data/pusht/pusht_cchi_v7_replay.zarr
          │   │       ├─ data/img    (25650, 96, 96, 3) float32
          │   │       ├─ data/state  (25650, 5) float32
          │   │       ├─ data/action (25650, 2) float32
          │   │       └─ meta/episode_ends (206,) int64
          │   └─ env_runner: PushTImageRunner
          │       └─ MultiStepWrapper(VideoRecordingWrapper(PushTImageEnv))
          └─ policy: DiffusionUnetHybridImagePolicy
              ├─ obs_encoder: MultiImageObsEncoder (ResNet18)
              └─ diffusion: ConditionalUnet1D
```

## 2. 环境创建步骤

### 2.1 系统依赖

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

### 2.2 Conda 环境

官方文件：`conda_environment.yaml`（环境名 `robodiff`）
我们重命名为 `dp-pusht` 避免冲突。

关键版本（实测自文件内容）：

| 包 | 版本 |
|---|---|
| Python | 3.9 |
| PyTorch | 1.12.1 |
| CUDA Toolkit | 11.6 |
| torchvision | 0.13.1 |
| gym | 0.21.0 |
| pymunk | 6.2.1 |
| zarr | 2.12.0 |
| hydra-core | 1.2.0 |
| py-opencv | 4.6.0 |
| shapely | 1.8.4 |
| pygame | 2.1.2 (pip) |

### 2.3 安装

```bash
conda env create -f conda_environment.yaml
# 或：
mamba env create -f conda_environment.yaml
pip install -e .
```

## 3. 数据下载与放置

### 3.1 下载

```bash
cd diffusion_policy-main
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip
cd ..
```

### 3.2 预期目录结构

```
diffusion_policy-main/
  data/
    pusht/
      pusht_cchi_v7_replay.zarr/
        data/
          action/       (25650, 2) float32
          img/          (25650, 96, 96, 3) float32
          keypoint/     (25650, 9, 2) float32
          n_contacts/   (25650, 1) float32
          state/        (25650, 5) float32
        meta/
          episode_ends  (206,) int64
```

### 3.3 数据来源

官网：https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip

## 4. 命令汇总

### 4.1 Dataset/Env smoke

```bash
cd diffusion_policy-main
python scripts/smoke_pusht_dataset.py
python scripts/smoke_pusht_env.py
```

### 4.2 10-step 训练 smoke（debug 模式）

```bash
cd diffusion_policy-main
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.debug=True \
  training.device=cuda:0 \
  logging.mode=disabled \
  hydra.run.dir='data/outputs/smoke_test'
```

`training.debug=True` 自动设置：num_epochs=2, max_train_steps=3, max_val_steps=3, rollout_every=1, checkpoint_every=1。

### 4.3 仅 10 step 非 debug

```bash
cd diffusion_policy-main
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.num_epochs=1 \
  training.max_train_steps=10 \
  training.max_val_steps=5 \
  training.rollout_every=1 \
  training.checkpoint_every=1 \
  training.device=cuda:0 \
  logging.mode=disabled \
  hydra.run.dir='data/outputs/smoke_10step'
```

### 4.4 正式训练

```bash
cd diffusion_policy-main
conda activate dp-pusht
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  logging.mode=disabled \
  hydra.run.dir='data/outputs/dp_pusht_baseline'
```

### 4.5 Eval

```bash
python eval.py \
  --checkpoint data/outputs/dp_pusht_baseline/checkpoints/latest.ckpt \
  --output_dir data/pusht_eval_output \
  --device cuda:0
```

## 5. Image Shape 检查结果

### 5.1 Zarr 原始数据（已实际读取 .zarray metadata 确认）

| 数组 | Shape | Dtype | 说明 |
|---|---|---|---|
| `data/img` | (25650, 96, 96, 3) | float32 | HWC，值域 [0, 255] |
| `data/state` | (25650, 5) | float32 | agent_pos(2) + block_pose(3) |
| `data/action` | (25650, 2) | float32 | 2D 位置控制 |
| `data/keypoint` | (25650, 9, 2) | float32 | keypoint 标签（训练中未使用） |
| `data/n_contacts` | (25650, 1) | float32 | 接触点数量（训练中未使用） |
| `meta/episode_ends` | (206,) | int64 | 206 episodes, 共 25650 steps |

### 5.2 Dataset 处理 pipeline

| 阶段 | Shape | dtype | 值域 |
|---|---|---|---|
| Zarr 原始 | `(T, 96, 96, 3)` | float32 | [0, 255] |
| `_sample_to_data` 后 | `(T, 3, 96, 96)` | float32 | [0, 1] |
| `get_image_range_normalizer` 后 | `(T, 3, 96, 96)` | float32 | [-1, 1] |

### 5.3 处理步骤明细

| 步骤 | 操作 | 代码位置 |
|---|---|---|
| 1. 读取 zarr | `ReplayBuffer.copy_from_path(zarr_path, keys=['img','state','action'])` | `pusht_image_dataset.py:26` |
| 2. Transpose + normalize | `np.moveaxis(sample['img'], -1, 1) / 255` → CHW + [0,1] | `pusht_image_dataset.py:75` |
| 3. 提取 agent_pos | `sample['state'][:,:2]` (前2维 = agent position) | `pusht_image_dataset.py:74` |
| 4. Image normalizer | `scale=2, offset=-1` → [-1, 1] | `normalize_util.py:23-36` |
| 5. Action/Agent 归一化 | `LinearNormalizer.fit(mode='limits')` → 拟合范围归一化 | `pusht_image_dataset.py:60-68` |
| 6. Pipeline Crop | obs_encoder 中 CropRandomizer (84x84 → ResNet18 input) | `diffusion_unet_hybrid_image_policy.py:78-88` |

### 5.4 key 映射总结

| Interface key | Zarr source | Shape |
|---|---|---|
| `obs['image']` | `data/img` | `(To, 3, 96, 96)` float32 [0,1] |
| `obs['agent_pos']` | `data/state[...,:2]` | `(To, 2)` float32 |
| `action` | `data/action` | `(Ta, 2)` float32 |

## 6. Metric

| Metric | 说明 |
|---|---|
| `test/mean_score` | 50 test episode 的 max_reward 均值（主指标） |
| `test/sim_max_reward_<seed>` | 单个 episode max_reward |
| `train_loss` | 训练 loss |
| `val_loss` | 验证 loss |
| `train_action_mse_error` | 训练 action MSE |

论文预期 ~0.9+ mean_score。

## 7. 当前状态

| 状态 | 项目 | 详情 |
|---|---|---|
| ✅ | DP 代码 | `diffusion_policy-main/` 已就位 |
| ✅ | PushT 数据 | 已下载，zarr 结构完整 (206 episodes, 25650 steps) |
| ✅ | 脚本已创建 | `scripts/autodl_dp_0{0,1,2}_*.sh` 就位 |
| ✅ | 文档已完成 | `PHASE1_DP_BASELINE.md` 本文件 |
| ❌ | conda env | 尚未创建 `dp-pusht` |
| ❌ | `pip install -e .` | 未执行 |
| ❌ | smoke test | 未执行 |

## 8. 当前阻塞项

| # | 阻塞项 | 影响 |
|---|---|---|
| 1 | conda env 未创建 | 无法执行任何测试 |
| 2 | CUDA 版本差异 | Driver 12.2 vs 官方 cudatoolkit 11.6，可能导致 PyTorch 1.12.1 不兼容 GPU |
| 3 | 无 | 数据已下载，代码已就位，脚本已创建 |

## 9. AutoDL 下一步执行顺序

```bash
# Step 0: 检查环境
bash /root/autodl-tmp/UWM_pushT/scripts/autodl_dp_00_check.sh

# Step 1: 创建环境
bash /root/autodl-tmp/UWM_pushT/scripts/autodl_dp_01_setup_env.sh 2>&1 | tee /root/autodl-tmp/UWM_pushT/logs/phase1_env_setup.log

# Step 2: Smoke test（依赖 Step 1 成功）
bash /root/autodl-tmp/UWM_pushT/scripts/autodl_dp_02_smoke_pusht.sh 2>&1 | tee /root/autodl-tmp/UWM_pushT/logs/phase1_pusht_smoke.log

# Step 3: 正式 baseline 训练（依赖 Step 2 通过）
conda activate dp-pusht
cd /root/autodl-tmp/UWM_pushT/diffusion_policy-main
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.seed=42 training.device=cuda:0 \
  logging.mode=disabled \
  hydra.run.dir='data/outputs/dp_pusht_baseline'
```
