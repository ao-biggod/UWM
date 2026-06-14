# AutoDL 复现 Diffusion Policy PushT Baseline

## 推荐执行顺序

```bash
# 0. 上传项目到 AutoDL 后，进入项目目录
cd /root/autodl-tmp/UWM_pushT

# 1. 创建 conda 环境
cd diffusion_policy-main
conda env create -f conda_environment.yaml
conda activate robodiff

# 2. 安装 diffusion_policy
pip install -e .

# 3. 下载 PushT 数据
cd /root/autodl-tmp/UWM_pushT
bash scripts/download_pusht_data.sh

# 4. 一键环境检查（含 dataset/env/train 三个 smoke test）
bash scripts/check_autodl_dp_env.sh

# 5. 正式训练
cd diffusion_policy-main
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn \
  training.seed=42 training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

---

## 1. AutoDL 推荐配置

| 项目 | 推荐 | 备注 |
|------|------|------|
| GPU | RTX 3090 / RTX 4090 / A5000 | A100 也兼容，但非必需 |
| 最低显存 | 16GB (smoke test) | 正式训练推荐 24GB |
| 系统 | Ubuntu 20.04 / 22.04 | |
| Python | 3.9 (via conda) | conda_environment.yaml 指定 |
| CUDA | 11.6+ | conda 环境自带 cudatoolkit=11.6 |
| 数据盘 | `/root/autodl-tmp/` (AutoDL 默认) | 或 `/root/autodl-fs/` |

**重要**：以下路径中的 `/root/autodl-tmp/UWM_pushT` 请替换为实际的 AutoDL 数据盘路径。

## 2. 目录结构

```text
/root/autodl-tmp/UWM_pushT/
├── diffusion_policy-main/            # Diffusion Policy 仓库
│   ├── data/
│   │   └── pusht/
│   │       └── pusht_cchi_v7_replay.zarr   # PushT 训练数据（由 download 脚本放置）
│   ├── scripts/
│   │   ├── smoke_pusht_dataset.py     # Dataset smoke test
│   │   ├── smoke_pusht_env.py         # Env smoke test
│   │   └── smoke_pusht_train.py       # 10-step training smoke test
│   ├── outputs/                       # （训练输出）
│   └── ...
├── unified-world-model-main/          # UWM 仓库（不在 Phase 1 中使用）
├── scripts/
│   ├── download_pusht_data.sh         # 数据下载脚本
│   └── check_autodl_dp_env.sh         # 一键环境检查
└── AUTODL_DP_BASELINE.md             # 本文档
```

## 3. 环境安装

```bash
# 基础工具
sudo apt-get update && sudo apt-get install -y wget unzip git

# 创建 conda 环境
cd /root/autodl-tmp/UWM_pushT/diffusion_policy-main
conda env create -f conda_environment.yaml
conda activate robodiff

# 安装 diffusion_policy
pip install -e .

# 验证
python -c "
import torch; import hydra; import zarr; import gym
import pygame; import pymunk; import wandb; import cv2; import numpy as np
print('All imports OK')
"

python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

### 常见安装问题

| 问题 | 排查方法 |
|------|----------|
| `mujoco_py` 安装失败 | `sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf` |
| `pygame` 无 display | `pip install pygame==2.1.2`；headless 用 `export SDL_VIDEODRIVER=dummy` |
| `pymunk` 报错 | `pip install pymunk==6.2.1` |
| `cv2` 找不到 | `pip install opencv-python==4.6.0.66` |
| `zarr` 版本冲突 | 确保 `zarr==2.12.0`, `numcodecs==0.10.2` |
| `imagecodecs` 冲突 | `pip install imagecodecs==2022.9.26` |
| `diffusers` 版本 | 必须 `0.11.1`，新版 API 不兼容 |
| `wandb` 登录 | `wandb login` 或 `export WANDB_MODE=offline` |

## 4. 数据下载

**方式 A：使用脚本（推荐）**

```bash
cd /root/autodl-tmp/UWM_pushT
bash scripts/download_pusht_data.sh
```

**方式 B：手动下载**

```bash
cd /root/autodl-tmp/UWM_pushT/diffusion_policy-main
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip
cd ..
```

**下载后应看到：**

```text
diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr/
├── data/
│   ├── img/            # (N, 96, 96, 3) uint8
│   ├── state/          # (N, 5)
│   ├── action/         # (N, 2)
│   ├── keypoint/       # (N, 9, 2)
│   └── n_contacts/     # (N, 1)
└── meta/
    └── episode_ends    # episode boundary indices
```

> **路径约定**：所有 smoke 脚本和数据读取默认使用 `data/pusht/pusht_cchi_v7_replay.zarr`（相对于 `diffusion_policy-main/`）。这与官方 config `image_pusht_diffusion_policy_cnn.yaml` 中 `zarr_path: data/pusht/pusht_cchi_v7_replay.zarr` 一致。

## 5. Smoke Test 脚本

三个 smoke test 脚本位于 `diffusion_policy-main/scripts/`，均从 `diffusion_policy-main/` 目录下运行。

### 5.1 Dataset Smoke Test

```bash
cd /root/autodl-tmp/UWM_pushT/diffusion_policy-main
python scripts/smoke_pusht_dataset.py
```

验证：zarr 加载 → PushTImageDataset 创建 → sample keys/shape/dtype/range → validation split。

期望最后一行：`SMOKE TEST PASSED`

### 5.2 Env Smoke Test

```bash
cd /root/autodl-tmp/UWM_pushT/diffusion_policy-main
export SDL_VIDEODRIVER=dummy    # headless 模式
python scripts/smoke_pusht_env.py
```

验证：PushTImageEnv 创建 → reset → 10 step random actions → render。

期望最后一行：`ENV SMOKE TEST PASSED`

### 5.3 Training Smoke Test (纯 forward/backward，不依赖 workspace)

```bash
cd /root/autodl-tmp/UWM_pushT/diffusion_policy-main
python scripts/smoke_pusht_train.py --device cuda:0 --num-steps 10
```

验证：
- 加载真实的 `PushTImageDataset` + `DiffusionUnetHybridImagePolicy`
- 10 步 forward → backward → optimizer.step()
- loss 有限且正数
- 不调用 wandb、不保存 checkpoint、不做 rollout

支持参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--zarr-path` | `data/pusht/pusht_cchi_v7_replay.zarr` | 数据路径 |
| `--device` | `cuda:0` | 设备 |
| `--num-steps` | `10` | 训练步数 |
| `--batch-size` | `16` | 批量大小 |

期望最后一行：`TRAINING SMOKE TEST PASSED`

## 6. Hydra 训练 Smoke Test（含 env rollout）

如果上面的 train smoke (纯 forward/backward) 通过，可以再跑 workspace 版本（需要 env 可用）：

```bash
cd /root/autodl-tmp/UWM_pushT/diffusion_policy-main

python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn \
  training.num_epochs=1 \
  training.max_train_steps=10 \
  training.max_val_steps=2 \
  training.rollout_every=50 \
  training.checkpoint_every=50 \
  training.val_every=1 \
  training.sample_every=50 \
  logging.mode=offline \
  hydra.run.dir=outputs/dp_pusht/smoke
```

> **注意**：epoch 0 恒触发 rollout（`0 % N == 0`），因此这个测试要求 PushT env 正常创建。

**字段名来源确认（实测自 config 和 workspace 代码）：**

| 覆盖字段 | 代码位置 |
|----------|----------|
| `training.num_epochs` | `workspace.py:153` |
| `training.max_train_steps` | `workspace.py:198-200` |
| `training.max_val_steps` | `workspace.py:229-231` |
| `training.rollout_every` | `workspace.py:214` |
| `training.checkpoint_every` | `workspace.py:257` |
| `training.val_every` | `workspace.py:220` |
| `training.sample_every` | `workspace.py:238` |
| `logging.mode` | `workspace.py:114` `wandb.init(**cfg.logging)` |
| `hydra.run.dir` | Hydra 内置 resolver |

## 7. 正式 Baseline 训练命令

```bash
cd /root/autodl-tmp/UWM_pushT/diffusion_policy-main

# 使用 wandb online
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# 或关闭 wandb (离线模式)
python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn \
  training.seed=42 \
  training.device=cuda:0 \
  logging.mode=offline \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

### 训练参数摘要（来自 `image_pusht_diffusion_policy_cnn.yaml`）

| 参数 | 值 | config 路径 |
|------|-----|------------|
| batch_size | 64 | `dataloader.batch_size` |
| horizon | 16 | `policy.horizon` |
| n_obs_steps | 2 | `n_obs_steps` |
| n_action_steps | 8 | `n_action_steps` |
| num_epochs | 3050 | `training.num_epochs` |
| learning_rate | 1e-4 | `optimizer.lr` |
| lr_scheduler | cosine | `training.lr_scheduler` |
| lr_warmup_steps | 500 | `training.lr_warmup_steps` |
| use_ema | true | `training.use_ema` |
| rollout_every | 50 epochs | `training.rollout_every` |
| checkpoint_every | 50 epochs | `training.checkpoint_every` |
| val_every | 1 epoch | `training.val_every` |
| seed | 42 | `training.seed` (通过 CLI 覆盖) |

### 预计资源消耗

| 指标 | 估计值 |
|------|--------|
| 单 epoch | ~10-30 秒 |
| 完整训练 | ~8-24 小时 (RTX 3090) |
| GPU 显存 | ~8-12 GB |
| 磁盘 (checkpoints) | ~2-5 GB |
| 数据大小 | ~130 MB |

## 8. 评估命令

```bash
cd /root/autodl-tmp/UWM_pushT/diffusion_policy-main

python eval.py \
  --checkpoint data/outputs/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt \
  --output_dir outputs/dp_pusht/eval \
  --device cuda:0
```

`eval.py:48-51` 工作机制：
1. 从 checkpoint 恢复 workspace（含 config、model、ema_model）
2. `instantiate(cfg.task.env_runner, output_dir=...)` 创建 eval runner
3. `env_runner.run(policy)` 执行 rollout
4. 结果写入 `eval_log.json`，视频写入 output_dir

主要指标（来自 `pusht_image_runner.py:245-249`）：
- `test/mean_score`：所有 test seed 的 max_reward 均值
- `sim_max_reward_{seed}`：每个 seed 的最大 reward
- `sim_video_{seed}`：rollout 视频

## 9. 一键环境检查

```bash
cd /root/autodl-tmp/UWM_pushT
bash scripts/check_autodl_dp_env.sh
```

此脚本检查 8 项：系统信息 → GPU → conda → Python/torch → 数据文件 → dataset smoke → env smoke → train smoke。

数据不存在时会提示 `bash scripts/download_pusht_data.sh`，不会自动下载。

## 10. 复现检查清单

- [ ] conda 环境 `robodiff` 创建成功
- [ ] `pip install -e .` 无报错
- [ ] `torch.cuda.is_available() == True`
- [ ] `data/pusht/pusht_cchi_v7_replay.zarr` 存在
- [ ] `scripts/smoke_pusht_dataset.py` 通过
- [ ] `scripts/smoke_pusht_env.py` 通过
- [ ] `scripts/smoke_pusht_train.py` 通过（10 steps, loss finite）
- [ ] 正式训练启动，`test/mean_score` 在日志中可见
- [ ] `test/mean_score` > 0.8（约 500 epochs 后）
- [ ] 最终 `test/mean_score` ~0.95（paper reported）
