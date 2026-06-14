# Patch Notes

## Phase 1: DP PushT Baseline

### 修改: `diffusion_policy-main/diffusion_policy/common/robomimic_config_util.py`
- robomimic 顶层 import → 惰性导入
- 原因: robomimic mujoco 依赖无法编译，DP PushT 不需要此功能

### 修改: `diffusion_policy-main/scripts/smoke_pusht_train.py`
- 添加 `model.normalizer.to(device)` 修复 CPU/CUDA 不匹配

## Phase 2: UWM PushT Dataset

### 新增: `unified-world-model-main/datasets/__init__.py`
- 空文件，使 datasets/ 成为 regular package
- 原因: 与 HF `datasets` 包名冲突

### 新增: `unified-world-model-main/datasets/pusht/__init__.py`
- `make_pusht_dataset()` 工厂函数

### 新增: `unified-world-model-main/datasets/pusht/dataset.py`
- `PushTDataset` 类，直接读取 DP zarr

### 新增: `unified-world-model-main/configs/dataset/pusht_image.yaml`
- PushT dataset Hydra 配置

### 新增: `scripts/smoke_uwm_pusht_dataset.py`
- 7-step dataset batch shape smoke test

## Phase 3: UWM Full Forward Smoke

### 新增: `unified-world-model-main/configs/train_uwm_pusht.yaml`
- UWM PushT Hydra 训练入口 config

### 新增: `scripts/smoke_uwm_pusht_forward.py`
- Full forward pass smoke test

### 依赖升级
- diffusers: 0.11.1 → 0.36.0 (SDXL VAE 下载需要)
- accelerate: 0.13.2 → 1.10.1 (配合新 diffusers)
- transformers: 4.25.0 → 4.57.6
- huggingface_hub: 0.19.4 → 0.36.2
- 新增 timm (ResNet backbone)
- VAE 下载需 `HF_ENDPOINT=https://hf-mirror.com`

## Phase 4: UWM 10-Step Training Smoke

### 新增: `scripts/smoke_uwm_pusht_train.py`
- 10-step forward/backward/optimizer smoke test
- 不复用 train.py (step=0 恒触发 eval)

## Phase 5: Policy Inference + Env Eval

### 新增: `scripts/smoke_uwm_pusht_policy_infer.py`
- `model.sample(obs_dict)` → normalized action [B,16,2] 验证

### 新增: `scripts/smoke_uwm_pusht_env_eval.py`
- PushTImageEnv + UWM policy inference 端到端验证
- Receding horizon 控制循环

### 新增: `unified-world-model-main/experiments/uwm/eval_pusht.py`
- 正式评估脚本，支持 --random-init 和 --checkpoint
- 保存 eval_log.json

### 关键事实
- `model.sample(obs_dict)` 输出 normalized action [B,16,2] in [-1,1]
- eval 时需 unnormalize: `action_raw = action_norm * scale + offset`
- PushT eval 只执行前 8 步 action (receding horizon)
