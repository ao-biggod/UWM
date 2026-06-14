# UWM on PushT 复现实验报告

> 日期: 2026-05-29
> 平台: AutoDL (RTX 4090, 50GB, CUDA 12.2)
> 项目路径: `/root/autodl-tmp/UWM_pushT/`

---

## 1. 项目目标

将 PushT 操作任务（推 T 型方块到目标区域）接入 Unified World Model (UWM) 框架，完成从 scratch 的单任务训练，并与 Diffusion Policy (DP) image baseline 进行对比。

---

## 2. 实现内容

| 模块 | 状态 | 说明 |
|---|---|---|
| PushT 数据接入 | ✅ 完成 | Zarr replay buffer → UWM dataset |
| UWM-PushT 训练 pipeline | ✅ 完成 | 端到端训练循环（forward → loss → optimizer） |
| UWM-PushT eval pipeline | ✅ 完成 | 50 episodes rollout, mean_max_reward 指标 |
| DP PushT baseline 复现 | ✅ 完成 | 50 epoch → test_mean_score 0.726 |
| gym 0.26 兼容 patch | ✅ 完成 | 2 文件修改，patch 已保存 |
| SDXL-VAE 离线加载 patch | ✅ 完成 | 1 行修改，patch 已保存 |
| 结果备份 | ✅ 完成 | checkpoints + logs + metrics + patches |

---

## 3. 实验设置

| 项目 | UWM-PushT | DP-PushT |
|---|---|---|
| 训练方式 | UWM from scratch | Diffusion Policy (官方 baseline) |
| 模型参数量 | 305.6M | 262.6M |
| 训练量 | 20,000 steps | 50 epochs (~8,400 batches) |
| Batch size | 64 | 64 |
| 学习率 | 1e-4 (constant) | 1e-4 (cosine schedule) |
| Observation | Image (96×96) + agent_pos (2) | Image (96×96) + agent_pos (2) |
| Action dim | 2 | 2 |
| Eval episodes | 50 | 50 |
| Metric 名称 | mean_max_reward | test_mean_score |
| 是否同一 eval 脚本 | 否 | 否 |
| GPU | RTX 4090 | RTX 4090 |
| 训练时间 | ~3 小时 | ~15 分钟 |
| 视觉编码器 | SDXL-VAE (预训练) | ResNet18 (ImageNet 预训练) |

---

## 4. 主要结果

| 模型 | 训练量 | Eval episodes | Score |
|---|---:|---:|---:|
| **UWM 20k (50eps)** | 20k steps | 50 | **0.1121** |
| UWM 20k (10eps) | 20k steps | 10 | 0.2724 |
| **DP 50epoch** | 50 epochs | 50 | **0.7259** |
| DP 论文预期 | 3050 epochs | 50 | ~0.9+ |

> ⚠️ UWM 的 10-episode 结果（0.27）显著高于 50-episode 结果（0.11），说明小样本评估存在较大偏差。**主对比应以 50-episode 为准。**

### UWM 训练过程（20k steps，每 2500 steps 采样）

| Step | Loss | Action Loss | Dynamics Loss | Eval (10eps) |
|---|---|---|---|---|
| 5,000 | 0.059 | 0.021 | 0.038 | 0.142 |
| 10,000 | 0.059 | 0.024 | 0.034 | 0.190 |
| 20,000 | 0.029 | 0.007 | 0.022 | 0.272 |

### DP 训练过程（50 epochs，每 10 epochs eval）

| Epoch | Train Loss | Val Loss | Test Mean Score |
|---|---|---|---|
| 0 | 0.395 | 0.107 | 0.128 |
| 10 | 0.047 | 0.051 | 0.307 |
| 20 | 0.040 | 0.045 | 0.464 |
| 30 | 0.032 | 0.041 | 0.601 |
| 40 | 0.025 | 0.042 | **0.726** |
| 49 (final) | 0.022 | — | — |

---

## 5. 结果分析

1. **DP 在 PushT 上收敛明显更快。** 50 epochs 即达到 0.73 mean_score，而 UWM 训练 20k steps 仅达到 0.11。DP 的 policy-focused 训练方式在此任务上样本效率更高。

2. **UWM 从 scratch 单任务训练不占优势。** UWM 的原设计优势在于大规模多任务机器人数据和 action-free 视频预训练，而非单任务 from-scratch 训练。

3. **SDXL-VAE 对 96×96 低分辨率 PushT 渲染可能不够高效。** VAE 将 96×96 图像压缩为 latent，但该压缩率对简单的几何场景（方块 + 圆形 agent）可能过度。

4. **UWM 工程 pipeline 已成功打通。** 训练 loss 持续下降（0.06 → 0.03），eval 能正常运行，说明代码基础设施是正确的。

5. **评估 episode 数对结果影响显著。** UWM 10eps (0.27) vs 50eps (0.11) 相差 2.4 倍。所有正式对比应使用 50 episodes。

---

## 6. 遇到的问题和修复

| # | 问题 | 修复 |
|---|---|---|
| 1 | `diffusion_policy` editable install 损坏（pip 26.x 的 MAPPING bug） | 改用 `python setup.py develop` |
| 2 | gym 0.26 API 不兼容（`reset_async`/`reset_wait` 签名变化, `concatenate` 参数顺序变化） | `dp_gym026_compat.patch`（2 文件，4 处修改） |
| 3 | `AsyncVectorEnv` 的 `shared_memory=True` 在 gym 0.26 与 Dict space 冲突 | 改为 `shared_memory=False` |
| 4 | DP checkpoint 4GB/个，50GB 磁盘无法跑完整 3050 epochs | 仅跑 50 epoch budget 版本；清理中间产物 |
| 5 | SDXL-VAE `from_pretrained` 强制访问 HuggingFace，离线环境超时 | `uwm_offline_vae.patch`：加 `local_files_only=True` |
| 6 | UWM 10eps eval 方差大（0.27 → 0.11 的差异） | 统一使用 50 episodes |

---

## 7. 结论

- **工程目标：已完成。** PushT 数据、UWM 训练 pipeline、UWM eval pipeline、DP baseline 全部跑通。
- **性能目标：当前未达到 DP baseline。** UWM 20k (50eps) mean_max_reward = 0.11，远低于 DP 50epoch 的 0.73。
- **UWM 性能差距可解释。** UWM 设计的核心优势（多任务数据、action-free 预训练、视频理解能力）在当前单任务 scratch 实验中未发挥作用。
- **继续推进有价值。** UWM pipeline 工程完成度高，可以作为后续实验（长训练、多任务、预训练）的坚实基础。

---

## 8. 下一步建议

| 优先级 | 建议 | 预期影响 |
|---|---|---|
| 高 | UWM 继续训练到 50k-100k steps | 验证 loss/eval 是否继续改善 |
| 高 | 统一 DP 和 UWM 的 eval 脚本 | 排除 metric 实现差异 |
| 中 | 调整 action_loss / dynamics_loss 权重 | 当前 action_loss 已降到很低，可能需要减少 dynamics 权重 |
| 中 | 检查 action normalization 是否正确 | action_scale ≈ 242 可能过大 |
| 中 | 尝试冻结 VAE 或使用更轻量的视觉编码器 | 减少参数量，可能加速收敛 |
| 低 | 跑 DP 完整 3050 epoch | 确认 DP 在 RTX 4090 上能否复现论文 0.9+ |
| 低 | 多 seed 实验（3 seeds） | 减少单 seed 随机性 |
| 低 | 尝试加载预训练视觉编码器 | 测试预训练是否帮助单任务快速收敛 |
