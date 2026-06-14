# UWM PushT Training — Notes

## 1. train.py 入口分析

- **入口**: `experiments/uwm/train.py`
- **Hydra config name**: `train_uwm.yaml` (可通过 `--config-name` 覆盖)
- **config_path**: `../../configs`
- **入口函数**: `main()` → `mp.spawn(train, nprocs=world_size)` 其中 `world_size = torch.cuda.device_count()`

### 核心流程

```
hydra.main(config_path, config_name) → main()
  → OmegaConf.resolve(config)
  → mp.spawn(train, nprocs=world_size)
    → set_seed(seed * world_size + rank)
    → init_distributed(rank, world_size)
    → init_wandb(config)  # main process only
    → instantiate(config.dataset) → (train_set, val_set)
    → make_distributed_data_loader()
    → instantiate(config.model)
    → AdamW optimizer
    → get_scheduler("constant")
    → GradScaler(use_amp)
    → DDP(model)
    → Training loop:
        → process_batch(batch, obs_horizon, action_horizon, device)
        → loss, info = model(curr_obs, next_obs, actions)
        → scaler.scale(loss).backward()
        → scaler.step(optimizer), scaler.update()
        → scheduler.step()
        → maybe_evaluate (if step % eval_every == 0)
        → maybe_save_checkpoint (if step % save_every == 0)
```

### Normalizer 绑定
- `train_set.action_normalizer` 用于 eval 时 unnormalize actions（forward 训练不需要）
- `train_set.lowdim_normalizer` 保存到 checkpoint
- Normalizer 在 dataset `__getitem__` 中应用

### 为何 smoke test 不用真实 train.py
- `step=0` 时 `0 % eval_every == 0` 恒为 True，触发 eval
- eval 遍历 12667 val 样本 + VAE decode + 多采样方法 → 极慢
- Smoke script 复刻了 train.py 的核心训练逻辑（process_batch → forward → backward → optimizer.step），跳过 eval/wandb/distributed

## 2. Config 字段确认

train.py 需要的所有字段（通过 Hydra 默认值 + 覆盖提供）:

| 字段 | 来源 | 值 |
|------|------|---|
| seed | template_train | 42 |
| num_steps | template_train → CLI override | 10 |
| batch_size | train_uwm_pusht | 1 |
| obs_num_frames | template_train | 2 |
| num_frames | template_train | 19 |
| optimizer | template_train | lr=1e-4, AdamW |
| scheduler | template_train | constant |
| use_amp | template_train | False |
| eval_every | template_train → CLI override | 999999 |
| save_every | template_train → CLI override | 999999 |
| eval_task_name | template_train | pusht_image |
| pretrain_checkpoint_path | template_train | null |
| clip_grad_norm | template_train | null |
| resume | template_train | False |

## 3. Smoke Test 对比

| 指标 | Phase 3 (forward only) | Phase 4 (10-step train) |
|------|----------------------|------------------------|
| loss | 2.17 | 1.26 ~ 1.98 (mean 1.70) |
| action_loss | 1.19 | 0.32 ~ 0.94 (decreasing) |
| dynamics_loss | 0.98 | 0.94 ~ 1.10 (stable) |
| GPU mem | ~5GB | 4.47 GB |
| backward | No | Yes |
| optimizer | No | Yes (AdamW) |
