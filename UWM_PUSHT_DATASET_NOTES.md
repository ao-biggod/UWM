# UWM PushT Dataset — Interface & Design Notes

## 1. UWM Dataset 接口确认

### 必须满足的接口

| 要求 | 来源 |
|------|------|
| `instantiate(config.dataset)` 返回 `(train_set, val_set)` | `train.py:228` |
| `train_set` 必须有 `action_normalizer` 属性 | `train.py:286, 297` |
| `train_set` 必须有 `lowdim_normalizer` 属性 | `train.py:298` |
| `__getitem__` 返回 flat-key dict: `{"obs.image": ..., "obs.agent_pos": ..., "action": ...}` | `RobomimicDataset:137-146`, `unflatten_obs()` |
| `unflatten_obs()` 将 `obs.xxx` 转为嵌套 `{"obs": {"xxx": ...}}` | `datasets/utils/obs_utils.py` |
| image 保持 HWC uint8 | `RobomimicDataset:83` `metadata["obs.image"] = {"shape": ..., "dtype": np.uint8}` |
| low_dim 保持 float32 | `RobomimicDataset:85` |
| `__len__` 返回有效序列窗口数 | `TrajectorySampler` |
| 序列不跨 episode | `TrajectorySampler.sample_sequence` in-episode 切分 |
| `get_validation_dataset()` 返回 val copy | `RobomimicDataset:148-152` |

### Normalizer 要求

- `action_normalizer`: `LinearNormalizer(scale, offset)` — 在 `__getitem__` 时 normalize action 到 [-1, 1]
- `lowdim_normalizer`: `NestedDictLinearNormalizer(stats)` — 在 `__getitem__` 时 normalize lowdim obs
- UWM train.py 在 save/load checkpoint 时使用这些 normalizer
- 本阶段需要实现 normalizer，smoke test 不依赖它们

### Batch 格式流程

```
dataset[idx] → {"obs.image": T,H,W,C uint8, "obs.agent_pos": T,2 float32, "action": T,2 float32}
     ↓ (dataloader collate → stack to batch)
batch before unflatten:
  {"obs.image": B,T,H,W,C, "obs.agent_pos": B,T,2, "action": B,T,2}
     ↓ (unflatten_obs)
batch after unflatten:
  {"obs": {"image": B,T,H,W,C, "agent_pos": B,T,2}, "action": B,T,2}
     ↓ (process_batch)
curr_obs: {"image": B,2,H,W,C, "agent_pos": B,2,2}
next_obs: {"image": B,2,H,W,C, "agent_pos": B,2,2}
actions:  B,16,2
```

## 2. PushT Zarr 结构

| key | shape | dtype | 说明 |
|-----|-------|-------|------|
| data/img | (25650, 96, 96, 3) | float32 | RGB 图像, 值域 [0, 255] |
| data/state | (25650, 5) | float32 | state[:2] = agent_pos |
| data/action | (25650, 2) | float32 | 动作 |
| meta/episode_ends | (206,) | int64 | episode 边界索引 |
| 总步数 | 25650 | | |
| episode 数 | 206 | | |

## 3. 设计决策

### 不使用 CompressedTrajectoryBuffer
- DP zarr 数据量小（25650 steps），直接内存读取即可
- 避免引入 buffer 序列化/反序列化的复杂度
- 本阶段只做 smoke test，不需要 buffer 的压缩优势

### 图像处理
- zarr 中 img 是 float32 [0, 255]
- 转换为 uint8: `(img).astype(np.uint8)`
- 保持 HWC 格式

### agent_pos 提取
- `state[:, :2]` 作为 agent_pos
- state 的 [2:5] 是 block pose，本阶段不加入

### seq_len = 19
- obs_num_frames = 2, action_len = 16, next_obs_num_frames = 2
- seq_len = 2 + 16 + 2 - 1 = 19

### Normalizer
- 本阶段按需实现最小兼容（stats 计算但不一定在 smoke test 中使用）
- action_normalizer: LinearNormalizer
- lowdim_normalizer: NestedDictLinearNormalizer
