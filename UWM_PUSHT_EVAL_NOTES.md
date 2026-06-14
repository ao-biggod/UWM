# UWM PushT Eval — Interface & Pipeline Notes

## 1. UWM 推理接口确认

### model.sample(obs_dict) → action
- 入口: `UnifiedWorldModel.sample()` (uwm.py:322)
- 实际调用: `self.sample_marginal_action(obs_dict)` (uwm.py:396-417)
- obs_dict: `{'image': (1, 2, 96, 96, 3) uint8, 'agent_pos': (1, 2, 2) float32}`
- 输出: `(1, action_len, action_dim)` = `(1, 16, 2)` **normalized** float32
- 推理流程:
  1. `encode_curr_obs(obs_dict)` → curr_feats
  2. 初始化 random action + next_obs latents
  3. DDIM denoising with `num_inference_steps=10`
  4. 返回 action sample

### Action unnormalize
- `model.sample()` 返回 normalized action ([-1, 1])
- 需要 unnormalize: `action * action_scale + action_offset`
- PushT env 期望 action 在原始空间 [0, 512]

### eval_robomimic.py 推理模式
```python
obs_tensor = {k: torch.tensor(v, device=device)[None] for k, v in obs.items()}
action = model.sample(obs_tensor)[0].cpu().numpy()  # (action_len, action_dim)
obs, reward, done, info = env.step(action)
```

### eval_droid.py 推理模式 (receding horizon)
```python
act_seq = model.sample(obs_seq)  # (1, 16, 7)
act_seq = act_seq * action_scale + action_offset  # unnormalize
# Store in buffer, pop one at a time
action = act_buffer.popleft()
```

## 2. PushT Env 接口

### PushTImageEnv (from DP)
- `env.reset()` → `{'image': (2, 3, 96, 96), 'agent_pos': (2, 2)}` — note: CHW, float32 [0,1]
- `env.step(action)` where action is `(8, 2)` — 8-step action in [0, 512]
- Returns: `(obs, reward, done, info)`
- Env is wrapped in `MultiStepWrapper(VideoRecordingWrapper(PushTImageEnv))`

### Obs conversion needed
- UWM expects: HWC uint8 `(T, H, W, C)` 
- DP env outputs: CHW float32 [0,1] `(T, C, H, W)`
- Need to: `(img * 255).astype(np.uint8).transpose(0, 2, 3, 1)` to get HWC uint8

### Action conversion needed
- UWM outputs: normalized [-1, 1] `(16, 2)`
- Need to: unnormalize to [0, 512], take first `n_action_steps=8`
- PushT env expects: raw [0, 512] `(8, 2)`

## 3. Receding Horizon

```
episode:
  reset → obs (2 frames)
  loop:
    stack obs → [2, H, W, C]
    UWM.sample() → action [16, 2]
    unnormalize → raw_action [16, 2]
    take first 8 steps → exec_action [8, 2]
    env.step(exec_action) → new obs [2, H, W, C]
    repeat
```

## 4. Smoke Test Results (Phase 5)

### Phase 5a: Policy Inference
- `model.sample(obs_dict)` → normalized action `[1, 16, 2]` in [-1, 1]
- Unnormalized: `action * action_scale + action_offset` → [12, 511] within [0, 512]
- PASS

### Phase 5b: Env Eval
- PushTImageEnv + MultiStepWrapper works
- Receding horizon control loop executes
- Random init model produces valid actions (range [12, 511])
- 2 episodes × 50 steps completed
- PASS

### Phase 5c: eval_pusht.py
- Supports `--random-init` and `--checkpoint` modes
- Saves `eval_log.json` with per-episode stats
- PASS

## 5. 模板参考

| 概念 | 参考 |
|------|------|
| 推理 loop | `eval_robomimic.py:37-58` |
| Receding horizon | `eval_droid.py` DroidPolicy class |
| Normalizer | `model.action_normalizer` (LinearNormalizer) |
| Env | `diffusion_policy.env.pusht.pusht_image_env.PushTImageEnv` |
| Env runner | `diffusion_policy.env_runner.pusht_image_runner.PushTImageRunner` |
