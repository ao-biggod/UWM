# UWM PushT Full Forward — Interface & Config Notes

## 1. UWM Model 接口确认

### UnifiedWorldModel.forward()
```python
forward(obs_dict, next_obs_dict, action, action_mask=None) -> (loss, info)
```
- `obs_dict`: `{'image': (B, 2, H, W, C) uint8, 'agent_pos': (B, 2, 2) float32}`
- `next_obs_dict`: 同上
- `action`: `(B, 16, 2) float32` — normalized to [-1, 1] by action_normalizer
- Returns: `(scalar_loss, {'loss', 'action_loss', 'dynamics_loss'})`

### UWMObservationEncoder
- `rgb_keys`: 自动从 shape_meta 检测 type=="rgb" 的 key → `['image']`
- `low_dim_keys`: 自动从 shape_meta 检测 type=="low_dim" 的 key → `['agent_pos']`
- Image 输入: `(B, T, H, W, C)` HWC **uint8**
- `ToTensor()` 在 transform pipeline 中将 HWC uint8 转为 CHW float [0,1]
- `use_low_dim: True` → 低维特征拼接到图像特征后
- `use_language: False` → 不使用 CLIP 文本编码
- `vision_backbone: "resnet"` → ResNet18 编码器
- VAE (SDXL-VAE) 将图像压缩到 `(B, V, C, T, pH, pW)` 潜在空间

### Normalizer
- `action_normalizer` (LinearNormalizer): 在 dataset `__getitem__` 中 normalize action 到 [-1,1]
- `lowdim_normalizer` (NestedDictLinearNormalizer): normalize agent_pos
- UWM train.py 将其保存在 checkpoint 中以备 eval/微调

### feats_dim() 计算
```
num_views * num_frames * embed_dim + use_low_dim * num_frames * low_dim_size
= 1 * 2 * 768 + True * 2 * 2 = 1536 + 4 = 1540
```

### latent_img_shape
通过 dummy forward 计算出: `(1, 4, 2, 12, 12)` — 1 view, 4 VAE channels, 2 frames, 12x12 spatial

## 2. PushT Config 决策

| 参数 | 值 | 原因 |
|------|-----|------|
| obs_num_frames | 2 | UWM/DP 标准 |
| action_len | 16 | UWM 默认 |
| seq_len | 19 | 2+16+2-1 |
| resize_shape | null | 96x96 无需缩放 |
| crop_shape | null | 96x96 被 8 整除，适合 VAE |
| random_crop | false | smoke test 不需要 |
| vision_backbone | resnet | 轻量，适合 smoke test |
| use_low_dim | true | PushT 有 agent_pos |
| use_language | false | PushT 无语言指令 |
| batch_size | 1 | smoke test 显存保护 |

## 3. 图像值域确认

- zarr `data/img`: float32, min=65, max=255
- PushTDataset 转换: `img.astype(np.uint8)` → 正确
- UWM `ToTensor()` 处理: `.float().div_(255.0)` → [0,1] float

## 4. 注意

- `HF_ENDPOINT=https://hf-mirror.com` 需要在下载 SDXL-VAE 时设置（国内网络限制）
- diffusers 升级到 0.36.0（原 0.11.1 无法下载 SDXL VAE）
- accelerate 升级到 1.10.1（配合新 diffusers）
- transformers 升级到 4.57.6
