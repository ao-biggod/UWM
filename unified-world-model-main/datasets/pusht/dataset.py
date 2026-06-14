import copy
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from datasets.utils.normalizer import LinearNormalizer, NestedDictLinearNormalizer
from datasets.utils.obs_utils import unflatten_obs


class PushTDataset(Dataset):
    def __init__(
        self,
        name: str,
        zarr_path: str,
        shape_meta: dict,
        seq_len: int = 19,
        val_ratio: float = 0.02,
        max_train_episodes: int = 90,
        seed: int = 42,
        split: str = "train",
        normalize_action: bool = True,
        normalize_lowdim: bool = True,
    ):
        self.name = name
        self.zarr_path = zarr_path
        self.seq_len = seq_len
        self.split = split

        # Parse shape_meta
        obs_shape_meta = shape_meta["obs"]
        self._image_shapes = {}
        self._lowdim_shapes = {}
        for key, attr in obs_shape_meta.items():
            obs_type = attr["type"]
            obs_shape = tuple(attr["shape"])
            if obs_type == "rgb":
                self._image_shapes[key] = obs_shape
            elif obs_type == "low_dim":
                self._lowdim_shapes[key] = obs_shape
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        self._action_shape = tuple(shape_meta["action"]["shape"])

        # Open zarr
        root = zarr.open(zarr_path, mode="r")
        self._zarr_img = root["data"]["img"]
        self._zarr_state = root["data"]["state"]
        self._zarr_action = root["data"]["action"]
        episode_ends = np.array(root["meta"]["episode_ends"])
        num_episodes = len(episode_ends)

        # Train/val split: use first max_train_episodes for train
        rng = np.random.default_rng(seed=seed)
        if num_episodes <= max_train_episodes:
            train_episodes = list(range(num_episodes))
        else:
            train_episodes = sorted(
                rng.choice(num_episodes, max_train_episodes, replace=False).tolist()
            )

        val_mask = np.ones(num_episodes, dtype=bool)
        val_mask[train_episodes] = False
        train_mask = np.zeros(num_episodes, dtype=bool)
        train_mask[train_episodes] = True

        if val_ratio > 0 and split == "train":
            # Carve out a val subset from training episodes
            num_val = max(1, round(val_ratio * len(train_episodes)))
            train_idx = np.where(train_mask)[0]
            val_from_train = sorted(
                rng.choice(train_idx, num_val, replace=False).tolist()
            )
            train_mask[val_from_train] = False
            val_mask[val_from_train] = True

        self._episode_ends = episode_ends
        self._train_mask = train_mask
        self._val_mask = val_mask

        # Build sequence indices
        self._indices = self._build_indices(train_mask if split == "train" else val_mask)

        # Normalizers
        if normalize_action:
            self.action_normalizer = self._init_action_normalizer()
        if normalize_lowdim:
            self.lowdim_normalizer = self._init_lowdim_normalizer()

    def _build_indices(self, episode_mask):
        indices = []
        episode_start = 0
        for i, episode_end in enumerate(self._episode_ends):
            if episode_mask[i]:
                for j in range(episode_start, episode_end + 1 - self.seq_len):
                    indices.append([j, j + self.seq_len])
            episode_start = episode_end
        return np.array(indices, dtype=np.int64)

    def _init_action_normalizer(self):
        # Compute stats over train episodes only
        train_indices = self._build_indices(self._train_mask)
        actions = []
        for start, end in train_indices:
            actions.append(self._zarr_action[start:end])
        actions = np.concatenate(actions, axis=0).reshape(-1, self._action_shape[0])
        min_val = actions.min(axis=0)
        max_val = actions.max(axis=0)
        scale = (max_val - min_val) / 2.0
        offset = (max_val + min_val) / 2.0
        return LinearNormalizer(scale, offset)

    def _init_lowdim_normalizer(self):
        # Compute stats over train episodes for each lowdim key
        train_indices = self._build_indices(self._train_mask)
        stats = {}
        for key in self._lowdim_shapes.keys():
            all_data = []
            for start, end in train_indices:
                if key == "agent_pos":
                    all_data.append(self._zarr_state[start:end, :2])
                else:
                    # For future lowdim keys
                    pass
            if all_data:
                data = np.concatenate(all_data, axis=0)
                min_val = data.min(axis=0)
                max_val = data.max(axis=0)
                scale = (max_val - min_val) / 2.0
                offset = (max_val + min_val) / 2.0
                stats[key] = (scale, offset)
        return NestedDictLinearNormalizer(stats)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        start, end = self._indices[idx]

        # Read image: float32 [0,255] -> uint8
        img = self._zarr_img[start:end]
        img = img.astype(np.uint8)
        assert img.shape == (self.seq_len, 96, 96, 3), f"img shape {img.shape}"

        # Read agent_pos from state[:, :2]
        agent_pos = np.array(self._zarr_state[start:end, :2], dtype=np.float32)
        assert agent_pos.shape == (self.seq_len, 2), f"agent_pos shape {agent_pos.shape}"

        # Read action
        action = np.array(self._zarr_action[start:end], dtype=np.float32)
        assert action.shape == (self.seq_len, 2), f"action shape {action.shape}"

        # Normalize lowdim
        if hasattr(self, "lowdim_normalizer") and "agent_pos" in self.lowdim_normalizer:
            agent_pos = self.lowdim_normalizer["agent_pos"](agent_pos)

        # Normalize action
        if hasattr(self, "action_normalizer"):
            action = self.action_normalizer(action)

        data = {
            "obs.image": torch.from_numpy(img),
            "obs.agent_pos": torch.from_numpy(agent_pos),
            "action": torch.from_numpy(action),
        }
        data = unflatten_obs(data)
        return data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.split = "val"
        val_set._indices = self._build_indices(self._val_mask)
        return val_set
